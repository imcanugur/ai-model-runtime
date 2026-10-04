package runtime_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	internalHTTP "github.com/imcanugur/ai-model-runtime/internal/http"
	storeMemory "github.com/imcanugur/ai-model-runtime/internal/jobstore/memory"
	"github.com/imcanugur/ai-model-runtime/internal/pipeline"
	queueMemory "github.com/imcanugur/ai-model-runtime/internal/queue/memory"
	"github.com/imcanugur/ai-model-runtime/internal/registry"
	"github.com/imcanugur/ai-model-runtime/internal/runtime"
	runtimeMock "github.com/imcanugur/ai-model-runtime/internal/runtime/mock"
	"github.com/imcanugur/ai-model-runtime/internal/service"
)

// TestUniversalZeroCodeModelAddition tests that a brand-new model can be placed on disk,
// discovered, validated, loaded into READY state, and executed through generic pipeline stages
// without ANY changes to Go core source code.
func TestUniversalZeroCodeModelAddition(t *testing.T) {
	tempModelsDir := t.TempDir()
	newModelDir := filepath.Join(tempModelsDir, "zero-code-model")
	if err := os.MkdirAll(newModelDir, 0755); err != nil {
		t.Fatalf("failed to create model dir: %v", err)
	}

	dummyWeightsPath := filepath.Join(newModelDir, "model.bin")
	if err := os.WriteFile(dummyWeightsPath, []byte("DUMMY_ZERO_CODE_WEIGHTS"), 0644); err != nil {
		t.Fatalf("failed to write weights: %v", err)
	}

	manifestYAML := `schema_version: "v1"
id: "zero-code-model"
name: "Zero Code Dynamic Model"
version: "2.1.0"
metadata:
  languages: ["en"]
  task: "custom-classification"
  framework: "mock"
resources:
  max_concurrency: 4
  timeout_ms: 5000
runtime:
  provider: "mock"
  profile: "standard_cpu"
  entrypoint: "model.bin"
pipeline:
  stages:
    - name: "normalizer"
      type: "text_normalizer"
    - name: "inference"
      type: "runtime_infer"
    - name: "calibrator"
      type: "threshold_calibrator"
      config:
        threshold: 0.5
`
	if err := os.WriteFile(filepath.Join(newModelDir, "model.yaml"), []byte(manifestYAML), 0644); err != nil {
		t.Fatalf("failed to write model.yaml: %v", err)
	}

	rtFactory := runtime.NewDefaultFactory()
	_ = rtFactory.RegisterProvider("mock", runtimeMock.NewProvider())

	stageReg := pipeline.NewDefaultStageRegistry()
	modelReg := registry.NewDefaultModelRegistry(stageReg, registry.AdaptRuntimeFactory(rtFactory))

	ctx := context.Background()

	// 1. Discover models
	if err := modelReg.Discover(ctx, tempModelsDir); err != nil {
		t.Fatalf("discovery failed: %v", err)
	}

	// 2. Load model
	key := domain.NewModelKey("zero-code-model", "2.1.0")
	if err := modelReg.Load(ctx, key); err != nil {
		t.Fatalf("failed to load zero-code-model: %v", err)
	}
	defer func() { _ = modelReg.Unload(ctx, key) }()

	// 3. Acquire handle and execute
	handle, err := modelReg.Acquire(key)
	if err != nil {
		t.Fatalf("failed to acquire handle: %v", err)
	}
	defer handle.Release()

	// Execute with universal input
	infInput := domain.InferenceInput{
		Documents: []domain.InputDocument{
			{ID: "doc-1", Text: "Testing zero-code dynamic model execution.", Language: "en"},
		},
	}

	res, err := handle.Execute(ctx, infInput.ToRawInput())
	if err != nil {
		t.Fatalf("execution failed: %v", err)
	}

	if res.Model.ID != "zero-code-model" {
		t.Errorf("expected model.id 'zero-code-model', got %q", res.Model.ID)
	}
	valMap, ok := res.Output.Value.(map[string]any)
	if !ok {
		t.Fatalf("expected map[string]any in Output.Value, got %T (%+v)", res.Output.Value, res.Output.Value)
	}
	if valMap["label"] == "" {
		t.Errorf("expected non-empty output label")
	}
	if res.Execution.DurationMs < 0 {
		t.Errorf("expected valid execution duration, got %d", res.Execution.DurationMs)
	}
	if len(res.Execution.StageDurationsMs) == 0 {
		t.Errorf("expected recorded stage durations")
	}

	t.Logf("Zero-code model executed successfully: Value=%+v ExecutionDuration=%dms",
		valMap, res.Execution.DurationMs)
}

// TestUniversalEmbeddingContract tests that embedding models and embedding_extractor
// output dense vectors conforming to the universal output contract.
func TestUniversalEmbeddingContract(t *testing.T) {
	repoRoot := getRepoRoot(t)
	modelsDir := filepath.Join(repoRoot, "models")

	rtFactory := runtime.NewDefaultFactory()
	_ = rtFactory.RegisterProvider("mock", runtimeMock.NewProvider())

	stageReg := pipeline.NewDefaultStageRegistry()
	modelReg := registry.NewDefaultModelRegistry(stageReg, registry.AdaptRuntimeFactory(rtFactory))

	ctx := context.Background()
	if err := modelReg.Discover(ctx, modelsDir); err != nil {
		t.Fatalf("discovery failed: %v", err)
	}

	embedKey := domain.NewModelKey("embedding-test", "1.0.0")
	if err := modelReg.Load(ctx, embedKey); err != nil {
		t.Fatalf("failed to load embedding-test: %v", err)
	}
	defer func() { _ = modelReg.Unload(ctx, embedKey) }()

	handle, err := modelReg.Acquire(embedKey)
	if err != nil {
		t.Fatalf("failed to acquire embedding-test handle: %v", err)
	}
	defer handle.Release()

	// 1. Direct Execution
	res, err := handle.Execute(ctx, domain.RawInput{
		Text: "AI Model Runtime universal inference platform.",
	})
	if err != nil {
		t.Fatalf("embedding execution failed: %v", err)
	}

	vec, ok := res.Output.Value.([]float32)
	if !ok {
		t.Fatalf("expected []float32 in Output.Value, got %T (%+v)", res.Output.Value, res.Output.Value)
	}
	if len(vec) != 64 {
		t.Errorf("expected dimension 64, got %d", len(vec))
	}

	// 2. HTTP E2E Endpoint Test
	jobStore := storeMemory.NewMemoryJobStore()
	jobQueue := queueMemory.NewMemoryQueue()
	defer jobQueue.Close()

	inferSvc, err := service.NewInferenceService(service.ServiceConfig{
		Registry: modelReg,
		JobStore: jobStore,
		Queue:    jobQueue,
	})
	if err != nil {
		t.Fatalf("failed to create inference service: %v", err)
	}

	httpCfg := internalHTTP.DefaultServerConfig()
	server, err := internalHTTP.NewServer(httpCfg, inferSvc, internalHTTP.ServerDependencies{
		Queue: jobQueue,
	})
	if err != nil {
		t.Fatalf("failed to create HTTP server: %v", err)
	}

	ts := httptest.NewServer(server.Handler())
	defer ts.Close()

	payload := map[string]any{
		"input": map[string]any{
			"text": "Vector embeddings represent semantic meaning.",
		},
	}
	reqBytes, _ := json.Marshal(payload)

	resp, err := http.Post(
		ts.URL+"/v1/models/embedding-test/predict",
		"application/json",
		bytes.NewReader(reqBytes),
	)
	if err != nil {
		t.Fatalf("HTTP predict failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200 OK, got %d: %s", resp.StatusCode, string(b))
	}

	var inferResp internalHTTP.InferenceResponse
	if err := json.NewDecoder(resp.Body).Decode(&inferResp); err != nil {
		t.Fatalf("failed to decode HTTP response as InferenceResponse: %v", err)
	}

	valSlice, ok := inferResp.Data.Value.([]any)
	if !ok {
		t.Fatalf("expected []any in data.value, got %T (%+v)", inferResp.Data.Value, inferResp.Data.Value)
	}

	if len(valSlice) != 64 {
		t.Errorf("expected dimension 64, got %d", len(valSlice))
	}

	t.Logf("Universal embedding verified over HTTP: dimension=%d, model=%s", len(valSlice), inferResp.Meta.Model.ID)
}

func getRepoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("failed to get wd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("could not find repository root containing go.mod")
		}
		dir = parent
	}
}
