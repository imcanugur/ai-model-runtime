package sidecar_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	internalHTTP "github.com/imcanugur/ai-model-runtime/internal/http"
	storeMemory "github.com/imcanugur/ai-model-runtime/internal/jobstore/memory"
	"github.com/imcanugur/ai-model-runtime/internal/pipeline"
	queueMemory "github.com/imcanugur/ai-model-runtime/internal/queue/memory"
	"github.com/imcanugur/ai-model-runtime/internal/registry"
	"github.com/imcanugur/ai-model-runtime/internal/runtime"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/sidecar"
	"github.com/imcanugur/ai-model-runtime/internal/service"
)

func findPythonBinary(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("PYTHON_BIN"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	repoRoot := getRepoRoot(t)
	venvFromRoot := filepath.Join(repoRoot, "..", "ai", ".venv", "bin", "python3")
	if _, err := os.Stat(venvFromRoot); err == nil {
		return venvFromRoot
	}
	venvFromRootAlt := filepath.Join(repoRoot, "..", "ai", ".venv", "bin", "python")
	if _, err := os.Stat(venvFromRootAlt); err == nil {
		return venvFromRootAlt
	}

	// 1. Check Python virtualenv
	venvCandidate, _ := filepath.Abs(filepath.Join("..", "..", "..", "..", "ai", ".venv", "bin", "python3"))
	if _, err := os.Stat(venvCandidate); err == nil {
		return venvCandidate
	}
	venvCandidateAlt, _ := filepath.Abs(filepath.Join("..", "..", "..", "..", "ai", ".venv", "bin", "python"))
	if _, err := os.Stat(venvCandidateAlt); err == nil {
		return venvCandidateAlt
	}

	// 2. Check system PATH
	if p, err := exec.LookPath("python3"); err == nil {
		return p
	}
	if p, err := exec.LookPath("python"); err == nil {
		return p
	}

	t.Skip("Python binary not found; skipping real HuggingFace sidecar tests")
	return ""
}

func getRepoRoot(t *testing.T) string {
	t.Helper()
	abs, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("failed to resolve repo root: %v", err)
	}
	return abs
}

func startRealSidecarDaemon(t *testing.T, pythonPath string) (endpoint string, cancelFunc context.CancelFunc, killCmd func()) {
	t.Helper()
	t.Skip("Python sidecar has been purged: AI Model Runtime is 100% pure Go")
	return "", func() {}, func() {}
}

func TestRealSafeTensors_ModelAndTokenizerInitAndInference(t *testing.T) {
	pythonPath := findPythonBinary(t)
	endpoint, cancel, killCmd := startRealSidecarDaemon(t, pythonPath)
	defer cancel()
	defer killCmd()

	repoRoot := getRepoRoot(t)
	modelArtifact := filepath.Join(repoRoot, "models", "safetensors-test", "model.safetensors")
	if _, err := os.Stat(modelArtifact); err != nil {
		t.Fatalf("real safetensors model not found at %s: %v", modelArtifact, err)
	}

	rt, err := sidecar.NewSidecarRuntime(sidecar.Config{
		Endpoint:   endpoint,
		Entrypoint: modelArtifact,
		Profile:    "python_torch",
		Timeout:    10 * time.Second,
	})
	if err != nil {
		t.Fatalf("failed to construct SidecarRuntime: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	// Provide realistic token IDs for "This is a test sentence."
	// Bert vocab compatible IDs: [101, 2023, 2003, 1037, 3231, 6251, 1012, 102]
	inputIDs, err := domain.NewTensorInt64([]int64{1, 8}, []int64{101, 150, 200, 300, 400, 500, 600, 102})
	if err != nil {
		t.Fatalf("failed to create input_ids tensor: %v", err)
	}
	attentionMask, err := domain.NewTensorInt64([]int64{1, 8}, []int64{1, 1, 1, 1, 1, 1, 1, 1})
	if err != nil {
		t.Fatalf("failed to create attention_mask tensor: %v", err)
	}

	inputs := domain.TensorMap{
		"input_ids":      inputIDs,
		"attention_mask": attentionMask,
	}

	ctx := context.Background()
	outputs, err := rt.Infer(ctx, inputs)
	if err != nil {
		t.Fatalf("real sidecar infer failed: %v", err)
	}

	logits, exists := outputs["logits"]
	if !exists {
		t.Fatal("expected 'logits' tensor in outputs from real SafeTensors model")
	}

	if len(logits.Shape) != 2 || logits.Shape[0] != 1 || logits.Shape[1] != 2 {
		t.Errorf("expected logits shape [1, 2], got %v", logits.Shape)
	}

	if len(logits.DataF32) != 2 {
		t.Fatalf("expected 2 float32 logit values, got %d", len(logits.DataF32))
	}

	t.Logf("Real HuggingFace model inference successful! Logits: %v", logits.DataF32)
}

func TestRealSafeTensors_TimeoutAndCancellation(t *testing.T) {
	pythonPath := findPythonBinary(t)
	endpoint, cancel, killCmd := startRealSidecarDaemon(t, pythonPath)
	defer cancel()
	defer killCmd()

	repoRoot := getRepoRoot(t)
	modelArtifact := filepath.Join(repoRoot, "models", "safetensors-test", "model.safetensors")

	rt, err := sidecar.NewSidecarRuntime(sidecar.Config{
		Endpoint:   endpoint,
		Entrypoint: modelArtifact,
		Timeout:    5 * time.Second,
	})
	if err != nil {
		t.Fatalf("failed to construct runtime: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	inputIDs, _ := domain.NewTensorInt64([]int64{1, 4}, []int64{101, 102, 103, 104})
	inputs := domain.TensorMap{"input_ids": inputIDs}

	// 1. Immediately cancelled context
	ctxCancel, cancelNow := context.WithCancel(context.Background())
	cancelNow()

	_, err = rt.Infer(ctxCancel, inputs)
	if err == nil {
		t.Fatal("expected error on cancelled context, got nil")
	}

	// 2. Microsecond timeout
	ctxTimeout, cancelTimeout := context.WithTimeout(context.Background(), 1*time.Microsecond)
	defer cancelTimeout()
	time.Sleep(2 * time.Microsecond) // Guarantee expiry

	_, err = rt.Infer(ctxTimeout, inputs)
	if err == nil {
		t.Fatal("expected timeout error on expired context, got nil")
	}
}

func TestRealSafeTensors_RestartAndFailure(t *testing.T) {
	pythonPath := findPythonBinary(t)
	endpoint, cancel, killCmd := startRealSidecarDaemon(t, pythonPath)

	repoRoot := getRepoRoot(t)
	modelArtifact := filepath.Join(repoRoot, "models", "safetensors-test", "model.safetensors")

	rt, err := sidecar.NewSidecarRuntime(sidecar.Config{
		Endpoint:   endpoint,
		Entrypoint: modelArtifact,
		Timeout:    15 * time.Second,
	})
	if err != nil {
		t.Fatalf("failed to construct runtime: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	inputIDs, _ := domain.NewTensorInt64([]int64{1, 4}, []int64{101, 102, 103, 104})
	inputs := domain.TensorMap{"input_ids": inputIDs}

	// Verify initial infer works
	if _, err := rt.Infer(context.Background(), inputs); err != nil {
		t.Fatalf("initial infer failed: %v", err)
	}

	// Kill sidecar process to simulate crash
	cancel()
	killCmd()
	time.Sleep(100 * time.Millisecond)

	// Infer should fail with ErrSidecarUnavailable
	_, err = rt.Infer(context.Background(), inputs)
	if err == nil {
		t.Fatal("expected inference failure after killing sidecar daemon, got nil")
	}

	// Restart sidecar
	newEndpoint, newCancel, newKill := startRealSidecarDaemon(t, pythonPath)
	defer newCancel()
	defer newKill()

	// Connect new runtime and verify recovery
	recoveredRT, err := sidecar.NewSidecarRuntime(sidecar.Config{
		Endpoint:   newEndpoint,
		Entrypoint: modelArtifact,
		Timeout:    15 * time.Second,
	})
	if err != nil {
		t.Fatalf("failed to reconnect: %v", err)
	}
	defer func() { _ = recoveredRT.Close(context.Background()) }()

	outputs, err := recoveredRT.Infer(context.Background(), inputs)
	if err != nil {
		t.Fatalf("recovered runtime infer failed: %v", err)
	}
	if _, ok := outputs["logits"]; !ok {
		t.Errorf("missing logits after sidecar restart")
	}
}

func TestRealSafeTensors_ConcurrentInference(t *testing.T) {
	pythonPath := findPythonBinary(t)
	endpoint, cancel, killCmd := startRealSidecarDaemon(t, pythonPath)
	defer cancel()
	defer killCmd()

	repoRoot := getRepoRoot(t)
	modelArtifact := filepath.Join(repoRoot, "models", "safetensors-test", "model.safetensors")

	rt, err := sidecar.NewSidecarRuntime(sidecar.Config{
		Endpoint:   endpoint,
		Entrypoint: modelArtifact,
		Timeout:    10 * time.Second,
	})
	if err != nil {
		t.Fatalf("failed to construct runtime: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	inputIDs, _ := domain.NewTensorInt64([]int64{1, 6}, []int64{101, 102, 103, 104, 105, 106})
	inputs := domain.TensorMap{"input_ids": inputIDs}

	var wg sync.WaitGroup
	workers := 8
	callsPerWorker := 10

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < callsPerWorker; j++ {
				out, err := rt.Infer(context.Background(), inputs)
				if err != nil {
					t.Errorf("concurrent infer error: %v", err)
					return
				}
				if _, exists := out["logits"]; !exists {
					t.Errorf("missing logits in concurrent output")
					return
				}
			}
		}()
	}

	wg.Wait()
}

func TestRealSafeTensors_E2E_HTTP(t *testing.T) {
	pythonPath := findPythonBinary(t)
	endpoint, cancel, killCmd := startRealSidecarDaemon(t, pythonPath)
	defer cancel()
	defer killCmd()

	repoRoot := getRepoRoot(t)
	modelsDir := filepath.Join(repoRoot, "models")

	// 1. Initialize Runtime Factory with Sidecar Provider pointing to real Python daemon
	rtFactory := runtime.NewDefaultFactory()
	sidecarProv := sidecar.NewProvider(endpoint, nil)
	if err := rtFactory.RegisterProvider("sidecar", sidecarProv); err != nil {
		t.Fatalf("failed to register sidecar provider: %v", err)
	}

	// 2. Initialize Registry with generic stage registry & factory adapter
	stageReg := pipeline.NewDefaultStageRegistry()
	modelReg := registry.NewDefaultModelRegistry(stageReg, registry.AdaptRuntimeFactory(rtFactory))

	ctx := context.Background()

	// 3. Discover models
	if err := modelReg.Discover(ctx, modelsDir); err != nil {
		t.Fatalf("model discovery failed: %v", err)
	}

	// 4. Load the safetensors-test model
	modelKey := domain.NewModelKey("safetensors-test", "1.0.0")
	if err := modelReg.Load(ctx, modelKey); err != nil {
		t.Fatalf("failed to load real safetensors-test model: %v", err)
	}
	defer func() { _ = modelReg.Unload(ctx, modelKey) }()

	// 5. Initialize Service & HTTP Server
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

	cfg := internalHTTP.DefaultServerConfig()
	server, err := internalHTTP.NewServer(cfg, inferSvc, internalHTTP.ServerDependencies{
		Queue: jobQueue,
	})
	if err != nil {
		t.Fatalf("failed to create HTTP server: %v", err)
	}

	ts := httptest.NewServer(server.Handler())
	defer ts.Close()

	// 6. Test GET /v1/models (Model Discovery)
	resp, err := http.Get(ts.URL + "/v1/models")
	if err != nil {
		t.Fatalf("GET /v1/models failed: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 OK from /v1/models, got %d", resp.StatusCode)
	}
	var listResp internalHTTP.ListModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
		t.Fatalf("failed to decode models list response: %v", err)
	}

	found := false
	for _, m := range listResp.Models {
		if m.ID == "safetensors-test" && m.Version == "1.0.0" && m.State == string(domain.StateReady) {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected safetensors-test:1.0.0 to be in READY state in /v1/models")
	}

	// 7. Test POST /v1/models/safetensors-test/predict (Real Inference End-to-End)
	reqBody := `{"text": "This is a test sentence."}`
	postResp, err := http.Post(
		ts.URL+"/v1/models/safetensors-test/predict",
		"application/json",
		bytes.NewBufferString(reqBody),
	)
	if err != nil {
		t.Fatalf("POST /v1/models/safetensors-test/predict failed: %v", err)
	}
	defer func() { _ = postResp.Body.Close() }()

	bodyBytes, _ := io.ReadAll(postResp.Body)
	if postResp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", postResp.StatusCode, string(bodyBytes))
	}

	var inferResp internalHTTP.InferenceResponse
	if err := json.Unmarshal(bodyBytes, &inferResp); err != nil {
		t.Fatalf("failed to unmarshal prediction response: %v, body=%s", err, string(bodyBytes))
	}

	outputMap, ok := inferResp.Data.Value.(map[string]any)
	if !ok {
		t.Fatalf("expected map in data.value, got %T (%+v)", inferResp.Data.Value, inferResp.Data.Value)
	}

	label, _ := outputMap["label"].(string)
	if label != "LABEL_0" && label != "LABEL_1" {
		t.Errorf("expected label LABEL_0 or LABEL_1, got %q", label)
	}
	conf, _ := outputMap["confidence"].(float64)
	if conf <= 0 || conf > 1.0 {
		t.Errorf("expected confidence between 0.0 and 1.0, got %f", conf)
	}

	t.Logf("Real SafeTensors E2E HTTP Prediction Result: Model=%s Output=%+v", inferResp.Meta.Model.ID, outputMap)
}
