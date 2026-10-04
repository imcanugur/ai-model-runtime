package sidecar_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

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

func TestQAModel_FullRealInference(t *testing.T) {
	pythonPath := findPythonBinary(t)
	if pythonPath == "" {
		t.Skip("Python interpreter not found; skipping QA real model tests")
	}

	endpoint, cancelDaemon, killDaemon := startRealSidecarDaemon(t, pythonPath)
	defer killDaemon()
	defer cancelDaemon()

	repoRoot := getRepoRoot(t)

	// 1. Initialize Runtime Factory & Provider
	rtFactory := runtime.NewDefaultFactory()
	sidecarProv := sidecar.NewProvider(endpoint, nil)
	if err := rtFactory.RegisterProvider("sidecar", sidecarProv); err != nil {
		t.Fatalf("failed to register sidecar provider: %v", err)
	}

	// 2. Initialize Registry
	stageReg := pipeline.NewDefaultStageRegistry()
	modelReg := registry.NewDefaultModelRegistry(stageReg, registry.AdaptRuntimeFactory(rtFactory))

	ctx := context.Background()

	// 3. Discover models
	modelsDir := filepath.Join(repoRoot, "models")
	if err := modelReg.Discover(ctx, modelsDir); err != nil {
		t.Fatalf("model discovery failed: %v", err)
	}

	// Verify model.yaml architecture matches config.json
	qaKey := domain.NewModelKey("qa-test", "1.0.0")
	if err := modelReg.Load(ctx, qaKey); err != nil {
		t.Fatalf("failed to load qa-test model: %v", err)
	}
	defer func() { _ = modelReg.Unload(ctx, qaKey) }()

	// Verify safetensors-test classification model still loads simultaneously
	clfKey := domain.NewModelKey("safetensors-test", "1.0.0")
	if err := modelReg.Load(ctx, clfKey); err != nil {
		t.Fatalf("failed to load classification model: %v", err)
	}
	defer func() { _ = modelReg.Unload(ctx, clfKey) }()

	// 4. Test QA Model Handle Execution
	handle, err := modelReg.Acquire(qaKey)
	if err != nil {
		t.Fatalf("failed to acquire qa-test handle: %v", err)
	}
	defer handle.Release()

	// Test Case 1: "What programming language is the runtime written in?" -> "Go"
	context1 := "The runtime is an AI inference platform written in Go. It uses Redis for asynchronous job processing."
	question1 := "What programming language is the runtime written in?"

	result1, err := handle.Execute(ctx, domain.RawInput{
		Context:  context1,
		Question: question1,
	})
	if err != nil {
		t.Fatalf("QA execution 1 failed: %v", err)
	}

	t.Logf("QA Result 1: Answer=%q Confidence=%f Score=%f", result1.Answer, result1.Confidence, result1.PrimaryScore)
	if !strings.EqualFold(strings.TrimSpace(result1.Answer), "Go") {
		t.Errorf("expected answer 'Go', got %q", result1.Answer)
	}
	if result1.Confidence <= 0 || result1.Confidence > 1.0 {
		t.Errorf("expected confidence between 0.0 and 1.0, got %f", result1.Confidence)
	}

	// Test Case 2: "What does the runtime use for asynchronous job processing?" -> "Redis"
	context2 := "The runtime platform uses Redis for asynchronous job processing."
	question2 := "What does the runtime use for asynchronous job processing?"

	result2, err := handle.Execute(ctx, domain.RawInput{
		Context:  context2,
		Question: question2,
	})
	if err != nil {
		t.Fatalf("QA execution 2 failed: %v", err)
	}

	t.Logf("QA Result 2: Answer=%q Confidence=%f Score=%f", result2.Answer, result2.Confidence, result2.PrimaryScore)
	if !strings.EqualFold(strings.TrimSpace(result2.Answer), "Redis") {
		t.Errorf("expected answer 'Redis', got %q", result2.Answer)
	}

	// 5. Test Classification Model Regression (Verify it still predicts properly)
	clfHandle, err := modelReg.Acquire(clfKey)
	if err != nil {
		t.Fatalf("failed to acquire safetensors-test handle: %v", err)
	}
	defer clfHandle.Release()

	clfRes, err := clfHandle.Execute(ctx, domain.RawInput{Text: "Sample test text."})
	if err != nil {
		t.Fatalf("classification execution failed: %v", err)
	}
	if clfRes.Label != "LABEL_0" && clfRes.Label != "LABEL_1" {
		t.Errorf("expected classification label LABEL_0/1, got %q", clfRes.Label)
	}

	// 6. Test HTTP E2E on POST /v1/models/qa-test/predict
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

	httpPayload := map[string]any{
		"input": map[string]any{
			"context":  context1,
			"question": question1,
			"language": "en",
		},
	}
	reqBytes, _ := json.Marshal(httpPayload)

	httpResp, err := http.Post(
		ts.URL+"/v1/models/qa-test/predict",
		"application/json",
		bytes.NewReader(reqBytes),
	)
	if err != nil {
		t.Fatalf("HTTP POST to qa-test failed: %v", err)
	}
	defer func() { _ = httpResp.Body.Close() }()

	bodyBytes, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 OK from qa-test predict, got %d: %s", httpResp.StatusCode, string(bodyBytes))
	}

	var resp internalHTTP.InferenceResponse
	if err := json.Unmarshal(bodyBytes, &resp); err != nil {
		t.Fatalf("failed to parse HTTP response: %v, body: %s", err, string(bodyBytes))
	}

	valMap, ok := resp.Data.Value.(map[string]any)
	if !ok {
		t.Fatalf("expected map in data.value, got %T (%+v)", resp.Data.Value, resp.Data.Value)
	}

	ans, _ := valMap["answer"].(string)
	if !strings.EqualFold(strings.TrimSpace(ans), "Go") {
		t.Errorf("expected HTTP response answer 'Go', got %q", ans)
	}
	t.Logf("QA HTTP E2E Successful: Model=%s Output=%+v", resp.Meta.Model.ID, valMap)
}
