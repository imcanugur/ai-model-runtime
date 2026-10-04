package safetensors_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	internalHTTP "github.com/imcanugur/ai-model-runtime/internal/http"
	storeMemory "github.com/imcanugur/ai-model-runtime/internal/jobstore/memory"
	"github.com/imcanugur/ai-model-runtime/internal/pipeline"
	queueMemory "github.com/imcanugur/ai-model-runtime/internal/queue/memory"
	"github.com/imcanugur/ai-model-runtime/internal/registry"
	"github.com/imcanugur/ai-model-runtime/internal/runtime"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/safetensors"
	"github.com/imcanugur/ai-model-runtime/internal/service"
)

func TestNativeTextGeneration_ZeroPython(t *testing.T) {
	root := findRepoRoot(t)
	modelArtifact := filepath.Join(root, "models", "text-generation-tr", "model.safetensors")
	if _, err := os.Stat(modelArtifact); err != nil {
		t.Skip("models/text-generation-tr/model.safetensors not present")
	}

	// 1. Initialize Runtime Factory & Pure-Go SafeTensors Provider (NO PYTHON)
	rtFactory := runtime.NewDefaultFactory()
	stProv := safetensors.NewProvider()
	if err := rtFactory.RegisterProvider("safetensors", stProv); err != nil {
		t.Fatalf("failed to register safetensors provider: %v", err)
	}

	// 2. Initialize Registry & Pipeline
	stageReg := pipeline.NewDefaultStageRegistry()
	modelReg := registry.NewDefaultModelRegistry(stageReg, registry.AdaptRuntimeFactory(rtFactory))

	ctx := context.Background()

	// 3. Discover models
	modelsDir := filepath.Join(root, "models")
	if err := modelReg.Discover(ctx, modelsDir); err != nil {
		t.Fatalf("model discovery failed: %v", err)
	}

	genKey := domain.NewModelKey("text-generation-tr", "1.0.0")
	if err := modelReg.Load(ctx, genKey); err != nil {
		t.Fatalf("failed to load text-generation-tr with native pure-Go runtime: %v", err)
	}
	defer func() { _ = modelReg.Unload(ctx, genKey) }()

	// 4. Acquire Handle & Test Direct Execution (Zero Python)
	handle, err := modelReg.Acquire(genKey)
	if err != nil {
		t.Fatalf("failed to acquire handle: %v", err)
	}
	defer handle.Release()

	prompt := "Türkiye'nin başkenti nedir?"
	start := time.Now()
	res, err := handle.Execute(ctx, domain.RawInput{
		Text: prompt,
		Parameters: map[string]string{
			"max_new_tokens": "32",
		},
	})
	dur := time.Since(start)
	if err != nil {
		t.Fatalf("native generation execution failed: %v", err)
	}

	genStr, ok := res.Output.Value.(string)
	if !ok {
		t.Fatalf("expected string output, got %T (%+v)", res.Output.Value, res.Output.Value)
	}
	if strings.TrimSpace(genStr) == "" {
		t.Fatalf("generated text is empty")
	}
	t.Logf("[NATIVE GO] Prompt: %q -> Generated: %q (Duration: %v)", prompt, genStr, dur)

	// 5. Test Full HTTP API (Raw Output Contract)
	jobStore := storeMemory.NewMemoryJobStore()
	jobQueue := queueMemory.NewMemoryQueue()
	defer jobQueue.Close()

	appSvc, err := service.NewInferenceService(service.ServiceConfig{
		Registry: modelReg,
		JobStore: jobStore,
		Queue:    jobQueue,
	})
	if err != nil {
		t.Fatalf("failed to create inference service: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/models/{modelID}/predict", internalHTTP.HandleSyncPredict(appSvc))

	httpReqBody := `{
		"input": {
			"text": "Atatürk hakkında kısa bir cümle yaz."
		},
		"options": {
			"max_new_tokens": 32
		}
	}`

	httpReq := httptest.NewRequest(http.MethodPost, "/v1/models/text-generation-tr/predict", bytes.NewBufferString(httpReqBody))
	httpReq.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.ServeHTTP(w, httpReq)

	if w.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200, got %d: %s", w.Code, w.Body.String())
	}

	rawBody := strings.TrimSpace(w.Body.String())
	var resp internalHTTP.InferenceResponse
	if err := json.Unmarshal([]byte(rawBody), &resp); err != nil {
		t.Fatalf("expected canonical InferenceResponse in response body: %v, raw: %s", err, rawBody)
	}
	httpGenStr, ok := resp.Data.Value.(string)
	if !ok || strings.TrimSpace(httpGenStr) == "" {
		t.Fatalf("expected non-empty string in data.value, got: %T (%+v)", resp.Data.Value, resp.Data.Value)
	}
	if resp.Meta.Model.ID != "text-generation-tr" {
		t.Fatalf("expected model.id text-generation-tr, got: %q", resp.Meta.Model.ID)
	}
	t.Logf("[NATIVE GO HTTP] Output: %q (Raw Body: %s)", httpGenStr, rawBody)
}

func TestNativeAllModels_ZeroPython(t *testing.T) {
	root := findRepoRoot(t)
	modelsDir := filepath.Join(root, "models")

	// Initialize Pure Go Runtime Factory
	rtFactory := runtime.NewDefaultFactory()
	stProv := safetensors.NewProvider()
	_ = rtFactory.RegisterProvider("safetensors", stProv)
	_ = rtFactory.RegisterProvider("native", stProv)

	stageReg := pipeline.NewDefaultStageRegistry()
	modelReg := registry.NewDefaultModelRegistry(stageReg, registry.AdaptRuntimeFactory(rtFactory))

	ctx := context.Background()
	if err := modelReg.Discover(ctx, modelsDir); err != nil {
		t.Fatalf("discovery failed: %v", err)
	}

	jobStore := storeMemory.NewMemoryJobStore()
	jobQueue := queueMemory.NewMemoryQueue()
	defer jobQueue.Close()

	appSvc, err := service.NewInferenceService(service.ServiceConfig{
		Registry: modelReg,
		JobStore: jobStore,
		Queue:    jobQueue,
	})
	if err != nil {
		t.Fatalf("failed to create inference service: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/models/{modelID}/predict", internalHTTP.HandleSyncPredict(appSvc))

	tests := []struct {
		name       string
		modelID    string
		reqBody    string
		validateFn func(t *testing.T, resp internalHTTP.InferenceResponse)
	}{
		{
			name:    "Text Generation Pure Go",
			modelID: "text-generation-tr",
			reqBody: `{"input":{"text":"Atatürk hakkında kısa bir cümle yaz."},"options":{"max_new_tokens":64}}`,
			validateFn: func(t *testing.T, resp internalHTTP.InferenceResponse) {
				str, ok := resp.Data.Value.(string)
				if !ok || strings.TrimSpace(str) == "" {
					t.Fatalf("expected non-empty string in text-generation-tr, got: %T (%+v)", resp.Data.Value, resp.Data.Value)
				}
				if resp.Meta.Model.ID != "text-generation-tr" {
					t.Fatalf("expected model.id text-generation-tr, got: %s", resp.Meta.Model.ID)
				}
				t.Logf("Generation result: %q", str)
			},
		},
		{
			name:    "Embedding Pure Go",
			modelID: "embedding-tr-real",
			reqBody: `{"input":{"text":"Merhaba dünya"},"options":{}}`,
			validateFn: func(t *testing.T, resp internalHTTP.InferenceResponse) {
				vec, ok := resp.Data.Value.([]any)
				if !ok {
					t.Fatalf("expected []any in embedding-tr-real, got: %T (%+v)", resp.Data.Value, resp.Data.Value)
				}
				if len(vec) != 384 {
					t.Fatalf("expected 384 dimensions, got: %d", len(vec))
				}
				if resp.Meta.Model.ID != "embedding-tr-real" {
					t.Fatalf("expected model.id embedding-tr-real, got: %s", resp.Meta.Model.ID)
				}
				t.Logf("Embedding result: dimension=%d", len(vec))
			},
		},
		{
			name:    "Classification Pure Go",
			modelID: "safetensors-real",
			reqBody: `{"input":{"text":"Bu ürün gerçekten harika ve kaliteli."},"options":{}}`,
			validateFn: func(t *testing.T, resp internalHTTP.InferenceResponse) {
				resMap, ok := resp.Data.Value.(map[string]any)
				if !ok {
					t.Fatalf("expected map in safetensors-real, got: %T (%+v)", resp.Data.Value, resp.Data.Value)
				}
				if resMap["label"] == "" {
					t.Fatalf("expected label in classification result: %+v", resMap)
				}
				if resp.Meta.Model.ID != "safetensors-real" {
					t.Fatalf("expected model.id safetensors-real, got: %s", resp.Meta.Model.ID)
				}
				t.Logf("Classification result: %+v", resMap)
			},
		},
		{
			name:    "QA Pure Go",
			modelID: "qa-test",
			reqBody: `{"input":{"text":"AI Model Runtime Go ile yazılmıştır.","question":"AI Model Runtime hangi dil ile yazılmıştır?"},"options":{}}`,
			validateFn: func(t *testing.T, resp internalHTTP.InferenceResponse) {
				resMap, ok := resp.Data.Value.(map[string]any)
				if !ok {
					t.Fatalf("expected map in qa-test, got: %T (%+v)", resp.Data.Value, resp.Data.Value)
				}
				if resMap["answer"] == "" {
					t.Fatalf("expected non-empty answer in qa-test: %+v", resMap)
				}
				if resp.Meta.Model.ID != "qa-test" {
					t.Fatalf("expected model.id qa-test, got: %s", resp.Meta.Model.ID)
				}
				t.Logf("QA result: %+v", resMap)
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			key := domain.NewModelKey(domain.ModelID(tc.modelID), "1.0.0")
			if err := modelReg.Load(ctx, key); err != nil {
				t.Fatalf("failed to load model %s: %v", tc.modelID, err)
			}
			defer func() { _ = modelReg.Unload(ctx, key) }()

			httpReq := httptest.NewRequest(http.MethodPost, "/v1/models/"+tc.modelID+"/predict", bytes.NewBufferString(tc.reqBody))
			httpReq.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			mux.ServeHTTP(w, httpReq)
			if w.Code != http.StatusOK {
				t.Fatalf("expected HTTP 200, got %d: %s", w.Code, w.Body.String())
			}

			var resp internalHTTP.InferenceResponse
			if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
				t.Fatalf("failed to unmarshal InferenceResponse: %v, raw: %s", err, w.Body.String())
			}

			tc.validateFn(t, resp)
		})
	}
}
