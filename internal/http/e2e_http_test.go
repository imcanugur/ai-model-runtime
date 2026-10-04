package http_test

import (
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
	"github.com/imcanugur/ai-model-runtime/internal/runtime/mock"
	"github.com/imcanugur/ai-model-runtime/internal/service"
	"github.com/imcanugur/ai-model-runtime/internal/worker"
)

func setupHTTPTestModelPackage(t *testing.T, rootDir string) (string, domain.ModelKey) {
	t.Helper()
	modelDir := filepath.Join(rootDir, "detector-v1")
	if err := os.MkdirAll(filepath.Join(modelDir, "weights"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(modelDir, "assets"), 0755); err != nil {
		t.Fatal(err)
	}

	manifestContent := `
schema_version: "v1"
id: "detector"
name: "HTTP E2E Detector"
version: "1.0.0"

metadata:
  task: "text-classification"
  languages: ["en"]

resources:
  max_concurrency: 4
  timeout_ms: 5000

runtime:
  provider: "mock"
  profile: "cpu"
  entrypoint: "weights/model.bin"

pipeline:
  stages:
    - name: "normalizer"
      type: "text_normalizer"
      config:
        strip_whitespace: true
        lowercase: true

    - name: "chunker"
      type: "sliding_window_chunker"
      config:
        max_tokens: 16
        stride: 0

    - name: "infer"
      type: "runtime_infer"

    - name: "reduction"
      type: "token_weighted_logits"
      config:
        human_class_index: 0
        ai_class_index: 1

    - name: "calibrator"
      type: "threshold_calibrator"
      config:
        threshold: 0.5
        positive_label: "AI"
        negative_label: "Human"
`
	if err := os.WriteFile(filepath.Join(modelDir, "model.yaml"), []byte(manifestContent), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "weights", "model.bin"), []byte("mock binary"), 0644); err != nil {
		t.Fatal(err)
	}

	return modelDir, domain.NewModelKey("detector", "1.0.0")
}

func TestE2E_HTTP_FullFlow(t *testing.T) {
	ctx := context.Background()

	// 1. Model Registry with mock runtime
	tmpDir := t.TempDir()
	_, modelKey := setupHTTPTestModelPackage(t, tmpDir)

	stageReg := pipeline.NewDefaultStageRegistry()
	mockRT := mock.NewMockRuntime()

	factory := func(manifest *registry.ModelManifest, modelDir string) (domain.Runtime, error) {
		return mockRT, nil
	}

	modelReg := registry.NewDefaultModelRegistry(stageReg, factory)
	if err := modelReg.Discover(ctx, tmpDir); err != nil {
		t.Fatalf("discovery failed: %v", err)
	}
	if err := modelReg.Load(ctx, modelKey); err != nil {
		t.Fatalf("load failed: %v", err)
	}

	// 2. Queue & JobStore
	q := queueMemory.NewMemoryQueue()
	defer q.Close()
	store := storeMemory.NewMemoryJobStore()

	// 3. Worker Pool
	pool, err := worker.NewWorkerPool(worker.WorkerPoolConfig{
		Count:          2,
		Queue:          q,
		Store:          store,
		Registry:       modelReg,
		DequeueTimeout: 50 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("failed to create worker pool: %v", err)
	}
	if err := pool.Start(ctx); err != nil {
		t.Fatalf("failed to start worker pool: %v", err)
	}

	// 4. Inference Service
	svc, err := service.NewInferenceService(service.ServiceConfig{
		Registry: modelReg,
		JobStore: store,
		Queue:    q,
	})
	if err != nil {
		t.Fatalf("failed to create service: %v", err)
	}

	// 5. HTTP Server
	cfg := internalHTTP.DefaultServerConfig()
	cfg.DefaultModelID = "detector"

	server, err := internalHTTP.NewServer(cfg, svc, internalHTTP.ServerDependencies{
		WorkerPool: pool,
		Queue:      q,
	})
	if err != nil {
		t.Fatalf("failed to create server: %v", err)
	}
	handler := server.Handler()

	// 6. Test Sync Prediction via HTTP
	syncBody := `{"input":{"text":"Synchronous academic detection sample"}}`
	reqSync := httptest.NewRequest(http.MethodPost, "/v1/models/detector/predict", strings.NewReader(syncBody))
	reqSync.Header.Set("Content-Type", "application/json")
	wSync := httptest.NewRecorder()
	handler.ServeHTTP(wSync, reqSync)

	if wSync.Code != http.StatusOK {
		t.Fatalf("sync predict expected 200, got %d: %s", wSync.Code, wSync.Body.String())
	}
	var syncResp internalHTTP.InferenceResponse
	_ = json.Unmarshal(wSync.Body.Bytes(), &syncResp)
	valMap, ok := syncResp.Data.Value.(map[string]any)
	if !ok || valMap["label"] != "AI" {
		t.Fatalf("expected AI label in data.value, got: %+v", syncResp.Data.Value)
	}

	// 7. Test Async Job Submission via HTTP
	asyncBody := `{"model":"detector","version":"1.0.0","input":{"text":"Asynchronous test text processed through HTTP -> Queue -> Worker"}}`
	reqAsync := httptest.NewRequest(http.MethodPost, "/v1/jobs", strings.NewReader(asyncBody))
	reqAsync.Header.Set("Content-Type", "application/json")
	wAsync := httptest.NewRecorder()
	handler.ServeHTTP(wAsync, reqAsync)

	if wAsync.Code != http.StatusCreated {
		t.Fatalf("async job create expected 201, got %d: %s", wAsync.Code, wAsync.Body.String())
	}
	var jobResp internalHTTP.CreateJobResponse
	_ = json.Unmarshal(wAsync.Body.Bytes(), &jobResp)
	jobID := jobResp.JobID

	// 8. Poll GET /v1/jobs/{jobID} until completed
	timeout := time.After(3 * time.Second)
	ticker := time.NewTicker(25 * time.Millisecond)
	defer ticker.Stop()

	var finalStatus internalHTTP.JobStatusResponse
	completed := false

	for !completed {
		select {
		case <-timeout:
			t.Fatal("timed out waiting for HTTP async job to complete")
		case <-ticker.C:
			reqStatus := httptest.NewRequest(http.MethodGet, "/v1/jobs/"+jobID, nil)
			wStatus := httptest.NewRecorder()
			handler.ServeHTTP(wStatus, reqStatus)

			if wStatus.Code == http.StatusOK {
				var status internalHTTP.JobStatusResponse
				_ = json.Unmarshal(wStatus.Body.Bytes(), &status)
				if status.State == "completed" {
					finalStatus = status
					completed = true
				}
			}
		}
	}

	jobVal, ok := finalStatus.Data.Value.(map[string]any)
	if !ok || jobVal["label"] != "AI" {
		t.Fatalf("unexpected completed job response: %+v", finalStatus)
	}

	// 9. Shutdown Server (drains HTTP, stops worker pool, closes queue)
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer shutdownCancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		t.Fatalf("server shutdown failed: %v", err)
	}
}
