package http_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	internalHTTP "github.com/imcanugur/ai-model-runtime/internal/http"
	storeMemory "github.com/imcanugur/ai-model-runtime/internal/jobstore/memory"
	queueMemory "github.com/imcanugur/ai-model-runtime/internal/queue/memory"
	"github.com/imcanugur/ai-model-runtime/internal/service"
	"github.com/imcanugur/ai-model-runtime/internal/worker"
)

type mockHandle struct {
	id      domain.ModelID
	version domain.ModelVersion
}

func (m *mockHandle) ID() domain.ModelID           { return m.id }
func (m *mockHandle) Version() domain.ModelVersion { return m.version }
func (m *mockHandle) Key() domain.ModelKey         { return domain.NewModelKey(m.id, m.version) }
func (m *mockHandle) Execute(ctx context.Context, input domain.RawInput) (*domain.PredictionResult, error) {
	if input.Text == "panic" {
		panic("unexpected execution panic")
	}
	if input.Text == "pipeline-error" {
		return nil, errors.New("underlying model error")
	}
	res := &domain.PredictionResult{
		ModelID:      m.id,
		ModelVersion: m.version,
		PrimaryScore: 0.92,
		Label:        "AI",
		Confidence:   0.92,
		Classifications: []domain.Classification{
			{Label: "AI", Probability: 0.92},
			{Label: "Human", Probability: 0.08},
		},
		ExecutionTimeMs: 5,
	}
	res.Sync()
	return res, nil
}
func (m *mockHandle) Release() {}

type mockRegistry struct{}

func (r *mockRegistry) Discover(ctx context.Context, rootDir string) error    { return nil }
func (r *mockRegistry) Load(ctx context.Context, key domain.ModelKey) error   { return nil }
func (r *mockRegistry) Unload(ctx context.Context, key domain.ModelKey) error { return nil }
func (r *mockRegistry) List() []domain.ModelDescriptor {
	return []domain.ModelDescriptor{
		{
			Key: "text-detector-tr:2.1.0",
			Metadata: domain.ModelMetadata{
				ID:        "text-detector-tr",
				Version:   "2.1.0",
				Name:      "Turkish Detector",
				Task:      "text-classification",
				Languages: []string{"tr"},
			},
			State:        domain.StateReady,
			ActiveLeases: 0,
		},
	}
}

func (r *mockRegistry) Acquire(key domain.ModelKey) (domain.ModelHandle, error) {
	id, ver, err := key.Split()
	if err != nil {
		return nil, err
	}
	if id != "text-detector-tr" {
		return nil, domain.ErrModelNotFound
	}
	return &mockHandle{id: id, version: ver}, nil
}

func (r *mockRegistry) AcquireDefault(id domain.ModelID) (domain.ModelHandle, error) {
	if id != "text-detector-tr" {
		return nil, domain.ErrModelNotFound
	}
	return r.Acquire(domain.NewModelKey(id, "2.1.0"))
}

type readyCheck struct {
	err error
}

func (c *readyCheck) IsReady(ctx context.Context) error {
	return c.err
}

func setupTestServer(t *testing.T, maxBodyBytes int64, checker *readyCheck) (*internalHTTP.Server, *storeMemory.MemoryJobStore, *queueMemory.MemoryQueue) {
	t.Helper()
	reg := &mockRegistry{}
	store := storeMemory.NewMemoryJobStore()
	q := queueMemory.NewMemoryQueue()

	checkers := make(map[string]service.ReadyChecker)
	if checker != nil {
		checkers["database"] = checker
	}

	svc, err := service.NewInferenceService(service.ServiceConfig{
		Registry: reg,
		JobStore: store,
		Queue:    q,
		Checkers: checkers,
	})
	if err != nil {
		t.Fatalf("failed to create service: %v", err)
	}

	pool, err := worker.NewWorkerPool(worker.WorkerPoolConfig{
		Count:          2,
		Queue:          q,
		Store:          store,
		Registry:       reg,
		DequeueTimeout: 20 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("failed to create worker pool: %v", err)
	}
	if err := pool.Start(context.Background()); err != nil {
		t.Fatalf("failed to start worker pool: %v", err)
	}

	cfg := internalHTTP.DefaultServerConfig()
	if maxBodyBytes > 0 {
		cfg.MaxBodyBytes = maxBodyBytes
	}
	cfg.DefaultModelID = "text-detector-tr"

	srv, err := internalHTTP.NewServer(cfg, svc, internalHTTP.ServerDependencies{
		WorkerPool: pool,
		Queue:      q,
	})
	if err != nil {
		t.Fatalf("failed to create server: %v", err)
	}

	return srv, store, q
}

func TestHTTP_SyncPredict_Success(t *testing.T) {
	srv, _, _ := setupTestServer(t, 0, nil)
	handler := srv.Handler()

	// 1. POST /v1/models/text-detector-tr/predict
	payload := `{"input":{"text":"Bu bir yapay zeka metnidir."}}`
	req := httptest.NewRequest(http.MethodPost, "/v1/models/text-detector-tr/predict", strings.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp internalHTTP.InferenceResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	valMap, ok := resp.Data.Value.(map[string]any)
	if !ok || valMap["label"] != "AI" || valMap["confidence"] != 0.92 {
		t.Fatalf("unexpected result in data.value: %+v", resp.Data.Value)
	}
	if resp.Meta.Model.ID != "text-detector-tr" {
		t.Fatalf("unexpected model ID in meta: %s", resp.Meta.Model.ID)
	}

	// Verify X-Request-ID header generated
	if w.Header().Get("X-Request-ID") == "" {
		t.Fatal("expected X-Request-ID response header")
	}

	// 2. Specific version: POST /v1/models/text-detector-tr/versions/1.0.0/predict
	reqVer := httptest.NewRequest(http.MethodPost, "/v1/models/text-detector-tr/versions/1.0.0/predict", strings.NewReader(payload))
	reqVer.Header.Set("Content-Type", "application/json")
	wVer := httptest.NewRecorder()

	handler.ServeHTTP(wVer, reqVer)
	if wVer.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", wVer.Code, wVer.Body.String())
	}
}

func TestHTTP_SyncPredict_Errors(t *testing.T) {
	srv, _, _ := setupTestServer(t, 0, nil)
	handler := srv.Handler()

	// 1. Model Not Found
	payload := `{"input":{"text":"hello"}}`
	req := httptest.NewRequest(http.MethodPost, "/v1/models/non-existent/predict", strings.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d: %s", w.Code, w.Body.String())
	}
	var errResp internalHTTP.ErrorResponse
	_ = json.Unmarshal(w.Body.Bytes(), &errResp)
	if errResp.Error.Code != "MODEL_NOT_FOUND" {
		t.Fatalf("expected MODEL_NOT_FOUND, got: %s", errResp.Error.Code)
	}

	// 2. Empty Text -> 422
	emptyPayload := `{"input":{"text":"   "}}`
	req2 := httptest.NewRequest(http.MethodPost, "/v1/models/text-detector-tr/predict", strings.NewReader(emptyPayload))
	req2.Header.Set("Content-Type", "application/json")
	w2 := httptest.NewRecorder()
	handler.ServeHTTP(w2, req2)
	if w2.Code != http.StatusUnprocessableEntity {
		t.Fatalf("expected 422, got %d: %s", w2.Code, w2.Body.String())
	}

	// 3. Malformed JSON -> 400
	malformed := `{"input": {broken json`
	req3 := httptest.NewRequest(http.MethodPost, "/v1/models/text-detector-tr/predict", strings.NewReader(malformed))
	req3.Header.Set("Content-Type", "application/json")
	w3 := httptest.NewRecorder()
	handler.ServeHTTP(w3, req3)
	if w3.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", w3.Code, w3.Body.String())
	}
}

func TestHTTP_AsyncJobs_LifecycleAndIdempotency(t *testing.T) {
	srv, _, _ := setupTestServer(t, 0, nil)
	handler := srv.Handler()

	// 1. Create Job: POST /v1/jobs
	body := `{"model":"text-detector-tr","version":"2.1.0","input":{"text":"Asenkron is kuyrugu metni"}}`
	req := httptest.NewRequest(http.MethodPost, "/v1/jobs", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Idempotency-Key", "idemp-key-abc")
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201 Created, got %d: %s", w.Code, w.Body.String())
	}

	var createResp internalHTTP.CreateJobResponse
	_ = json.Unmarshal(w.Body.Bytes(), &createResp)
	if createResp.JobID != "idemp-key-abc" || createResp.State != "queued" {
		t.Fatalf("unexpected create response: %+v", createResp)
	}

	// 2. Resubmit with same Idempotency-Key -> 200 OK (no duplicate job created)
	reqDup := httptest.NewRequest(http.MethodPost, "/v1/jobs", strings.NewReader(body))
	reqDup.Header.Set("Content-Type", "application/json")
	reqDup.Header.Set("Idempotency-Key", "idemp-key-abc")
	wDup := httptest.NewRecorder()

	handler.ServeHTTP(wDup, reqDup)
	if wDup.Code != http.StatusOK {
		t.Fatalf("expected 200 OK for duplicate idempotency key, got %d: %s", wDup.Code, wDup.Body.String())
	}

	var dupResp internalHTTP.CreateJobResponse
	_ = json.Unmarshal(wDup.Body.Bytes(), &dupResp)
	if dupResp.JobID != createResp.JobID {
		t.Fatalf("expected same job ID for duplicate idempotency, got %s vs %s", dupResp.JobID, createResp.JobID)
	}

	// 3. Worker dequeues from queue, claims, executes, and completes the job in store.
	// Poll GET /v1/jobs/{jobID} until completed.
	timeout := time.After(3 * time.Second)
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()

	var statusResp internalHTTP.JobStatusResponse
	completed := false

	for !completed {
		select {
		case <-timeout:
			t.Fatalf("timed out waiting for job to complete, last state: %s", statusResp.State)
		case <-ticker.C:
			reqGetDone := httptest.NewRequest(http.MethodGet, "/v1/jobs/"+createResp.JobID, nil)
			wGetDone := httptest.NewRecorder()
			handler.ServeHTTP(wGetDone, reqGetDone)

			if wGetDone.Code == http.StatusOK {
				_ = json.Unmarshal(wGetDone.Body.Bytes(), &statusResp)
				if statusResp.State == "completed" {
					completed = true
				}
			}
		}
	}

	valMap, ok := statusResp.Data.Value.(map[string]any)
	if !ok || valMap["label"] != "AI" {
		t.Fatalf("expected completed with AI result, got: %+v", statusResp.Data)
	}

	// 4. Unknown job -> 404
	reqGetUnknown := httptest.NewRequest(http.MethodGet, "/v1/jobs/unknown-job-id", nil)
	wUnknown := httptest.NewRecorder()
	handler.ServeHTTP(wUnknown, reqGetUnknown)
	if wUnknown.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", wUnknown.Code)
	}
}

func TestHTTP_ModelDiscovery(t *testing.T) {
	srv, _, _ := setupTestServer(t, 0, nil)
	handler := srv.Handler()

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var modelsResp internalHTTP.ListModelsResponse
	if err := json.Unmarshal(w.Body.Bytes(), &modelsResp); err != nil {
		t.Fatalf("failed to decode models: %v", err)
	}
	if len(modelsResp.Models) != 1 || modelsResp.Models[0].ID != "text-detector-tr" {
		t.Fatalf("unexpected models list: %+v", modelsResp)
	}
}

func TestHTTP_HealthAndReadiness(t *testing.T) {
	checker := &readyCheck{err: nil}
	srv, _, _ := setupTestServer(t, 0, checker)
	handler := srv.Handler()

	// 1. GET /health -> 200
	reqH := httptest.NewRequest(http.MethodGet, "/health", nil)
	wH := httptest.NewRecorder()
	handler.ServeHTTP(wH, reqH)
	if wH.Code != http.StatusOK {
		t.Fatalf("health expected 200, got %d", wH.Code)
	}

	// 2. GET /ready -> 200 (healthy)
	reqR := httptest.NewRequest(http.MethodGet, "/ready", nil)
	wR := httptest.NewRecorder()
	handler.ServeHTTP(wR, reqR)
	if wR.Code != http.StatusOK {
		t.Fatalf("ready expected 200, got %d", wR.Code)
	}

	// 3. GET /ready -> 503 (dependency failed)
	checker.err = errors.New("database connection refused")
	wR2 := httptest.NewRecorder()
	handler.ServeHTTP(wR2, reqR)
	if wR2.Code != http.StatusServiceUnavailable {
		t.Fatalf("ready expected 503, got %d", wR2.Code)
	}
}

func TestHTTP_SecurityAndMiddlewares(t *testing.T) {
	// Max body size = 64 bytes
	srv, _, _ := setupTestServer(t, 64, nil)
	handler := srv.Handler()

	// 1. Oversized body -> 413 Payload Too Large
	largePayload := `{"input":{"text":"` + strings.Repeat("A", 200) + `"}}`
	reqLarge := httptest.NewRequest(http.MethodPost, "/v1/models/text-detector-tr/predict", strings.NewReader(largePayload))
	reqLarge.Header.Set("Content-Type", "application/json")
	wLarge := httptest.NewRecorder()
	handler.ServeHTTP(wLarge, reqLarge)

	if wLarge.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected 413, got %d: %s", wLarge.Code, wLarge.Body.String())
	}

	// 2. Unsupported Content-Type -> 415
	reqCT := httptest.NewRequest(http.MethodPost, "/v1/models/text-detector-tr/predict", strings.NewReader(`{"text":"hello"}`))
	reqCT.Header.Set("Content-Type", "text/plain")
	wCT := httptest.NewRecorder()
	handler.ServeHTTP(wCT, reqCT)

	if wCT.Code != http.StatusUnsupportedMediaType {
		t.Fatalf("expected 415, got %d: %s", wCT.Code, wCT.Body.String())
	}

	// 3. Panic Recovery -> 500
	panicPayload := `{"input":{"text":"panic"}}`
	reqPanic := httptest.NewRequest(http.MethodPost, "/v1/models/text-detector-tr/predict", strings.NewReader(panicPayload))
	reqPanic.Header.Set("Content-Type", "application/json")
	wPanic := httptest.NewRecorder()
	handler.ServeHTTP(wPanic, reqPanic)

	if wPanic.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500 on panic recovery, got %d: %s", wPanic.Code, wPanic.Body.String())
	}

	// 4. Client-supplied X-Request-ID preserved
	reqID := httptest.NewRequest(http.MethodGet, "/health", nil)
	reqID.Header.Set("X-Request-ID", "custom-client-id-999")
	wID := httptest.NewRecorder()
	handler.ServeHTTP(wID, reqID)

	if wID.Header().Get("X-Request-ID") != "custom-client-id-999" {
		t.Fatalf("expected preserved X-Request-ID, got: %s", wID.Header().Get("X-Request-ID"))
	}
}

func TestHTTP_LegacyPredictCompatibility(t *testing.T) {
	srv, _, _ := setupTestServer(t, 0, nil)
	handler := srv.Handler()

	// POST /predict (legacy Python format)
	legacyBody := `{"text":"Eski Python istemcisinden gelen istek.","language":"tr"}`
	req := httptest.NewRequest(http.MethodPost, "/predict", strings.NewReader(legacyBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 for legacy endpoint, got %d: %s", w.Code, w.Body.String())
	}

	var resp internalHTTP.LegacyPredictResponse
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Model.ID != "text-detector-tr" || resp.Result["label"] != "AI" {
		t.Fatalf("unexpected legacy response: %+v", resp)
	}
}

func TestHTTP_GracefulShutdown(t *testing.T) {
	srv, _, q := setupTestServer(t, 0, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		t.Fatalf("shutdown failed: %v", err)
	}

	// Queue should be closed
	if err := q.Enqueue(ctx, "test-job"); !errors.Is(err, domain.ErrRuntimeClosed) {
		t.Fatalf("expected queue to be closed after server shutdown, got: %v", err)
	}
}
