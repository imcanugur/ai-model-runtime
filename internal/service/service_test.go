package service_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	storeMemory "github.com/imcanugur/ai-model-runtime/internal/jobstore/memory"
	queueMemory "github.com/imcanugur/ai-model-runtime/internal/queue/memory"
	"github.com/imcanugur/ai-model-runtime/internal/service"
)

type mockHandle struct {
	id      domain.ModelID
	version domain.ModelVersion
}

func (m *mockHandle) ID() domain.ModelID           { return m.id }
func (m *mockHandle) Version() domain.ModelVersion { return m.version }
func (m *mockHandle) Key() domain.ModelKey         { return domain.NewModelKey(m.id, m.version) }
func (m *mockHandle) Execute(ctx context.Context, input domain.RawInput) (*domain.PredictionResult, error) {
	if input.Text == "fail" {
		return nil, errors.New("pipeline execution error")
	}
	return &domain.PredictionResult{
		ModelID:      m.id,
		ModelVersion: m.version,
		PrimaryScore: 0.95,
		Label:        "AI",
		Confidence:   0.95,
	}, nil
}
func (m *mockHandle) Release() {}

type mockRegistry struct {
	acquireCount atomic.Int64
}

func (r *mockRegistry) Discover(ctx context.Context, rootDir string) error    { return nil }
func (r *mockRegistry) Load(ctx context.Context, key domain.ModelKey) error   { return nil }
func (r *mockRegistry) Unload(ctx context.Context, key domain.ModelKey) error { return nil }
func (r *mockRegistry) List() []domain.ModelDescriptor {
	return []domain.ModelDescriptor{
		{
			Key: "test-model:1.0.0",
			Metadata: domain.ModelMetadata{
				ID:      "test-model",
				Version: "1.0.0",
				Name:    "Test Model",
			},
			State: domain.StateReady,
		},
	}
}

func (r *mockRegistry) Acquire(key domain.ModelKey) (domain.ModelHandle, error) {
	id, ver, err := key.Split()
	if err != nil {
		return nil, err
	}
	if id != "test-model" {
		return nil, domain.ErrModelNotFound
	}
	r.acquireCount.Add(1)
	return &mockHandle{id: id, version: ver}, nil
}

func (r *mockRegistry) AcquireDefault(id domain.ModelID) (domain.ModelHandle, error) {
	if id != "test-model" {
		return nil, domain.ErrModelNotFound
	}
	return r.Acquire(domain.NewModelKey(id, "1.0.0"))
}

type mockChecker struct {
	err error
}

func (c *mockChecker) IsReady(ctx context.Context) error {
	return c.err
}

func TestService_SyncPredict(t *testing.T) {
	ctx := context.Background()
	reg := &mockRegistry{}
	svc, err := service.NewInferenceService(service.ServiceConfig{
		Registry: reg,
	})
	if err != nil {
		t.Fatalf("failed to create service: %v", err)
	}

	// 1. Success with default model
	res, key, err := svc.SyncPredict(ctx, "test-model", "", domain.RawInput{Text: "hello"})
	if err != nil {
		t.Fatalf("predict failed: %v", err)
	}
	if res.Label != "AI" || key != "test-model:1.0.0" {
		t.Fatalf("unexpected result: %+v, key: %s", res, key)
	}

	// 2. Success with specific version
	res, key, err = svc.SyncPredict(ctx, "test-model", "2.0.0", domain.RawInput{Text: "hello"})
	if err != nil {
		t.Fatalf("predict with version failed: %v", err)
	}
	if key != "test-model:2.0.0" {
		t.Fatalf("expected key test-model:2.0.0, got: %s", key)
	}

	// 3. Model not found
	_, _, err = svc.SyncPredict(ctx, "unknown-model", "", domain.RawInput{Text: "hello"})
	if !errors.Is(err, domain.ErrModelNotFound) {
		t.Fatalf("expected ErrModelNotFound, got: %v", err)
	}

	// 4. Execution error
	_, _, err = svc.SyncPredict(ctx, "test-model", "", domain.RawInput{Text: "fail"})
	if err == nil {
		t.Fatal("expected execution error, got nil")
	}
}

func TestService_SubmitJobAndIdempotency(t *testing.T) {
	ctx := context.Background()
	reg := &mockRegistry{}
	store := storeMemory.NewMemoryJobStore()
	q := queueMemory.NewMemoryQueue()
	defer q.Close()

	svc, _ := service.NewInferenceService(service.ServiceConfig{
		Registry: reg,
		JobStore: store,
		Queue:    q,
	})

	// 1. Submit new job
	job1, wasExisting, err := svc.SubmitJob(ctx, "test-model", "1.0.0", domain.RawInput{Text: "job text"}, 3, "key-123")
	if err != nil {
		t.Fatalf("submit job failed: %v", err)
	}
	if wasExisting {
		t.Fatal("expected new job, got wasExisting=true")
	}
	if job1.State != domain.JobQueued {
		t.Fatalf("expected state queued, got: %s", job1.State)
	}

	depth, _ := q.Depth(ctx)
	if depth != 1 {
		t.Fatalf("expected queue depth 1, got %d", depth)
	}

	// 2. Resubmit with same Idempotency Key
	job2, wasExisting, err := svc.SubmitJob(ctx, "test-model", "1.0.0", domain.RawInput{Text: "job text"}, 3, "key-123")
	if err != nil {
		t.Fatalf("resubmit failed: %v", err)
	}
	if !wasExisting {
		t.Fatal("expected idempotent existing job, got wasExisting=false")
	}
	if job2.ID != job1.ID {
		t.Fatalf("expected same job ID, got %s vs %s", job1.ID, job2.ID)
	}

	// Queue depth must NOT increase on duplicate submit
	depth, _ = q.Depth(ctx)
	if depth != 1 {
		t.Fatalf("queue depth increased on idempotent duplicate, got %d", depth)
	}

	// 3. Get job
	fetched, err := svc.GetJob(ctx, job1.ID)
	if err != nil || fetched.ID != job1.ID {
		t.Fatalf("failed to fetch job: %v", err)
	}

	// 4. Get unknown job
	_, err = svc.GetJob(ctx, "unknown-id")
	if !errors.Is(err, domain.ErrJobNotFound) {
		t.Fatalf("expected ErrJobNotFound, got: %v", err)
	}
}

func TestService_CheckReadiness(t *testing.T) {
	ctx := context.Background()
	reg := &mockRegistry{}
	checker := &mockChecker{err: nil}

	svc, _ := service.NewInferenceService(service.ServiceConfig{
		Registry: reg,
		Checkers: map[string]service.ReadyChecker{"worker_pool": checker},
	})

	ready, deps := svc.CheckReadiness(ctx)
	if !ready || deps["worker_pool"] != "ok" {
		t.Fatalf("expected ready=true, got ready=%v, deps=%v", ready, deps)
	}

	// Failing checker
	checker.err = errors.New("redis offline")
	ready, deps = svc.CheckReadiness(ctx)
	if ready {
		t.Fatal("expected ready=false with failing checker, got true")
	}
}
