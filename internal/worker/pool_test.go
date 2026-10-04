package worker_test

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	storeMemory "github.com/imcanugur/ai-model-runtime/internal/jobstore/memory"
	queueMemory "github.com/imcanugur/ai-model-runtime/internal/queue/memory"
	"github.com/imcanugur/ai-model-runtime/internal/worker"
)

// mockHandle implements domain.ModelHandle for worker tests.
type mockHandle struct {
	id          domain.ModelID
	version     domain.ModelVersion
	executeFunc func(ctx context.Context, input domain.RawInput) (*domain.PredictionResult, error)
	released    atomic.Bool
	onRelease   func()
}

func (m *mockHandle) ID() domain.ModelID           { return m.id }
func (m *mockHandle) Version() domain.ModelVersion { return m.version }
func (m *mockHandle) Key() domain.ModelKey         { return domain.NewModelKey(m.id, m.version) }
func (m *mockHandle) Execute(ctx context.Context, input domain.RawInput) (*domain.PredictionResult, error) {
	if m.executeFunc != nil {
		return m.executeFunc(ctx, input)
	}
	return &domain.PredictionResult{
		ModelID:      m.id,
		ModelVersion: m.version,
		PrimaryScore: 0.95,
		Label:        "AI",
		Confidence:   0.95,
	}, nil
}
func (m *mockHandle) Release() {
	if m.released.CompareAndSwap(false, true) && m.onRelease != nil {
		m.onRelease()
	}
}

// mockRegistry implements registry.ModelRegistry for worker tests.
type mockRegistry struct {
	mu           sync.Mutex
	acquireCount int32
	releaseCount int32
	acquireErr   error
	executeFunc  func(ctx context.Context, input domain.RawInput) (*domain.PredictionResult, error)
}

func (r *mockRegistry) Discover(ctx context.Context, rootDir string) error    { return nil }
func (r *mockRegistry) Load(ctx context.Context, key domain.ModelKey) error   { return nil }
func (r *mockRegistry) Unload(ctx context.Context, key domain.ModelKey) error { return nil }
func (r *mockRegistry) List() []domain.ModelDescriptor                        { return nil }

func (r *mockRegistry) Acquire(key domain.ModelKey) (domain.ModelHandle, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.acquireErr != nil {
		return nil, r.acquireErr
	}
	atomic.AddInt32(&r.acquireCount, 1)
	id, ver, _ := key.Split()
	return &mockHandle{
		id:          id,
		version:     ver,
		executeFunc: r.executeFunc,
		onRelease: func() {
			atomic.AddInt32(&r.releaseCount, 1)
		},
	}, nil
}

func (r *mockRegistry) AcquireDefault(id domain.ModelID) (domain.ModelHandle, error) {
	return r.Acquire(domain.NewModelKey(id, "1.0.0"))
}

// mockSleeper provides a non-blocking sleeper for test speed.
type mockSleeper struct{}

func (s mockSleeper) Sleep(ctx context.Context, d time.Duration) error {
	return nil
}

func TestWorker_SuccessfulJob(t *testing.T) {
	ctx := context.Background()
	q := queueMemory.NewMemoryQueue()
	defer q.Close()
	store := storeMemory.NewMemoryJobStore()
	reg := &mockRegistry{}

	input := domain.RawInput{Text: "sample input text"}
	job := domain.NewJob("job-success-1", "test-model", input, 3)
	if err := store.Create(ctx, job); err != nil {
		t.Fatalf("create failed: %v", err)
	}
	if err := q.Enqueue(ctx, job.ID); err != nil {
		t.Fatalf("enqueue failed: %v", err)
	}

	w, err := worker.NewWorker(worker.WorkerConfig{
		ID:       "worker-test-1",
		Queue:    q,
		Store:    store,
		Registry: reg,
		Sleeper:  mockSleeper{},
	})
	if err != nil {
		t.Fatalf("create worker failed: %v", err)
	}

	processed, err := w.ProcessOne(ctx)
	if err != nil || !processed {
		t.Fatalf("expected job to be processed, got processed=%v, err: %v", processed, err)
	}

	// Verify job store state
	completed, err := store.Get(ctx, job.ID)
	if err != nil {
		t.Fatalf("failed to fetch job: %v", err)
	}
	if completed.State != domain.JobCompleted {
		t.Fatalf("expected state %s, got %s", domain.JobCompleted, completed.State)
	}
	if completed.Result == nil || completed.Result.PrimaryScore != 0.95 || completed.Result.Label != "AI" {
		t.Fatalf("unexpected result: %+v", completed.Result)
	}

	// Verify handle was acquired and released
	if atomic.LoadInt32(&reg.acquireCount) != 1 || atomic.LoadInt32(&reg.releaseCount) != 1 {
		t.Fatalf("handle leak: acquire=%d, release=%d", reg.acquireCount, reg.releaseCount)
	}
}

func TestWorker_PermanentFailure(t *testing.T) {
	ctx := context.Background()
	q := queueMemory.NewMemoryQueue()
	defer q.Close()
	store := storeMemory.NewMemoryJobStore()

	// Unknown model error is permanent (non-retryable)
	reg := &mockRegistry{
		acquireErr: domain.ErrModelNotFound,
	}

	input := domain.RawInput{Text: "sample input text"}
	job := domain.NewJob("job-fail-1", "unknown-model", input, 3)
	_ = store.Create(ctx, job)
	_ = q.Enqueue(ctx, job.ID)

	w, _ := worker.NewWorker(worker.WorkerConfig{
		ID:       "worker-test-fail",
		Queue:    q,
		Store:    store,
		Registry: reg,
		Sleeper:  mockSleeper{},
	})

	processed, err := w.ProcessOne(ctx)
	if err != nil || !processed {
		t.Fatalf("expected processed=true, got err: %v", err)
	}

	// Should be JobFailed immediately with NO requeue
	failedJob, _ := store.Get(ctx, job.ID)
	if failedJob.State != domain.JobFailed {
		t.Fatalf("expected JobFailed, got: %s", failedJob.State)
	}

	depth, _ := q.Depth(ctx)
	if depth != 0 {
		t.Fatalf("expected queue depth 0 (not requeued), got %d", depth)
	}
}

type retryableErr struct {
	msg string
}

func (e *retryableErr) Error() string   { return e.msg }
func (e *retryableErr) Retryable() bool { return true }

func TestWorker_RetryableFailureAndRecovery(t *testing.T) {
	ctx := context.Background()
	q := queueMemory.NewMemoryQueue()
	defer q.Close()
	store := storeMemory.NewMemoryJobStore()

	var attemptCounter int32
	reg := &mockRegistry{
		executeFunc: func(ctx context.Context, input domain.RawInput) (*domain.PredictionResult, error) {
			att := atomic.AddInt32(&attemptCounter, 1)
			if att == 1 {
				return nil, &retryableErr{msg: "temporary gpu driver glitch"}
			}
			return &domain.PredictionResult{
				ModelID:      "test-model",
				ModelVersion: "1.0.0",
				PrimaryScore: 0.88,
				Label:        "AI",
				Confidence:   0.88,
			}, nil
		},
	}

	input := domain.RawInput{Text: "retryable sample"}
	job := domain.NewJob("job-retry-1", "test-model", input, 3)
	_ = store.Create(ctx, job)
	_ = q.Enqueue(ctx, job.ID)

	w, _ := worker.NewWorker(worker.WorkerConfig{
		ID:       "worker-retry",
		Queue:    q,
		Store:    store,
		Registry: reg,
		Sleeper:  mockSleeper{},
	})

	// First pass: fails with retryable error -> requeued
	processed, err := w.ProcessOne(ctx)
	if err != nil || !processed {
		t.Fatalf("pass 1 failed: %v", err)
	}

	j1, _ := store.Get(ctx, job.ID)
	if j1.State != domain.JobQueued || j1.Attempt != 1 {
		t.Fatalf("expected QUEUED with attempt 1, got state=%s attempt=%d", j1.State, j1.Attempt)
	}

	// Queue should have the requeued item
	depth, _ := q.Depth(ctx)
	if depth != 1 {
		t.Fatalf("expected queue depth 1, got %d", depth)
	}

	// Second pass: succeeds
	processed, err = w.ProcessOne(ctx)
	if err != nil || !processed {
		t.Fatalf("pass 2 failed: %v", err)
	}

	j2, _ := store.Get(ctx, job.ID)
	if j2.State != domain.JobCompleted || j2.Attempt != 2 {
		t.Fatalf("expected COMPLETED with attempt 2, got state=%s attempt=%d", j2.State, j2.Attempt)
	}
}

func TestWorker_RetryExhaustion(t *testing.T) {
	ctx := context.Background()
	q := queueMemory.NewMemoryQueue()
	defer q.Close()
	store := storeMemory.NewMemoryJobStore()

	reg := &mockRegistry{
		executeFunc: func(ctx context.Context, input domain.RawInput) (*domain.PredictionResult, error) {
			return nil, &retryableErr{msg: "persistent transient error"}
		},
	}

	// Max 2 attempts
	input := domain.RawInput{Text: "exhaust sample"}
	job := domain.NewJob("job-exhaust-1", "test-model", input, 2)
	_ = store.Create(ctx, job)
	_ = q.Enqueue(ctx, job.ID)

	w, _ := worker.NewWorker(worker.WorkerConfig{
		ID:       "worker-exhaust",
		Queue:    q,
		Store:    store,
		Registry: reg,
		Sleeper:  mockSleeper{},
	})

	// Attempt 1 -> fails & requeues
	_, _ = w.ProcessOne(ctx)
	j, _ := store.Get(ctx, job.ID)
	if j.State != domain.JobQueued || j.Attempt != 1 {
		t.Fatalf("expected attempt 1 QUEUED, got: %+v", j)
	}

	// Attempt 2 -> fails & exhausts max attempts -> JobFailed
	_, _ = w.ProcessOne(ctx)
	j, _ = store.Get(ctx, job.ID)
	if j.State != domain.JobFailed || j.Attempt != 2 {
		t.Fatalf("expected attempt 2 JobFailed, got: %+v", j)
	}
}

func TestWorkerPool_ConcurrentAndGracefulShutdown(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	q := queueMemory.NewMemoryQueue()
	defer q.Close()
	store := storeMemory.NewMemoryJobStore()
	reg := &mockRegistry{}

	pool, err := worker.NewWorkerPool(worker.WorkerPoolConfig{
		Count:          4,
		Queue:          q,
		Store:          store,
		Registry:       reg,
		DequeueTimeout: 50 * time.Millisecond,
		Sleeper:        mockSleeper{},
	})
	if err != nil {
		t.Fatalf("failed to create worker pool: %v", err)
	}

	// Enqueue 10 jobs
	const totalJobs = 10
	for i := 0; i < totalJobs; i++ {
		input := domain.RawInput{Text: "pool text"}
		id := domain.JobID("pool-job-" + string(rune('0'+i)))
		j := domain.NewJob(id, "test-model", input, 3)
		_ = store.Create(ctx, j)
		_ = q.Enqueue(ctx, id)
	}

	if err := pool.Start(ctx); err != nil {
		t.Fatalf("pool start failed: %v", err)
	}

	// Allow pool to process jobs
	time.Sleep(100 * time.Millisecond)

	// Graceful shutdown
	stopCtx, stopCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer stopCancel()

	if err := pool.Stop(stopCtx); err != nil {
		t.Fatalf("pool graceful stop failed: %v", err)
	}

	// Verify all 10 jobs completed
	for i := 0; i < totalJobs; i++ {
		id := domain.JobID("pool-job-" + string(rune('0'+i)))
		j, err := store.Get(ctx, id)
		if err != nil || j.State != domain.JobCompleted {
			t.Fatalf("job %s not completed, state: %v, err: %v", id, j.State, err)
		}
	}

	// Verify all handles acquired were released
	if reg.acquireCount != int32(totalJobs) || reg.releaseCount != int32(totalJobs) {
		t.Fatalf("handle leak: acquire=%d, release=%d", reg.acquireCount, reg.releaseCount)
	}
}
