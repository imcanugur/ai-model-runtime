package recovery_test

import (
	"context"
	"testing"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	storeMemory "github.com/imcanugur/ai-model-runtime/internal/jobstore/memory"
	queueMemory "github.com/imcanugur/ai-model-runtime/internal/queue/memory"
	"github.com/imcanugur/ai-model-runtime/internal/recovery"
)

func TestRecoveryManager_StaleJobRequeued(t *testing.T) {
	ctx := context.Background()
	store := storeMemory.NewMemoryJobStore()
	q := queueMemory.NewMemoryQueue()
	defer q.Close()

	rec, err := recovery.NewRecoveryManager(recovery.RecoveryConfig{
		Store:     store,
		Queue:     q,
		BatchSize: 10,
	})
	if err != nil {
		t.Fatalf("failed to create recovery manager: %v", err)
	}

	// 1. Create and claim job with short lease (1ms)
	input := domain.RawInput{Text: "stale recovery test"}
	job := domain.NewJob("job-orphan-1", "test-model", input, 3)
	_ = store.Create(ctx, job)
	_, _ = store.Claim(ctx, job.ID, "crashed-worker-A", 1*time.Millisecond)

	// Wait for lease to expire
	time.Sleep(10 * time.Millisecond)

	// Run recovery pass
	recovered, err := rec.RecoverOnce(ctx)
	if err != nil {
		t.Fatalf("recovery failed: %v", err)
	}
	if recovered != 1 {
		t.Fatalf("expected 1 job recovered, got %d", recovered)
	}

	// Verify job is back in JobQueued state
	j, _ := store.Get(ctx, job.ID)
	if j.State != domain.JobQueued || j.WorkerID != "" {
		t.Fatalf("expected state QUEUED and empty worker, got %+v", j)
	}

	// Verify job was pushed back onto the queue
	depth, _ := q.Depth(ctx)
	if depth != 1 {
		t.Fatalf("expected queue depth 1, got %d", depth)
	}
}

func TestRecoveryManager_MaxAttemptsExhausted(t *testing.T) {
	ctx := context.Background()
	store := storeMemory.NewMemoryJobStore()
	q := queueMemory.NewMemoryQueue()
	defer q.Close()

	rec, _ := recovery.NewRecoveryManager(recovery.RecoveryConfig{
		Store:     store,
		Queue:     q,
		BatchSize: 10,
	})

	// Job with maxAttempts = 1
	input := domain.RawInput{Text: "exhaustion recovery test"}
	job := domain.NewJob("job-orphan-exhaust", "test-model", input, 1)
	_ = store.Create(ctx, job)
	// Claim once -> attempt becomes 1 (which equals maxAttempts)
	_, _ = store.Claim(ctx, job.ID, "crashed-worker-B", 1*time.Millisecond)

	time.Sleep(10 * time.Millisecond)

	recovered, err := rec.RecoverOnce(ctx)
	if err != nil {
		t.Fatalf("recovery error: %v", err)
	}
	// Job was marked failed, not requeued
	if recovered != 0 {
		t.Fatalf("expected 0 requeued jobs, got %d", recovered)
	}

	j, _ := store.Get(ctx, job.ID)
	if j.State != domain.JobFailed {
		t.Fatalf("expected JobFailed due to exhausted attempts, got: %s", j.State)
	}

	depth, _ := q.Depth(ctx)
	if depth != 0 {
		t.Fatalf("expected queue depth 0, got %d", depth)
	}
}

func TestRecoveryManager_ActiveLeaseIgnored(t *testing.T) {
	ctx := context.Background()
	store := storeMemory.NewMemoryJobStore()
	q := queueMemory.NewMemoryQueue()
	defer q.Close()

	rec, _ := recovery.NewRecoveryManager(recovery.RecoveryConfig{
		Store:     store,
		Queue:     q,
		BatchSize: 10,
	})

	// Active job with 1 hour lease
	input := domain.RawInput{Text: "healthy test"}
	job := domain.NewJob("job-healthy", "test-model", input, 3)
	_ = store.Create(ctx, job)
	_, _ = store.Claim(ctx, job.ID, "healthy-worker", 1*time.Hour)

	recovered, _ := rec.RecoverOnce(ctx)
	if recovered != 0 {
		t.Fatalf("expected 0 recovered jobs for healthy lease, got %d", recovered)
	}

	j, _ := store.Get(ctx, job.ID)
	if j.State != domain.JobRunning {
		t.Fatalf("healthy job was altered, state: %s", j.State)
	}
}
