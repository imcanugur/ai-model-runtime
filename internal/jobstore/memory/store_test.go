package memory_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/jobstore/memory"
)

func TestMemoryJobStore_CreateAndGet(t *testing.T) {
	store := memory.NewMemoryJobStore()
	ctx := context.Background()

	input := domain.RawInput{Text: "test text for model"}
	job := domain.NewJob("job-001", "text-detector-tr", input, 3)

	if err := store.Create(ctx, job); err != nil {
		t.Fatalf("failed to create job: %v", err)
	}

	// Duplicate create should error
	if err := store.Create(ctx, job); err == nil {
		t.Fatal("expected duplicate create to error, got nil")
	}

	// Retrieve
	fetched, err := store.Get(ctx, "job-001")
	if err != nil {
		t.Fatalf("failed to get job: %v", err)
	}
	if fetched.ID != "job-001" || fetched.State != domain.JobQueued {
		t.Fatalf("unexpected fetched job state: %+v", fetched)
	}

	// Non-existent job
	_, err = store.Get(ctx, "non-existent")
	if !errors.Is(err, domain.ErrJobNotFound) {
		t.Fatalf("expected ErrJobNotFound, got: %v", err)
	}
}

func TestMemoryJobStore_ClaimIdempotencyRace(t *testing.T) {
	store := memory.NewMemoryJobStore()
	ctx := context.Background()

	input := domain.RawInput{Text: "race test"}
	job := domain.NewJob("job-race", "text-detector-tr", input, 3)
	if err := store.Create(ctx, job); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	// Two workers try to claim the exact same job simultaneously
	const workerCount = 10
	var wg sync.WaitGroup
	wg.Add(workerCount)

	successCount := 0
	claimedErrCount := 0
	var mu sync.Mutex

	for i := 0; i < workerCount; i++ {
		workerID := "worker-" + string(rune('A'+i))
		go func() {
			defer wg.Done()
			_, err := store.Claim(ctx, "job-race", workerID, 10*time.Second)
			mu.Lock()
			defer mu.Unlock()
			if err == nil {
				successCount++
			} else if errors.Is(err, domain.ErrJobAlreadyClaimed) {
				claimedErrCount++
			}
		}()
	}

	wg.Wait()

	if successCount != 1 {
		t.Fatalf("expected exactly 1 worker to claim the job, got %d", successCount)
	}
	if claimedErrCount != workerCount-1 {
		t.Fatalf("expected %d workers to receive ErrJobAlreadyClaimed, got %d", workerCount-1, claimedErrCount)
	}

	// Confirm final state
	j, _ := store.Get(ctx, "job-race")
	if j.State != domain.JobRunning || j.Attempt != 1 {
		t.Fatalf("unexpected job state after claim: %+v", j)
	}
}

func TestMemoryJobStore_StateTransitions(t *testing.T) {
	store := memory.NewMemoryJobStore()
	ctx := context.Background()

	input := domain.RawInput{Text: "state test"}
	job := domain.NewJob("job-state", "text-detector-tr", input, 3)
	_ = store.Create(ctx, job)

	// 1. Invalid direct transition: QUEUED -> COMPLETED
	err := store.Complete(ctx, "job-state", "worker-1", &domain.PredictionResult{})
	if err == nil {
		t.Fatal("expected error completing queued job directly, got nil")
	}

	// 2. Claim -> RUNNING
	claimed, err := store.Claim(ctx, "job-state", "worker-1", 10*time.Second)
	if err != nil {
		t.Fatalf("claim failed: %v", err)
	}
	if claimed.State != domain.JobRunning {
		t.Fatalf("expected RUNNING state, got: %s", claimed.State)
	}

	// 3. Heartbeat
	if err := store.Heartbeat(ctx, "job-state", "worker-1", 15*time.Second); err != nil {
		t.Fatalf("heartbeat failed: %v", err)
	}

	// 4. Heartbeat by wrong worker rejected
	if err := store.Heartbeat(ctx, "job-state", "imposter", 15*time.Second); !errors.Is(err, domain.ErrJobAlreadyClaimed) {
		t.Fatalf("expected ErrJobAlreadyClaimed for imposter, got: %v", err)
	}

	// 5. Retry transition: RUNNING -> QUEUED
	if err := store.Retry(ctx, "job-state", "worker-1", "transient error"); err != nil {
		t.Fatalf("retry failed: %v", err)
	}

	j, _ := store.Get(ctx, "job-state")
	if j.State != domain.JobQueued || j.WorkerID != "" {
		t.Fatalf("expected reset to QUEUED with empty worker, got %+v", j)
	}

	// 6. Claim again
	_, err = store.Claim(ctx, "job-state", "worker-2", 10*time.Second)
	if err != nil {
		t.Fatalf("re-claim failed: %v", err)
	}

	// 7. Complete: RUNNING -> COMPLETED
	res := &domain.PredictionResult{
		ModelID:      "text-detector-tr",
		ModelVersion: "1.0.0",
		PrimaryScore: 0.95,
		Label:        "AI",
		Confidence:   0.95,
	}
	if err := store.Complete(ctx, "job-state", "worker-2", res); err != nil {
		t.Fatalf("complete failed: %v", err)
	}

	completedJob, _ := store.Get(ctx, "job-state")
	if completedJob.State != domain.JobCompleted || completedJob.Result == nil {
		t.Fatalf("expected completed job with result, got: %+v", completedJob)
	}

	// 8. Terminal state cannot transition
	if err := store.Fail(ctx, "job-state", "worker-2", "cannot fail completed"); err == nil {
		t.Fatal("expected error transitioning from terminal state COMPLETED, got nil")
	}
}

func TestMemoryJobStore_FindStaleJobs(t *testing.T) {
	store := memory.NewMemoryJobStore()
	ctx := context.Background()

	input := domain.RawInput{Text: "stale test"}
	job := domain.NewJob("job-stale", "text-detector-tr", input, 3)
	_ = store.Create(ctx, job)

	// Claim with very short lease (1 millisecond)
	_, _ = store.Claim(ctx, "job-stale", "worker-1", 1*time.Millisecond)

	time.Sleep(10 * time.Millisecond) // Ensure lease expires

	stale, err := store.FindStaleJobs(ctx, time.Now().UTC(), 10)
	if err != nil {
		t.Fatalf("failed to find stale jobs: %v", err)
	}

	if len(stale) != 1 || stale[0] != "job-stale" {
		t.Fatalf("expected [job-stale], got %v", stale)
	}
}
