package memory_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/queue/memory"
)

func TestMemoryQueue_EnqueueDequeue(t *testing.T) {
	q := memory.NewMemoryQueue()
	defer q.Close()

	ctx := context.Background()

	// 1. Initial depth should be 0
	depth, err := q.Depth(ctx)
	if err != nil || depth != 0 {
		t.Fatalf("expected depth 0, got %d, err: %v", depth, err)
	}

	// 2. Enqueue two jobs
	j1 := domain.JobID("job-1")
	j2 := domain.JobID("job-2")

	if err := q.Enqueue(ctx, j1); err != nil {
		t.Fatalf("enqueue j1 failed: %v", err)
	}
	if err := q.Enqueue(ctx, j2); err != nil {
		t.Fatalf("enqueue j2 failed: %v", err)
	}

	depth, _ = q.Depth(ctx)
	if depth != 2 {
		t.Fatalf("expected depth 2, got %d", depth)
	}

	// 3. Dequeue in FIFO order
	item, err := q.Dequeue(ctx, 100*time.Millisecond)
	if err != nil || item != j1 {
		t.Fatalf("expected %s, got %s, err: %v", j1, item, err)
	}

	item, err = q.Dequeue(ctx, 100*time.Millisecond)
	if err != nil || item != j2 {
		t.Fatalf("expected %s, got %s, err: %v", j2, item, err)
	}

	// 4. Dequeue on empty should return ErrQueueEmpty after timeout
	_, err = q.Dequeue(ctx, 20*time.Millisecond)
	if !errors.Is(err, domain.ErrQueueEmpty) {
		t.Fatalf("expected ErrQueueEmpty, got: %v", err)
	}
}

func TestMemoryQueue_Requeue(t *testing.T) {
	q := memory.NewMemoryQueue()
	defer q.Close()

	ctx := context.Background()
	j1 := domain.JobID("job-1")
	j2 := domain.JobID("job-2")

	_ = q.Enqueue(ctx, j1)
	_ = q.Requeue(ctx, j2) // Requeue has high priority (prepended)

	item, err := q.Dequeue(ctx, 100*time.Millisecond)
	if err != nil || item != j2 {
		t.Fatalf("expected requeued item %s first, got %s, err: %v", j2, item, err)
	}

	item, err = q.Dequeue(ctx, 100*time.Millisecond)
	if err != nil || item != j1 {
		t.Fatalf("expected %s, got %s, err: %v", j1, item, err)
	}
}

func TestMemoryQueue_ContextCancellation(t *testing.T) {
	q := memory.NewMemoryQueue()
	defer q.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	_, err := q.Dequeue(ctx, 1*time.Second)
	if !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context deadline error, got: %v", err)
	}
}
