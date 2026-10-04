package memory

import (
	"context"
	"sync"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

// MemoryQueue is a thread-safe in-memory FIFO queue implementing queue.JobQueue.
type MemoryQueue struct {
	mu     sync.Mutex
	items  []domain.JobID
	notify chan struct{}
	closed bool
}

// NewMemoryQueue creates an initialized MemoryQueue.
func NewMemoryQueue() *MemoryQueue {
	return &MemoryQueue{
		items:  make([]domain.JobID, 0),
		notify: make(chan struct{}, 1),
	}
}

// Enqueue adds a job ID to the back of the queue.
func (q *MemoryQueue) Enqueue(ctx context.Context, jobID domain.JobID) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	if q.closed {
		return domain.ErrRuntimeClosed
	}

	q.items = append(q.items, jobID)
	q.signal()
	return nil
}

// Dequeue retrieves the next job ID, waiting up to timeout or context expiration.
func (q *MemoryQueue) Dequeue(ctx context.Context, timeout time.Duration) (domain.JobID, error) {
	deadline := time.Now().Add(timeout)

	for {
		q.mu.Lock()
		if q.closed {
			q.mu.Unlock()
			return "", domain.ErrRuntimeClosed
		}
		if len(q.items) > 0 {
			item := q.items[0]
			q.items = q.items[1:]
			q.mu.Unlock()
			return item, nil
		}
		q.mu.Unlock()

		remaining := time.Until(deadline)
		if remaining <= 0 {
			return "", domain.ErrQueueEmpty
		}

		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(remaining):
			return "", domain.ErrQueueEmpty
		case <-q.notify:
			// Item arrived, re-check queue
		}
	}
}

// Requeue adds a job ID back to the queue.
func (q *MemoryQueue) Requeue(ctx context.Context, jobID domain.JobID) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	if q.closed {
		return domain.ErrRuntimeClosed
	}

	// Prepend for high retry priority
	q.items = append([]domain.JobID{jobID}, q.items...)
	q.signal()
	return nil
}

// Depth returns the current queue length.
func (q *MemoryQueue) Depth(ctx context.Context) (int64, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	return int64(len(q.items)), nil
}

// Close marks the queue as closed and wakes any waiting dequeuers.
func (q *MemoryQueue) Close() error {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.closed = true
	q.signal()
	return nil
}

func (q *MemoryQueue) signal() {
	select {
	case q.notify <- struct{}{}:
	default:
	}
}
