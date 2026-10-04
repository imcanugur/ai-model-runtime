package queue

import (
	"context"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

// JobQueue defines the abstract queue contract for scheduling and consuming asynchronous jobs.
type JobQueue interface {
	// Enqueue pushes a job ID to the back of the queue.
	Enqueue(ctx context.Context, jobID domain.JobID) error

	// Dequeue pops a job ID from the front of the queue, blocking up to timeout duration.
	// Returns domain.ErrQueueEmpty if timeout expires without a job.
	Dequeue(ctx context.Context, timeout time.Duration) (domain.JobID, error)

	// Requeue returns a job ID to the front or back of the queue for retry processing.
	Requeue(ctx context.Context, jobID domain.JobID) error

	// Depth returns the current number of pending items in the queue.
	Depth(ctx context.Context) (int64, error)

	// Close terminates queue listeners and releases connection resources.
	Close() error
}
