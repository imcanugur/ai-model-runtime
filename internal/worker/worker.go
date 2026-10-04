package worker

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/jobstore"
	"github.com/imcanugur/ai-model-runtime/internal/queue"
	"github.com/imcanugur/ai-model-runtime/internal/registry"
)

// Sleeper allows custom sleeping strategies (useful for deterministic tests).
type Sleeper interface {
	Sleep(ctx context.Context, d time.Duration) error
}

// DefaultSleeper implements Sleeper using standard time.After.
type DefaultSleeper struct{}

func (s DefaultSleeper) Sleep(ctx context.Context, d time.Duration) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(d):
		return nil
	}
}

// JobEvent records execution lifecycle transitions for observability.
type JobEvent struct {
	JobID    domain.JobID
	ModelKey domain.ModelKey
	ModelID  domain.ModelID
	Attempt  int
	WorkerID string
	State    domain.JobState
	Duration time.Duration
	Error    error
}

// JobObserver receives structured job lifecycle notifications.
type JobObserver interface {
	OnJobEvent(event JobEvent)
}

// SlogJobObserver logs events via log/slog.
type SlogJobObserver struct {
	logger *slog.Logger
}

func NewSlogJobObserver(logger *slog.Logger) *SlogJobObserver {
	if logger == nil {
		logger = slog.Default()
	}
	return &SlogJobObserver{logger: logger}
}

func (o *SlogJobObserver) OnJobEvent(e JobEvent) {
	attrs := []any{
		"job_id", string(e.JobID),
		"model_id", string(e.ModelID),
		"model_key", string(e.ModelKey),
		"attempt", e.Attempt,
		"worker_id", e.WorkerID,
		"state", string(e.State),
		"duration_ms", e.Duration.Milliseconds(),
	}
	if e.Error != nil {
		attrs = append(attrs, "error", e.Error.Error())
		o.logger.Warn("job state transitioned with error", attrs...)
	} else {
		o.logger.Info("job state transitioned", attrs...)
	}
}

// CalculateBackoff computes exponential backoff: base * 2^(attempt-1), capped at max.
func CalculateBackoff(attempt int, base, max time.Duration) time.Duration {
	if attempt <= 0 {
		attempt = 1
	}
	shift := attempt - 1
	if shift > 30 {
		return max
	}
	factor := 1 << shift
	backoff := base * time.Duration(factor)
	if backoff > max || backoff <= 0 {
		return max
	}
	return backoff
}

// Worker executes jobs pulled from a JobQueue.
type Worker struct {
	id              string
	queue           queue.JobQueue
	store           jobstore.JobStore
	registry        registry.ModelRegistry
	leaseDuration   time.Duration
	heartbeatPeriod time.Duration
	dequeueTimeout  time.Duration
	backoffBase     time.Duration
	backoffMax      time.Duration
	sleeper         Sleeper
	observer        JobObserver
}

// WorkerConfig parameterizes a Worker instance.
type WorkerConfig struct {
	ID              string
	Queue           queue.JobQueue
	Store           jobstore.JobStore
	Registry        registry.ModelRegistry
	LeaseDuration   time.Duration
	HeartbeatPeriod time.Duration
	DequeueTimeout  time.Duration
	BackoffBase     time.Duration
	BackoffMax      time.Duration
	Sleeper         Sleeper
	Observer        JobObserver
}

// NewWorker constructs a Worker with validated defaults.
func NewWorker(cfg WorkerConfig) (*Worker, error) {
	if cfg.ID == "" {
		return nil, errors.New("worker ID cannot be empty")
	}
	if cfg.Queue == nil {
		return nil, errors.New("queue cannot be nil")
	}
	if cfg.Store == nil {
		return nil, errors.New("store cannot be nil")
	}
	if cfg.Registry == nil {
		return nil, errors.New("registry cannot be nil")
	}

	lease := cfg.LeaseDuration
	if lease <= 0 {
		lease = 30 * time.Second
	}
	hb := cfg.HeartbeatPeriod
	if hb <= 0 {
		hb = 10 * time.Second
	}
	dqTimeout := cfg.DequeueTimeout
	if dqTimeout <= 0 {
		dqTimeout = 2 * time.Second
	}
	bBase := cfg.BackoffBase
	if bBase <= 0 {
		bBase = 100 * time.Millisecond
	}
	bMax := cfg.BackoffMax
	if bMax <= 0 {
		bMax = 5 * time.Second
	}
	sleeper := cfg.Sleeper
	if sleeper == nil {
		sleeper = DefaultSleeper{}
	}
	observer := cfg.Observer
	if observer == nil {
		observer = NewSlogJobObserver(nil)
	}

	return &Worker{
		id:              cfg.ID,
		queue:           cfg.Queue,
		store:           cfg.Store,
		registry:        cfg.Registry,
		leaseDuration:   lease,
		heartbeatPeriod: hb,
		dequeueTimeout:  dqTimeout,
		backoffBase:     bBase,
		backoffMax:      bMax,
		sleeper:         sleeper,
		observer:        observer,
	}, nil
}

// ID returns the unique identifier of this worker.
func (w *Worker) ID() string {
	return w.id
}

// ProcessOne attempts to dequeue and process a single job.
// Returns processed=true if a job was found and attempted, or false if queue was empty / timed out.
func (w *Worker) ProcessOne(ctx context.Context) (bool, error) {
	jobID, err := w.queue.Dequeue(ctx, w.dequeueTimeout)
	if err != nil {
		if errors.Is(err, domain.ErrQueueEmpty) {
			return false, nil
		}
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return false, ctx.Err()
		}
		return false, fmt.Errorf("worker %s dequeue error: %w", w.id, err)
	}

	startTime := time.Now()

	// Atomically claim the job
	job, err := w.store.Claim(ctx, jobID, w.id, w.leaseDuration)
	if err != nil {
		if errors.Is(err, domain.ErrJobAlreadyClaimed) {
			// Another worker claimed or job already processed; safely skip
			return true, nil
		}
		return true, fmt.Errorf("worker %s failed to claim job %s: %w", w.id, jobID, err)
	}

	w.observer.OnJobEvent(JobEvent{
		JobID:    job.ID,
		ModelKey: job.ModelKey,
		ModelID:  job.ModelID,
		Attempt:  job.Attempt,
		WorkerID: w.id,
		State:    domain.JobRunning,
		Duration: 0,
		Error:    nil,
	})

	// Start active heartbeat loop in background for the claimed job
	hbCtx, hbCancel := context.WithCancel(ctx)
	defer hbCancel()
	go w.runHeartbeat(hbCtx, job.ID)

	// Execute job
	execErr := w.executeJob(ctx, job)
	duration := time.Since(startTime)

	if execErr == nil {
		w.observer.OnJobEvent(JobEvent{
			JobID:    job.ID,
			ModelKey: job.ModelKey,
			ModelID:  job.ModelID,
			Attempt:  job.Attempt,
			WorkerID: w.id,
			State:    domain.JobCompleted,
			Duration: duration,
			Error:    nil,
		})
		return true, nil
	}

	// Handle execution failure: check retry eligibility
	isRetryable := domain.IsRetryable(execErr)
	hasAttemptsLeft := job.Attempt < job.MaxAttempts

	if isRetryable && hasAttemptsLeft {
		backoff := CalculateBackoff(job.Attempt, w.backoffBase, w.backoffMax)

		// Transition back to QUEUED in store
		retryErr := w.store.Retry(ctx, job.ID, w.id, execErr.Error())
		if retryErr != nil {
			return true, fmt.Errorf("worker %s failed to mark job %s for retry: %w", w.id, job.ID, retryErr)
		}

		// Wait backoff duration before re-adding to queue
		_ = w.sleeper.Sleep(ctx, backoff)

		// Put back onto the queue
		if qErr := w.queue.Requeue(ctx, job.ID); qErr != nil {
			return true, fmt.Errorf("worker %s failed to requeue job %s: %w", w.id, job.ID, qErr)
		}

		w.observer.OnJobEvent(JobEvent{
			JobID:    job.ID,
			ModelKey: job.ModelKey,
			ModelID:  job.ModelID,
			Attempt:  job.Attempt,
			WorkerID: w.id,
			State:    domain.JobQueued,
			Duration: duration,
			Error:    execErr,
		})
		return true, nil
	}

	// Non-retryable or retry attempts exhausted -> Mark as JobFailed
	failReason := execErr.Error()
	if !hasAttemptsLeft {
		failReason = fmt.Sprintf("max attempts (%d) reached: %s", job.MaxAttempts, execErr.Error())
	}

	if failErr := w.store.Fail(ctx, job.ID, w.id, failReason); failErr != nil {
		return true, fmt.Errorf("worker %s failed to mark job %s as failed: %w", w.id, job.ID, failErr)
	}

	w.observer.OnJobEvent(JobEvent{
		JobID:    job.ID,
		ModelKey: job.ModelKey,
		ModelID:  job.ModelID,
		Attempt:  job.Attempt,
		WorkerID: w.id,
		State:    domain.JobFailed,
		Duration: duration,
		Error:    execErr,
	})

	return true, nil
}

func (w *Worker) executeJob(ctx context.Context, job *domain.Job) error {
	// Acquire model handle
	var handle domain.ModelHandle
	var err error

	if job.ModelKey != "" {
		handle, err = w.registry.Acquire(job.ModelKey)
	} else {
		handle, err = w.registry.AcquireDefault(job.ModelID)
	}

	if err != nil {
		return fmt.Errorf("model acquisition failed: %w", err)
	}
	defer handle.Release()

	// Execute pipeline inference
	result, err := handle.Execute(ctx, job.Input)
	if err != nil {
		return fmt.Errorf("pipeline execution failed: %w", err)
	}

	// Persist completion and result
	if err := w.store.Complete(ctx, job.ID, w.id, result); err != nil {
		return fmt.Errorf("failed to persist job completion: %w", err)
	}

	return nil
}

func (w *Worker) runHeartbeat(ctx context.Context, jobID domain.JobID) {
	ticker := time.NewTicker(w.heartbeatPeriod)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Refresh lease in store
			_ = w.store.Heartbeat(ctx, jobID, w.id, w.leaseDuration)
		}
	}
}
