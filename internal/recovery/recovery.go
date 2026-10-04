package recovery

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/jobstore"
	"github.com/imcanugur/ai-model-runtime/internal/queue"
)

// RecoveryConfig defines settings for orphaned job recovery.
type RecoveryConfig struct {
	Store      jobstore.JobStore
	Queue      queue.JobQueue
	ScanPeriod time.Duration // e.g. 10s
	BatchSize  int           // e.g. 50
	Logger     *slog.Logger
}

// RecoveryManager scans for stale running jobs whose worker lease expired and requeues them.
type RecoveryManager struct {
	store      jobstore.JobStore
	queue      queue.JobQueue
	scanPeriod time.Duration
	batchSize  int
	logger     *slog.Logger
}

// NewRecoveryManager creates an initialized RecoveryManager.
func NewRecoveryManager(cfg RecoveryConfig) (*RecoveryManager, error) {
	if cfg.Store == nil {
		return nil, errors.New("store cannot be nil")
	}
	if cfg.Queue == nil {
		return nil, errors.New("queue cannot be nil")
	}
	scanPeriod := cfg.ScanPeriod
	if scanPeriod <= 0 {
		scanPeriod = 10 * time.Second
	}
	batchSize := cfg.BatchSize
	if batchSize <= 0 {
		batchSize = 50
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &RecoveryManager{
		store:      cfg.Store,
		queue:      cfg.Queue,
		scanPeriod: scanPeriod,
		batchSize:  batchSize,
		logger:     logger,
	}, nil
}

// RecoverOnce executes a single recovery pass over stale running jobs.
// Returns the number of successfully recovered jobs.
func (r *RecoveryManager) RecoverOnce(ctx context.Context) (int, error) {
	now := time.Now().UTC()
	staleIDs, err := r.store.FindStaleJobs(ctx, now, r.batchSize)
	if err != nil {
		return 0, fmt.Errorf("failed to scan stale jobs: %w", err)
	}

	recovered := 0
	for _, id := range staleIDs {
		job, err := r.store.Get(ctx, id)
		if err != nil {
			r.logger.Warn("could not fetch stale job for recovery", "job_id", id, "error", err)
			continue
		}

		// Double-check condition
		if job.State != domain.JobRunning || job.LeaseUntil == nil || !job.LeaseUntil.Before(now) {
			continue
		}

		if job.Attempt >= job.MaxAttempts {
			// Max attempts already exceeded; mark permanently failed
			failReason := fmt.Sprintf("lease expired on worker %s and max attempts (%d) reached", job.WorkerID, job.MaxAttempts)
			if err := r.store.Fail(ctx, job.ID, job.WorkerID, failReason); err != nil {
				r.logger.Warn("failed to mark stale job as failed", "job_id", id, "error", err)
			} else {
				r.logger.Info("marked stale job as failed due to exhausted attempts", "job_id", id)
			}
			continue
		}

		// Requeue job for another attempt
		reason := fmt.Sprintf("lease expired on worker %s; orphaned job recovered", job.WorkerID)
		if err := r.store.Retry(ctx, job.ID, job.WorkerID, reason); err != nil {
			// Transition failed (e.g. race with another recovery or job finished)
			r.logger.Warn("failed to transition stale job for retry", "job_id", id, "error", err)
			continue
		}

		if err := r.queue.Requeue(ctx, job.ID); err != nil {
			r.logger.Error("failed to push recovered job to queue", "job_id", id, "error", err)
			continue
		}

		r.logger.Info("successfully recovered and requeued orphaned job", "job_id", id, "previous_worker", job.WorkerID)
		recovered++
	}

	return recovered, nil
}

// Start runs periodic recovery passes until context is canceled.
func (r *RecoveryManager) Start(ctx context.Context) {
	ticker := time.NewTicker(r.scanPeriod)
	defer ticker.Stop()

	// Initial sweep on startup
	_, _ = r.RecoverOnce(ctx)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			_, _ = r.RecoverOnce(ctx)
		}
	}
}
