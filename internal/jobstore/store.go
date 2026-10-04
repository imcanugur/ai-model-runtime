package jobstore

import (
	"context"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

// JobStore defines persistence, atomic state transitions, leasing, and recovery queries for jobs.
type JobStore interface {
	// Create persists a new job in the store in the JobQueued state.
	Create(ctx context.Context, job *domain.Job) error

	// Get retrieves a job by ID. Returns domain.ErrJobNotFound if absent.
	Get(ctx context.Context, id domain.JobID) (*domain.Job, error)

	// Claim atomically transitions a job from JobQueued to JobRunning, assigns the WorkerID,
	// and sets LeaseUntil = now + leaseDuration. If already claimed or not queued, returns ErrJobAlreadyClaimed.
	Claim(ctx context.Context, id domain.JobID, workerID string, leaseDuration time.Duration) (*domain.Job, error)

	// Heartbeat refreshes the LeaseUntil timestamp for an actively running job owned by workerID.
	Heartbeat(ctx context.Context, id domain.JobID, workerID string, leaseDuration time.Duration) error

	// Complete transitions a job from JobRunning to JobCompleted and persists the PredictionResult.
	Complete(ctx context.Context, id domain.JobID, workerID string, result *domain.PredictionResult) error

	// Fail transitions a job from JobRunning to JobFailed and records the terminal error.
	Fail(ctx context.Context, id domain.JobID, workerID string, errMsg string) error

	// Retry transitions a job from JobRunning back to JobQueued, increments the attempt counter,
	// and clears the worker claim.
	Retry(ctx context.Context, id domain.JobID, workerID string, errMsg string) error

	// FindStaleJobs returns IDs of jobs currently in JobRunning whose LeaseUntil < now.
	FindStaleJobs(ctx context.Context, now time.Time, limit int) ([]domain.JobID, error)
}
