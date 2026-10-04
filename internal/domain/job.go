package domain

import (
	"errors"
	"fmt"
	"time"
)

// JobID represents the globally unique identifier of an asynchronous job.
type JobID string

// JobState represents the execution lifecycle stage of a Job.
type JobState string

const (
	JobQueued    JobState = "queued"
	JobRunning   JobState = "running"
	JobCompleted JobState = "completed"
	JobFailed    JobState = "failed"
)

// Additional job and queue domain sentinel errors.
var (
	ErrInvalidJobStateTransition = errors.New("invalid job state transition")
	ErrJobAlreadyClaimed         = errors.New("job already claimed by another worker")
	ErrJobNotFound               = errors.New("job not found")
	ErrQueueEmpty                = errors.New("queue is empty")
	ErrMaxAttemptsExceeded       = errors.New("maximum job retry attempts exceeded")
	ErrLeaseExpired              = errors.New("job worker lease has expired")
)

// RetryableError allows errors to declare whether a failure is transient or permanent.
type RetryableError interface {
	error
	Retryable() bool
}

// Job represents an asynchronous execution task processed through the platform queue.
type Job struct {
	ID          JobID             `json:"id"`
	ModelKey    ModelKey          `json:"model_key,omitempty"` // Specific model:version if pinned
	ModelID     ModelID           `json:"model_id"`            // Target model family
	Input       RawInput          `json:"input"`
	State       JobState          `json:"state"`
	Attempt     int               `json:"attempt"`
	MaxAttempts int               `json:"max_attempts"`
	WorkerID    string            `json:"worker_id,omitempty"`
	CreatedAt   time.Time         `json:"created_at"`
	StartedAt   *time.Time        `json:"started_at,omitempty"`
	CompletedAt *time.Time        `json:"completed_at,omitempty"`
	LeaseUntil  *time.Time        `json:"lease_until,omitempty"`
	Result      *PredictionResult `json:"result,omitempty"`
	Error       string            `json:"error,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// NewJob creates an initialized job in the JobQueued state.
func NewJob(id JobID, modelID ModelID, input RawInput, maxAttempts int) *Job {
	if maxAttempts <= 0 {
		maxAttempts = 3
	}
	return &Job{
		ID:          id,
		ModelID:     modelID,
		Input:       input,
		State:       JobQueued,
		Attempt:     0,
		MaxAttempts: maxAttempts,
		CreatedAt:   time.Now().UTC(),
		Metadata:    make(map[string]string),
	}
}

// CanTransition validates if a state change conforms to the finite state machine:
//
//	QUEUED  -> RUNNING
//	RUNNING -> COMPLETED
//	RUNNING -> FAILED
//	RUNNING -> QUEUED (retry / recovery)
func CanTransition(from, to JobState) bool {
	switch from {
	case JobQueued:
		return to == JobRunning
	case JobRunning:
		return to == JobCompleted || to == JobFailed || to == JobQueued
	case JobCompleted, JobFailed:
		return false // Terminal states cannot transition
	default:
		return false
	}
}

// ValidateTransition returns an error if transitioning from one state to another is disallowed.
func ValidateTransition(from, to JobState) error {
	if !CanTransition(from, to) {
		return fmt.Errorf("%w: cannot transition from %q to %q", ErrInvalidJobStateTransition, from, to)
	}
	return nil
}

// IsRetryable determines whether an error warrants a job retry.
// Non-retryable errors include validation errors, unknown models, and corrupt manifests.
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}
	var rErr RetryableError
	if errors.As(err, &rErr) {
		return rErr.Retryable()
	}

	// Permanent non-retryable domain errors
	if errors.Is(err, ErrInvalidManifest) ||
		errors.Is(err, ErrInvalidModelID) ||
		errors.Is(err, ErrInvalidModelVersion) ||
		errors.Is(err, ErrModelNotFound) ||
		errors.Is(err, ErrInvalidTensor) ||
		errors.Is(err, ErrEmptyPipeline) {
		return false
	}

	// Transient default for unknown runtime or network hiccups
	return true
}
