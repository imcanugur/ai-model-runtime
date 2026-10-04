package memory

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

// MemoryJobStore is an in-memory, thread-safe implementation of jobstore.JobStore.
type MemoryJobStore struct {
	mu   sync.RWMutex
	jobs map[domain.JobID]*domain.Job
}

// NewMemoryJobStore creates an initialized MemoryJobStore.
func NewMemoryJobStore() *MemoryJobStore {
	return &MemoryJobStore{
		jobs: make(map[domain.JobID]*domain.Job),
	}
}

// Create persists a new job in JobQueued state.
func (s *MemoryJobStore) Create(ctx context.Context, job *domain.Job) error {
	if job == nil {
		return domain.NewValidationError("job", "job cannot be nil", domain.ErrInvalidManifest)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.jobs[job.ID]; exists {
		return fmt.Errorf("job with ID %q already exists", job.ID)
	}

	cloned := cloneJob(job)
	cloned.State = domain.JobQueued
	s.jobs[job.ID] = cloned
	return nil
}

// Get returns a copy of the job. Returns ErrJobNotFound if not found.
func (s *MemoryJobStore) Get(ctx context.Context, id domain.JobID) (*domain.Job, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	job, exists := s.jobs[id]
	if !exists {
		return nil, domain.ErrJobNotFound
	}
	return cloneJob(job), nil
}

// Claim atomically claims a job for a worker with a lease timestamp.
func (s *MemoryJobStore) Claim(ctx context.Context, id domain.JobID, workerID string, leaseDuration time.Duration) (*domain.Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	job, exists := s.jobs[id]
	if !exists {
		return nil, domain.ErrJobNotFound
	}

	now := time.Now().UTC()

	// Eligible if queued OR if running but lease is expired (orphan recovery)
	isQueued := job.State == domain.JobQueued
	isStale := job.State == domain.JobRunning && job.LeaseUntil != nil && job.LeaseUntil.Before(now)

	if !isQueued && !isStale {
		return nil, domain.ErrJobAlreadyClaimed
	}

	if err := domain.ValidateTransition(job.State, domain.JobRunning); err != nil {
		return nil, err
	}

	lease := now.Add(leaseDuration)
	job.State = domain.JobRunning
	job.WorkerID = workerID
	job.StartedAt = &now
	job.LeaseUntil = &lease
	job.Attempt++

	return cloneJob(job), nil
}

// Heartbeat refreshes the active lease for a running job.
func (s *MemoryJobStore) Heartbeat(ctx context.Context, id domain.JobID, workerID string, leaseDuration time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	job, exists := s.jobs[id]
	if !exists {
		return domain.ErrJobNotFound
	}

	if job.State != domain.JobRunning || job.WorkerID != workerID {
		return domain.ErrJobAlreadyClaimed
	}

	now := time.Now().UTC()
	newLease := now.Add(leaseDuration)
	job.LeaseUntil = &newLease
	return nil
}

// Complete sets the job state to JobCompleted and records the prediction result.
func (s *MemoryJobStore) Complete(ctx context.Context, id domain.JobID, workerID string, result *domain.PredictionResult) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	job, exists := s.jobs[id]
	if !exists {
		return domain.ErrJobNotFound
	}

	if err := domain.ValidateTransition(job.State, domain.JobCompleted); err != nil {
		return err
	}

	if job.WorkerID != workerID {
		return domain.ErrJobAlreadyClaimed
	}

	now := time.Now().UTC()
	job.State = domain.JobCompleted
	job.CompletedAt = &now
	job.Result = result
	job.LeaseUntil = nil
	return nil
}

// Fail transitions the job to JobFailed and records the error string.
func (s *MemoryJobStore) Fail(ctx context.Context, id domain.JobID, workerID string, errMsg string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	job, exists := s.jobs[id]
	if !exists {
		return domain.ErrJobNotFound
	}

	if err := domain.ValidateTransition(job.State, domain.JobFailed); err != nil {
		return err
	}

	now := time.Now().UTC()
	job.State = domain.JobFailed
	job.CompletedAt = &now
	job.Error = errMsg
	job.LeaseUntil = nil
	return nil
}

// Retry transitions the job from JobRunning back to JobQueued and clears the worker lease.
func (s *MemoryJobStore) Retry(ctx context.Context, id domain.JobID, workerID string, errMsg string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	job, exists := s.jobs[id]
	if !exists {
		return domain.ErrJobNotFound
	}

	if err := domain.ValidateTransition(job.State, domain.JobQueued); err != nil {
		return err
	}

	job.State = domain.JobQueued
	job.WorkerID = ""
	job.Error = errMsg
	job.LeaseUntil = nil
	return nil
}

// FindStaleJobs returns IDs of running jobs with expired leases.
func (s *MemoryJobStore) FindStaleJobs(ctx context.Context, now time.Time, limit int) ([]domain.JobID, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var stale []domain.JobID
	for id, job := range s.jobs {
		if job.State == domain.JobRunning && job.LeaseUntil != nil && job.LeaseUntil.Before(now) {
			stale = append(stale, id)
			if limit > 0 && len(stale) >= limit {
				break
			}
		}
	}
	return stale, nil
}

func cloneJob(j *domain.Job) *domain.Job {
	if j == nil {
		return nil
	}
	cpy := *j
	if j.StartedAt != nil {
		t := *j.StartedAt
		cpy.StartedAt = &t
	}
	if j.CompletedAt != nil {
		t := *j.CompletedAt
		cpy.CompletedAt = &t
	}
	if j.LeaseUntil != nil {
		t := *j.LeaseUntil
		cpy.LeaseUntil = &t
	}
	if j.Result != nil {
		r := *j.Result
		cpy.Result = &r
	}
	if j.Metadata != nil {
		cpy.Metadata = make(map[string]string, len(j.Metadata))
		for k, v := range j.Metadata {
			cpy.Metadata[k] = v
		}
	}
	return &cpy
}
