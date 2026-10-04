package service

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/jobstore"
	"github.com/imcanugur/ai-model-runtime/internal/queue"
	"github.com/imcanugur/ai-model-runtime/internal/registry"
)

// InferenceService provides the unified application boundary for synchronous and asynchronous model serving.
type InferenceService interface {
	SyncPredict(ctx context.Context, modelID domain.ModelID, version domain.ModelVersion, input domain.RawInput) (*domain.PredictionResult, domain.ModelKey, error)
	SubmitJob(ctx context.Context, modelID domain.ModelID, version domain.ModelVersion, input domain.RawInput, maxAttempts int, idempotencyKey string) (*domain.Job, bool, error)
	GetJob(ctx context.Context, jobID domain.JobID) (*domain.Job, error)
	ListModels(ctx context.Context) []domain.ModelDescriptor
	CheckReadiness(ctx context.Context) (bool, map[string]string)
}

// ReadyChecker defines an optional dependency health inspectable interface.
type ReadyChecker interface {
	IsReady(ctx context.Context) error
}

type defaultInferenceService struct {
	registry registry.ModelRegistry
	jobStore jobstore.JobStore
	queue    queue.JobQueue
	checkers map[string]ReadyChecker
}

// ServiceConfig holds dependency injection references for InferenceService.
type ServiceConfig struct {
	Registry registry.ModelRegistry
	JobStore jobstore.JobStore
	Queue    queue.JobQueue
	Checkers map[string]ReadyChecker
}

// NewInferenceService constructs an initialized InferenceService.
func NewInferenceService(cfg ServiceConfig) (InferenceService, error) {
	if cfg.Registry == nil {
		return nil, errors.New("registry cannot be nil")
	}
	checkers := cfg.Checkers
	if checkers == nil {
		checkers = make(map[string]ReadyChecker)
	}

	return &defaultInferenceService{
		registry: cfg.Registry,
		jobStore: cfg.JobStore,
		queue:    cfg.Queue,
		checkers: checkers,
	}, nil
}

func (s *defaultInferenceService) SyncPredict(ctx context.Context, modelID domain.ModelID, version domain.ModelVersion, input domain.RawInput) (*domain.PredictionResult, domain.ModelKey, error) {
	if err := domain.ValidateModelID(modelID); err != nil {
		return nil, "", err
	}

	var handle domain.ModelHandle
	var err error

	if version != "" {
		if err := domain.ValidateModelVersion(version); err != nil {
			return nil, "", err
		}
		key := domain.NewModelKey(modelID, version)
		handle, err = s.registry.Acquire(key)
	} else {
		handle, err = s.registry.AcquireDefault(modelID)
	}

	if err != nil {
		return nil, "", err
	}
	defer handle.Release()

	result, err := handle.Execute(ctx, input)
	if err != nil {
		return nil, handle.Key(), err
	}

	return result, handle.Key(), nil
}

func (s *defaultInferenceService) SubmitJob(ctx context.Context, modelID domain.ModelID, version domain.ModelVersion, input domain.RawInput, maxAttempts int, idempotencyKey string) (*domain.Job, bool, error) {
	if s.jobStore == nil || s.queue == nil {
		return nil, false, errors.New("asynchronous job processing is not enabled (missing queue or job store)")
	}

	if err := domain.ValidateModelID(modelID); err != nil {
		return nil, false, err
	}

	// 1. Check idempotency if key provided
	if idempotencyKey != "" {
		idempJobID := normalizeIdempotencyKey(idempotencyKey)
		existing, err := s.jobStore.Get(ctx, idempJobID)
		if err == nil && existing != nil {
			// Job already submitted with this idempotency key
			return existing, true, nil
		}
	}

	// 2. Determine target job ID
	var jobID domain.JobID
	if idempotencyKey != "" {
		jobID = normalizeIdempotencyKey(idempotencyKey)
	} else {
		jobID = domain.JobID(generateSecureID("job-"))
	}

	job := domain.NewJob(jobID, modelID, input, maxAttempts)
	if version != "" {
		if err := domain.ValidateModelVersion(version); err != nil {
			return nil, false, err
		}
		job.ModelKey = domain.NewModelKey(modelID, version)
	}

	// 3. Persist in JobStore
	if err := s.jobStore.Create(ctx, job); err != nil {
		return nil, false, fmt.Errorf("failed to persist job: %w", err)
	}

	// 4. Push to Queue
	if err := s.queue.Enqueue(ctx, job.ID); err != nil {
		return nil, false, fmt.Errorf("failed to enqueue job: %w", err)
	}

	return job, false, nil
}

func (s *defaultInferenceService) GetJob(ctx context.Context, jobID domain.JobID) (*domain.Job, error) {
	if s.jobStore == nil {
		return nil, errors.New("job store is not configured")
	}
	if jobID == "" {
		return nil, domain.NewValidationError("job_id", "job id cannot be empty", domain.ErrJobNotFound)
	}
	return s.jobStore.Get(ctx, jobID)
}

func (s *defaultInferenceService) ListModels(ctx context.Context) []domain.ModelDescriptor {
	return s.registry.List()
}

func (s *defaultInferenceService) CheckReadiness(ctx context.Context) (bool, map[string]string) {
	deps := make(map[string]string)
	allReady := true

	// Check registry
	models := s.registry.List()
	deps["registry"] = fmt.Sprintf("ok (%d models registered)", len(models))

	// Check queue if configured
	if s.queue != nil {
		d, err := s.queue.Depth(ctx)
		if err != nil {
			deps["queue"] = fmt.Sprintf("error: %v", err)
			allReady = false
		} else {
			deps["queue"] = fmt.Sprintf("ok (depth %d)", d)
		}
	}

	// Check custom checkers (e.g. worker pool, redis)
	for name, checker := range s.checkers {
		if err := checker.IsReady(ctx); err != nil {
			deps[name] = fmt.Sprintf("error: %v", err)
			allReady = false
		} else {
			deps[name] = "ok"
		}
	}

	return allReady, deps
}

func generateSecureID(prefix string) string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return fmt.Sprintf("%s%d-%x", prefix, time.Now().UnixNano()%100000, b)
}

func normalizeIdempotencyKey(key string) domain.JobID {
	if strings.HasPrefix(key, "idemp-") {
		return domain.JobID(key)
	}
	return domain.JobID("idemp-" + key)
}
