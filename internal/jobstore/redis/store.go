package redis

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/redis/go-redis/v9"
)

// RedisJobStoreConfig holds Redis client and prefix settings for job storage.
type RedisJobStoreConfig struct {
	Client    *redis.Client
	KeyPrefix string // e.g. "ai-model-runtime:jobs:"
	TTL       time.Duration
}

// RedisJobStore implements jobstore.JobStore backed by Redis JSON hashes and sets.
type RedisJobStore struct {
	client    *redis.Client
	prefix    string
	activeKey string
	ttl       time.Duration
}

// NewRedisJobStore constructs a RedisJobStore.
func NewRedisJobStore(cfg RedisJobStoreConfig) (*RedisJobStore, error) {
	if cfg.Client == nil {
		return nil, errors.New("redis client cannot be nil")
	}
	prefix := cfg.KeyPrefix
	if prefix == "" {
		prefix = "ai-model-runtime:jobs:"
	}
	ttl := cfg.TTL
	if ttl <= 0 {
		ttl = 24 * time.Hour
	}

	return &RedisJobStore{
		client:    cfg.Client,
		prefix:    prefix,
		activeKey: prefix + "active",
		ttl:       ttl,
	}, nil
}

func (s *RedisJobStore) jobKey(id domain.JobID) string {
	return fmt.Sprintf("%sdata:%s", s.prefix, id)
}

// Create persists a new job in Redis.
func (s *RedisJobStore) Create(ctx context.Context, job *domain.Job) error {
	if job == nil {
		return errors.New("job cannot be nil")
	}

	key := s.jobKey(job.ID)
	data, err := json.Marshal(job)
	if err != nil {
		return fmt.Errorf("failed to marshal job to json: %w", err)
	}

	// SetNX to ensure we do not overwrite an existing job
	ok, err := s.client.SetNX(ctx, key, data, s.ttl).Result()
	if err != nil {
		return fmt.Errorf("failed to save job to redis: %w", err)
	}
	if !ok {
		return fmt.Errorf("job %s already exists", job.ID)
	}
	return nil
}

// Get fetches a job by ID from Redis.
func (s *RedisJobStore) Get(ctx context.Context, id domain.JobID) (*domain.Job, error) {
	key := s.jobKey(id)
	raw, err := s.client.Get(ctx, key).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return nil, domain.ErrJobNotFound
		}
		return nil, fmt.Errorf("failed to get job %s from redis: %w", id, err)
	}

	var job domain.Job
	if err := json.Unmarshal([]byte(raw), &job); err != nil {
		return nil, fmt.Errorf("corrupt job data in redis for %s: %w", id, err)
	}
	return &job, nil
}

// Claim atomically transitions a job to running if queued or stale.
func (s *RedisJobStore) Claim(ctx context.Context, id domain.JobID, workerID string, leaseDuration time.Duration) (*domain.Job, error) {
	key := s.jobKey(id)
	now := time.Now().UTC()
	leaseTime := now.Add(leaseDuration)

	// Watch key for atomic optimistic concurrency or read-update under transaction
	var updatedJob *domain.Job

	err := s.client.Watch(ctx, func(tx *redis.Tx) error {
		raw, err := tx.Get(ctx, key).Result()
		if err != nil {
			if errors.Is(err, redis.Nil) {
				return domain.ErrJobNotFound
			}
			return err
		}

		var job domain.Job
		if err := json.Unmarshal([]byte(raw), &job); err != nil {
			return fmt.Errorf("corrupt job json: %w", err)
		}

		isQueued := job.State == domain.JobQueued
		isStale := job.State == domain.JobRunning && job.LeaseUntil != nil && job.LeaseUntil.Before(now)

		if !isQueued && !isStale {
			return domain.ErrJobAlreadyClaimed
		}

		if err := domain.ValidateTransition(job.State, domain.JobRunning); err != nil {
			return err
		}

		job.State = domain.JobRunning
		job.WorkerID = workerID
		job.StartedAt = &now
		job.LeaseUntil = &leaseTime
		job.Attempt++

		bytes, err := json.Marshal(job)
		if err != nil {
			return err
		}

		pipe := tx.TxPipeline()
		pipe.Set(ctx, key, bytes, s.ttl)
		pipe.SAdd(ctx, s.activeKey, string(id))
		_, err = pipe.Exec(ctx)
		if err != nil {
			return err
		}

		updatedJob = &job
		return nil
	}, key)

	if err != nil {
		if errors.Is(err, redis.TxFailedErr) {
			return nil, domain.ErrJobAlreadyClaimed
		}
		return nil, err
	}

	return updatedJob, nil
}

// Heartbeat refreshes the lease expiration.
func (s *RedisJobStore) Heartbeat(ctx context.Context, id domain.JobID, workerID string, leaseDuration time.Duration) error {
	key := s.jobKey(id)
	now := time.Now().UTC()
	leaseTime := now.Add(leaseDuration)

	return s.client.Watch(ctx, func(tx *redis.Tx) error {
		raw, err := tx.Get(ctx, key).Result()
		if err != nil {
			return err
		}

		var job domain.Job
		if err := json.Unmarshal([]byte(raw), &job); err != nil {
			return err
		}

		if job.State != domain.JobRunning || job.WorkerID != workerID {
			return domain.ErrJobAlreadyClaimed
		}

		job.LeaseUntil = &leaseTime
		bytes, err := json.Marshal(job)
		if err != nil {
			return err
		}

		_, err = tx.TxPipelined(ctx, func(pipe redis.Pipeliner) error {
			pipe.Set(ctx, key, bytes, s.ttl)
			return nil
		})
		return err
	}, key)
}

// Complete saves the prediction result and transitions the job to completed.
func (s *RedisJobStore) Complete(ctx context.Context, id domain.JobID, workerID string, result *domain.PredictionResult) error {
	key := s.jobKey(id)
	now := time.Now().UTC()

	return s.client.Watch(ctx, func(tx *redis.Tx) error {
		raw, err := tx.Get(ctx, key).Result()
		if err != nil {
			return err
		}

		var job domain.Job
		if err := json.Unmarshal([]byte(raw), &job); err != nil {
			return err
		}

		if err := domain.ValidateTransition(job.State, domain.JobCompleted); err != nil {
			return err
		}

		if job.WorkerID != workerID {
			return domain.ErrJobAlreadyClaimed
		}

		job.State = domain.JobCompleted
		job.CompletedAt = &now
		job.Result = result
		job.LeaseUntil = nil

		bytes, err := json.Marshal(job)
		if err != nil {
			return err
		}

		_, err = tx.TxPipelined(ctx, func(pipe redis.Pipeliner) error {
			pipe.Set(ctx, key, bytes, s.ttl)
			pipe.SRem(ctx, s.activeKey, string(id))
			return nil
		})
		return err
	}, key)
}

// Fail transitions the job to failed state.
func (s *RedisJobStore) Fail(ctx context.Context, id domain.JobID, workerID string, errMsg string) error {
	key := s.jobKey(id)
	now := time.Now().UTC()

	return s.client.Watch(ctx, func(tx *redis.Tx) error {
		raw, err := tx.Get(ctx, key).Result()
		if err != nil {
			return err
		}

		var job domain.Job
		if err := json.Unmarshal([]byte(raw), &job); err != nil {
			return err
		}

		if err := domain.ValidateTransition(job.State, domain.JobFailed); err != nil {
			return err
		}

		job.State = domain.JobFailed
		job.CompletedAt = &now
		job.Error = errMsg
		job.LeaseUntil = nil

		bytes, err := json.Marshal(job)
		if err != nil {
			return err
		}

		_, err = tx.TxPipelined(ctx, func(pipe redis.Pipeliner) error {
			pipe.Set(ctx, key, bytes, s.ttl)
			pipe.SRem(ctx, s.activeKey, string(id))
			return nil
		})
		return err
	}, key)
}

// Retry transitions a job back to queued state.
func (s *RedisJobStore) Retry(ctx context.Context, id domain.JobID, workerID string, errMsg string) error {
	key := s.jobKey(id)

	return s.client.Watch(ctx, func(tx *redis.Tx) error {
		raw, err := tx.Get(ctx, key).Result()
		if err != nil {
			return err
		}

		var job domain.Job
		if err := json.Unmarshal([]byte(raw), &job); err != nil {
			return err
		}

		if err := domain.ValidateTransition(job.State, domain.JobQueued); err != nil {
			return err
		}

		job.State = domain.JobQueued
		job.WorkerID = ""
		job.Error = errMsg
		job.LeaseUntil = nil

		bytes, err := json.Marshal(job)
		if err != nil {
			return err
		}

		_, err = tx.TxPipelined(ctx, func(pipe redis.Pipeliner) error {
			pipe.Set(ctx, key, bytes, s.ttl)
			pipe.SRem(ctx, s.activeKey, string(id))
			return nil
		})
		return err
	}, key)
}

// FindStaleJobs inspects active jobs and returns those with expired leases.
func (s *RedisJobStore) FindStaleJobs(ctx context.Context, now time.Time, limit int) ([]domain.JobID, error) {
	activeIDs, err := s.client.SMembers(ctx, s.activeKey).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to query active jobs set: %w", err)
	}

	var stale []domain.JobID
	for _, idStr := range activeIDs {
		id := domain.JobID(idStr)
		job, err := s.Get(ctx, id)
		if err != nil {
			// Orphan set entry, clean it
			_ = s.client.SRem(ctx, s.activeKey, idStr)
			continue
		}

		if job.State == domain.JobRunning && job.LeaseUntil != nil && job.LeaseUntil.Before(now) {
			stale = append(stale, id)
			if limit > 0 && len(stale) >= limit {
				break
			}
		}
	}
	return stale, nil
}
