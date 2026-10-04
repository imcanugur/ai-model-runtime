package redis

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/redis/go-redis/v9"
)

// RedisQueueConfig holds configuration for the Redis-backed job queue.
type RedisQueueConfig struct {
	Client    *redis.Client
	KeyPrefix string // e.g. "ai-model-runtime:jobs:"
}

// RedisQueue implements queue.JobQueue backed by Redis Lists (LPUSH / BRPOP).
type RedisQueue struct {
	client   *redis.Client
	queueKey string
}

// NewRedisQueue constructs a RedisQueue.
func NewRedisQueue(cfg RedisQueueConfig) (*RedisQueue, error) {
	if cfg.Client == nil {
		return nil, errors.New("redis client cannot be nil")
	}
	prefix := cfg.KeyPrefix
	if prefix == "" {
		prefix = "ai-model-runtime:jobs:"
	}

	return &RedisQueue{
		client:   cfg.Client,
		queueKey: prefix + "queue",
	}, nil
}

// Enqueue prepends a job ID to the Redis list.
func (q *RedisQueue) Enqueue(ctx context.Context, jobID domain.JobID) error {
	if jobID == "" {
		return errors.New("job id cannot be empty")
	}
	if err := q.client.LPush(ctx, q.queueKey, string(jobID)).Err(); err != nil {
		return fmt.Errorf("failed to enqueue job to redis: %w", err)
	}
	return nil
}

// Dequeue pops a job ID from the Redis list, blocking up to timeout.
func (q *RedisQueue) Dequeue(ctx context.Context, timeout time.Duration) (domain.JobID, error) {
	// Minimum BRPOP timeout in Redis is 1 second when using integers
	t := timeout
	if t < time.Second {
		t = time.Second
	}

	res, err := q.client.BRPop(ctx, t, q.queueKey).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return "", domain.ErrQueueEmpty
		}
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return "", ctx.Err()
		}
		return "", fmt.Errorf("failed to dequeue from redis: %w", err)
	}

	if len(res) < 2 {
		return "", domain.ErrQueueEmpty
	}

	return domain.JobID(res[1]), nil
}

// Requeue puts a job ID back into the queue for retry.
func (q *RedisQueue) Requeue(ctx context.Context, jobID domain.JobID) error {
	return q.Enqueue(ctx, jobID)
}

// Depth returns the current number of elements in the Redis queue list.
func (q *RedisQueue) Depth(ctx context.Context) (int64, error) {
	depth, err := q.client.LLen(ctx, q.queueKey).Result()
	if err != nil {
		return 0, fmt.Errorf("failed to read queue depth from redis: %w", err)
	}
	return depth, nil
}

// Close closes the Redis connection.
func (q *RedisQueue) Close() error {
	return q.client.Close()
}
