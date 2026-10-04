package redis_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	queueRedis "github.com/imcanugur/ai-model-runtime/internal/queue/redis"
	"github.com/redis/go-redis/v9"
)

func getTestRedisClient(t *testing.T) (*redis.Client, string) {
	addr := os.Getenv("REDIS_ADDR")
	if addr == "" {
		addr = "localhost:6379"
	}

	client := redis.NewClient(&redis.Options{
		Addr: addr,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		t.Fatalf("REDIS INTEGRATION TEST FAILED: active Redis server required on %s (%v). Start Redis or configure REDIS_ADDR.", addr, err)
	}

	testPrefix := fmt.Sprintf("ai-model-runtime:test:queue:%d:", time.Now().UnixNano())
	return client, testPrefix
}

func cleanupKeys(t *testing.T, client *redis.Client, prefix string) {
	ctx := context.Background()
	iter := client.Scan(ctx, 0, prefix+"*", 0).Iterator()
	for iter.Next(ctx) {
		_ = client.Del(ctx, iter.Val()).Err()
	}
	_ = client.Close()
}

func TestRedisQueue_EnqueueDequeue(t *testing.T) {
	client, prefix := getTestRedisClient(t)
	defer cleanupKeys(t, client, prefix)

	q, err := queueRedis.NewRedisQueue(queueRedis.RedisQueueConfig{
		Client:    client,
		KeyPrefix: prefix,
	})
	if err != nil {
		t.Fatalf("failed to create redis queue: %v", err)
	}

	ctx := context.Background()

	// 1. Initial depth
	depth, err := q.Depth(ctx)
	if err != nil || depth != 0 {
		t.Fatalf("expected initial depth 0, got %d, err: %v", depth, err)
	}

	// 2. Enqueue two jobs
	j1 := domain.JobID("job-red-1")
	j2 := domain.JobID("job-red-2")

	if err := q.Enqueue(ctx, j1); err != nil {
		t.Fatalf("enqueue j1 failed: %v", err)
	}
	if err := q.Enqueue(ctx, j2); err != nil {
		t.Fatalf("enqueue j2 failed: %v", err)
	}

	depth, _ = q.Depth(ctx)
	if depth != 2 {
		t.Fatalf("expected depth 2, got %d", depth)
	}

	// 3. Dequeue in FIFO order
	item, err := q.Dequeue(ctx, 1*time.Second)
	if err != nil || item != j1 {
		t.Fatalf("expected %s, got %s, err: %v", j1, item, err)
	}

	item, err = q.Dequeue(ctx, 1*time.Second)
	if err != nil || item != j2 {
		t.Fatalf("expected %s, got %s, err: %v", j2, item, err)
	}

	// 4. Dequeue on empty returns ErrQueueEmpty
	_, err = q.Dequeue(ctx, 1*time.Second)
	if !errors.Is(err, domain.ErrQueueEmpty) {
		t.Fatalf("expected ErrQueueEmpty on timeout, got: %v", err)
	}
}

func TestRedisQueue_Requeue(t *testing.T) {
	client, prefix := getTestRedisClient(t)
	defer cleanupKeys(t, client, prefix)

	q, err := queueRedis.NewRedisQueue(queueRedis.RedisQueueConfig{
		Client:    client,
		KeyPrefix: prefix,
	})
	if err != nil {
		t.Fatalf("failed to create redis queue: %v", err)
	}

	ctx := context.Background()
	j1 := domain.JobID("job-rq-1")

	if err := q.Requeue(ctx, j1); err != nil {
		t.Fatalf("requeue failed: %v", err)
	}

	item, err := q.Dequeue(ctx, 1*time.Second)
	if err != nil || item != j1 {
		t.Fatalf("expected %s, got %s, err: %v", j1, item, err)
	}
}
