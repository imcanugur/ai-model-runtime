package redis_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	storeRedis "github.com/imcanugur/ai-model-runtime/internal/jobstore/redis"
	"github.com/redis/go-redis/v9"
)

func getTestRedisStoreClient(t *testing.T) (*redis.Client, string) {
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

	testPrefix := fmt.Sprintf("ai-model-runtime:test:store:%d:", time.Now().UnixNano())
	return client, testPrefix
}

func cleanupStoreKeys(client *redis.Client, prefix string) {
	ctx := context.Background()
	iter := client.Scan(ctx, 0, prefix+"*", 0).Iterator()
	for iter.Next(ctx) {
		_ = client.Del(ctx, iter.Val()).Err()
	}
	_ = client.Close()
}

func TestRedisJobStore_CreateAndGet(t *testing.T) {
	client, prefix := getTestRedisStoreClient(t)
	defer cleanupStoreKeys(client, prefix)

	store, err := storeRedis.NewRedisJobStore(storeRedis.RedisJobStoreConfig{
		Client:    client,
		KeyPrefix: prefix,
	})
	if err != nil {
		t.Fatalf("failed to create redis job store: %v", err)
	}

	ctx := context.Background()
	input := domain.RawInput{Text: "redis test input"}
	job := domain.NewJob("redis-job-1", "text-detector-tr", input, 3)

	if err := store.Create(ctx, job); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	// Duplicate create should error
	if err := store.Create(ctx, job); err == nil {
		t.Fatal("expected duplicate create to fail, got nil")
	}

	fetched, err := store.Get(ctx, "redis-job-1")
	if err != nil {
		t.Fatalf("failed to get job: %v", err)
	}
	if fetched.ID != "redis-job-1" || fetched.State != domain.JobQueued {
		t.Fatalf("unexpected fetched job: %+v", fetched)
	}
}

func TestRedisJobStore_ClaimIdempotencyRace(t *testing.T) {
	client, prefix := getTestRedisStoreClient(t)
	defer cleanupStoreKeys(client, prefix)

	store, err := storeRedis.NewRedisJobStore(storeRedis.RedisJobStoreConfig{
		Client:    client,
		KeyPrefix: prefix,
	})
	if err != nil {
		t.Fatalf("failed to create redis job store: %v", err)
	}

	ctx := context.Background()
	input := domain.RawInput{Text: "race test"}
	job := domain.NewJob("redis-race-1", "text-detector-tr", input, 3)
	if err := store.Create(ctx, job); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	const workerCount = 5
	var wg sync.WaitGroup
	wg.Add(workerCount)

	successCount := 0
	claimedErrCount := 0
	var mu sync.Mutex

	for i := 0; i < workerCount; i++ {
		workerID := fmt.Sprintf("worker-%d", i+1)
		go func() {
			defer wg.Done()
			_, err := store.Claim(ctx, "redis-race-1", workerID, 10*time.Second)
			mu.Lock()
			defer mu.Unlock()
			if err == nil {
				successCount++
			} else if errors.Is(err, domain.ErrJobAlreadyClaimed) {
				claimedErrCount++
			}
		}()
	}

	wg.Wait()

	if successCount != 1 {
		t.Fatalf("expected exactly 1 worker to claim, got %d", successCount)
	}
	if claimedErrCount != workerCount-1 {
		t.Fatalf("expected %d workers to receive ErrJobAlreadyClaimed, got %d", workerCount-1, claimedErrCount)
	}
}

func TestRedisJobStore_LifecycleTransitions(t *testing.T) {
	client, prefix := getTestRedisStoreClient(t)
	defer cleanupStoreKeys(client, prefix)

	store, err := storeRedis.NewRedisJobStore(storeRedis.RedisJobStoreConfig{
		Client:    client,
		KeyPrefix: prefix,
	})
	if err != nil {
		t.Fatalf("failed to create redis job store: %v", err)
	}

	ctx := context.Background()
	input := domain.RawInput{Text: "lifecycle test"}
	job := domain.NewJob("redis-life-1", "text-detector-tr", input, 3)
	_ = store.Create(ctx, job)

	// Claim
	claimed, err := store.Claim(ctx, "redis-life-1", "worker-1", 10*time.Second)
	if err != nil {
		t.Fatalf("claim failed: %v", err)
	}
	if claimed.State != domain.JobRunning {
		t.Fatalf("expected RUNNING, got %s", claimed.State)
	}

	// Complete
	res := &domain.PredictionResult{
		ModelID:      "text-detector-tr",
		ModelVersion: "1.0.0",
		PrimaryScore: 0.95,
		Label:        "AI",
		Confidence:   0.95,
	}
	if err := store.Complete(ctx, "redis-life-1", "worker-1", res); err != nil {
		t.Fatalf("complete failed: %v", err)
	}

	completed, _ := store.Get(ctx, "redis-life-1")
	if completed.State != domain.JobCompleted || completed.Result == nil {
		t.Fatalf("unexpected completed job: %+v", completed)
	}
}
