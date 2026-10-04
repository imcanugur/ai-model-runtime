package worker_test

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	storeRedis "github.com/imcanugur/ai-model-runtime/internal/jobstore/redis"
	"github.com/imcanugur/ai-model-runtime/internal/pipeline"
	queueRedis "github.com/imcanugur/ai-model-runtime/internal/queue/redis"
	"github.com/imcanugur/ai-model-runtime/internal/registry"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/mock"
	"github.com/imcanugur/ai-model-runtime/internal/worker"
	"github.com/redis/go-redis/v9"
)

func getTestRedis(t *testing.T) (*redis.Client, string) {
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

	prefix := fmt.Sprintf("ai-model-runtime:test:e2e:%d:", time.Now().UnixNano())
	return client, prefix
}

func cleanupRedisPrefix(client *redis.Client, prefix string) {
	ctx := context.Background()
	iter := client.Scan(ctx, 0, prefix+"*", 0).Iterator()
	for iter.Next(ctx) {
		_ = client.Del(ctx, iter.Val()).Err()
	}
	_ = client.Close()
}

func TestRedisE2E_FullAsyncFlow(t *testing.T) {
	client, prefix := getTestRedis(t)
	defer cleanupRedisPrefix(client, prefix)

	ctx := context.Background()

	// 1. Setup Model Registry with mock runtime
	tmpDir := t.TempDir()
	_, modelKey := setupTestModelPackage(t, tmpDir)

	stageReg := pipeline.NewDefaultStageRegistry()
	mockRT := mock.NewMockRuntime()

	factory := func(manifest *registry.ModelManifest, modelDir string) (domain.Runtime, error) {
		return mockRT, nil
	}

	modelReg := registry.NewDefaultModelRegistry(stageReg, factory)
	if err := modelReg.Discover(ctx, tmpDir); err != nil {
		t.Fatalf("discovery failed: %v", err)
	}
	if err := modelReg.Load(ctx, modelKey); err != nil {
		t.Fatalf("load failed: %v", err)
	}

	// 2. Setup Redis Queue and Redis JobStore
	q, err := queueRedis.NewRedisQueue(queueRedis.RedisQueueConfig{
		Client:    client,
		KeyPrefix: prefix,
	})
	if err != nil {
		t.Fatalf("failed to create redis queue: %v", err)
	}

	store, err := storeRedis.NewRedisJobStore(storeRedis.RedisJobStoreConfig{
		Client:    client,
		KeyPrefix: prefix,
	})
	if err != nil {
		t.Fatalf("failed to create redis store: %v", err)
	}

	// 3. Setup Worker Pool
	pool, err := worker.NewWorkerPool(worker.WorkerPoolConfig{
		Count:          2,
		Queue:          q,
		Store:          store,
		Registry:       modelReg,
		DequeueTimeout: 1 * time.Second,
		Sleeper:        mockSleeper{},
	})
	if err != nil {
		t.Fatalf("failed to create worker pool: %v", err)
	}

	if err := pool.Start(ctx); err != nil {
		t.Fatalf("pool start failed: %v", err)
	}

	// 4. Create and Enqueue Job
	input := domain.RawInput{Text: "Turkish academic detector text through redis queue."}
	jobID := domain.JobID("redis-e2e-job-1")
	job := domain.NewJob(jobID, "detector", input, 3)

	if err := store.Create(ctx, job); err != nil {
		t.Fatalf("failed to create job in redis store: %v", err)
	}
	if err := q.Enqueue(ctx, jobID); err != nil {
		t.Fatalf("failed to enqueue job to redis: %v", err)
	}

	// 5. Await job completion
	timeout := time.After(5 * time.Second)
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	var completedJob *domain.Job
	for completedJob == nil {
		select {
		case <-timeout:
			t.Fatal("timed out waiting for redis job to complete")
		case <-ticker.C:
			j, err := store.Get(ctx, jobID)
			if err == nil && j.State == domain.JobCompleted {
				completedJob = j
			}
		}
	}

	// 6. Verify result
	if completedJob.Result == nil || completedJob.Result.Label != "AI" {
		t.Fatalf("unexpected completed job in redis: %+v", completedJob)
	}

	// 7. Stop pool cleanly
	stopCtx, stopCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer stopCancel()
	if err := pool.Stop(stopCtx); err != nil {
		t.Fatalf("pool stop failed: %v", err)
	}

	// 8. Clean unload
	if err := modelReg.Unload(ctx, modelKey); err != nil {
		t.Fatalf("unload failed: %v", err)
	}
}
