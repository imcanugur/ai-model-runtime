package worker_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	storeMemory "github.com/imcanugur/ai-model-runtime/internal/jobstore/memory"
	"github.com/imcanugur/ai-model-runtime/internal/pipeline"
	queueMemory "github.com/imcanugur/ai-model-runtime/internal/queue/memory"
	"github.com/imcanugur/ai-model-runtime/internal/registry"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/mock"
	"github.com/imcanugur/ai-model-runtime/internal/worker"
)

func setupTestModelPackage(t *testing.T, rootDir string) (string, domain.ModelKey) {
	t.Helper()
	modelDir := filepath.Join(rootDir, "detector-v1")
	if err := os.MkdirAll(filepath.Join(modelDir, "weights"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(modelDir, "assets"), 0755); err != nil {
		t.Fatal(err)
	}

	manifestContent := `
schema_version: "v1"
id: "detector"
name: "E2E Detector"
version: "1.0.0"

metadata:
  task: "text-classification"
  languages: ["en"]

resources:
  max_concurrency: 4
  timeout_ms: 5000

runtime:
  provider: "mock"
  profile: "cpu"
  entrypoint: "weights/model.bin"

pipeline:
  stages:
    - name: "normalizer"
      type: "text_normalizer"
      config:
        strip_whitespace: true
        lowercase: true

    - name: "chunker"
      type: "sliding_window_chunker"
      config:
        max_tokens: 16
        stride: 0

    - name: "infer"
      type: "runtime_infer"

    - name: "reduction"
      type: "token_weighted_logits"
      config:
        human_class_index: 0
        ai_class_index: 1

    - name: "calibrator"
      type: "threshold_calibrator"
      config:
        threshold: 0.5
        positive_label: "AI"
        negative_label: "Human"
`
	if err := os.WriteFile(filepath.Join(modelDir, "model.yaml"), []byte(manifestContent), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "weights", "model.bin"), []byte("mock binary"), 0644); err != nil {
		t.Fatal(err)
	}

	return modelDir, domain.NewModelKey("detector", "1.0.0")
}

func TestE2E_FullPlatformJobLifecycle(t *testing.T) {
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
		t.Fatalf("model discovery failed: %v", err)
	}
	if err := modelReg.Load(ctx, modelKey); err != nil {
		t.Fatalf("model load failed: %v", err)
	}

	// 2. Setup Queue & Store
	q := queueMemory.NewMemoryQueue()
	defer q.Close()
	store := storeMemory.NewMemoryJobStore()

	// 3. Setup Worker Pool
	pool, err := worker.NewWorkerPool(worker.WorkerPoolConfig{
		Count:          2,
		Queue:          q,
		Store:          store,
		Registry:       modelReg,
		DequeueTimeout: 50 * time.Millisecond,
		Sleeper:        mockSleeper{},
	})
	if err != nil {
		t.Fatalf("worker pool creation failed: %v", err)
	}

	if err := pool.Start(ctx); err != nil {
		t.Fatalf("worker pool start failed: %v", err)
	}

	// 4. Submit 5 Jobs
	const jobCount = 5
	for i := 0; i < jobCount; i++ {
		input := domain.RawInput{Text: "Artificial Intelligence generated academic text sample."}

		jobID := domain.JobID("e2e-job-" + string(rune('A'+i)))
		j := domain.NewJob(jobID, "detector", input, 3)
		if err := store.Create(ctx, j); err != nil {
			t.Fatalf("failed to create job in store: %v", err)
		}
		if err := q.Enqueue(ctx, jobID); err != nil {
			t.Fatalf("failed to enqueue job: %v", err)
		}
	}

	// 5. Poll store until all jobs are completed (or timeout)
	timeout := time.After(3 * time.Second)
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()

	completedAll := false
	for !completedAll {
		select {
		case <-timeout:
			t.Fatal("timed out waiting for all jobs to complete")
		case <-ticker.C:
			allDone := true
			for i := 0; i < jobCount; i++ {
				jobID := domain.JobID("e2e-job-" + string(rune('A'+i)))
				j, err := store.Get(ctx, jobID)
				if err != nil || j.State != domain.JobCompleted {
					allDone = false
					break
				}
			}
			if allDone {
				completedAll = true
			}
		}
	}

	// 6. Verify each job's output result
	for i := 0; i < jobCount; i++ {
		jobID := domain.JobID("e2e-job-" + string(rune('A'+i)))
		j, err := store.Get(ctx, jobID)
		if err != nil {
			t.Fatalf("error fetching job %s: %v", jobID, err)
		}
		if j.State != domain.JobCompleted {
			t.Fatalf("job %s expected completed, got %s", jobID, j.State)
		}
		if j.Result == nil {
			t.Fatalf("job %s result is nil", jobID)
		}
		if j.Result.Label != "AI" {
			t.Fatalf("job %s unexpected label: %s (expected AI)", jobID, j.Result.Label)
		}
	}

	// 7. Stop Worker Pool
	stopCtx, stopCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer stopCancel()
	if err := pool.Stop(stopCtx); err != nil {
		t.Fatalf("failed to stop worker pool: %v", err)
	}

	// 8. Verify model unloads cleanly (proving zero handle leaks)
	if err := modelReg.Unload(ctx, modelKey); err != nil {
		t.Fatalf("failed to unload model cleanly after jobs completed: %v", err)
	}
}
