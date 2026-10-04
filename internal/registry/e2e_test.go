package registry_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/registry"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/mock"
)

func setupFullE2EModelPackage(t *testing.T, rootDir string) (string, domain.ModelKey) {
	t.Helper()
	modelDir := filepath.Join(rootDir, "academic-detector-v1")
	if err := os.MkdirAll(filepath.Join(modelDir, "weights"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(modelDir, "assets"), 0755); err != nil {
		t.Fatal(err)
	}

	manifestContent := `
schema_version: "v1"
id: "academic-detector"
name: "Academic AI Detector"
version: "1.0.0"

metadata:
  task: "text-classification"
  languages: ["en"]

resources:
  max_concurrency: 8
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
        remove_pattern: '\[cite[^\]]*\]'

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
        threshold: 1.0
        positive_label: "AI"
        negative_label: "Human"
`
	if err := os.WriteFile(filepath.Join(modelDir, "model.yaml"), []byte(manifestContent), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "weights", "model.bin"), []byte("model-data"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "assets", "tokenizer.json"), []byte("{}"), 0644); err != nil {
		t.Fatal(err)
	}

	return modelDir, domain.NewModelKey("academic-detector", "1.0.0")
}

func TestE2E_FullLifecycleAndExecution(t *testing.T) {
	tempRoot := t.TempDir()
	_, modelKey := setupFullE2EModelPackage(t, tempRoot)

	// 1. Initialize Registry with a MockRuntime provider factory
	mockRT := mock.NewMockRuntime()
	reg := registry.NewDefaultModelRegistry(nil, func(m *registry.ModelManifest, dir string) (domain.Runtime, error) {
		return mockRT, nil
	})

	// 2. Discover models on disk
	ctx := context.Background()
	if err := reg.Discover(ctx, tempRoot); err != nil {
		t.Fatalf("discovery failed: %v", err)
	}

	descriptors := reg.List()
	if len(descriptors) != 1 || descriptors[0].State != domain.StateDiscovered {
		t.Fatalf("expected 1 model in StateDiscovered, got %+v", descriptors)
	}

	// 3. Load model into memory
	if err := reg.Load(ctx, modelKey); err != nil {
		t.Fatalf("load failed: %v", err)
	}

	descriptorsAfterLoad := reg.List()
	if descriptorsAfterLoad[0].State != domain.StateReady {
		t.Fatalf("expected model to be in StateReady, got %s", descriptorsAfterLoad[0].State)
	}

	// 4. Acquire handle
	handle, err := reg.Acquire(modelKey)
	if err != nil {
		t.Fatalf("acquire failed: %v", err)
	}

	// 5. Execute inference through full pipeline
	input := domain.RawInput{
		Text: "  This is a comprehensive test document [cite:99] designed to test the entire Go AI serving pipeline end to end.  ",
	}

	res, err := handle.Execute(ctx, input)
	if err != nil {
		t.Fatalf("pipeline execution failed: %v", err)
	}

	// Validate deterministic end-to-end domain results
	if res.ModelID != "academic-detector" || res.ModelVersion != "1.0.0" {
		t.Errorf("unexpected model identity in result: %s:%s", res.ModelID, res.ModelVersion)
	}
	if res.Label != "AI" && res.Label != "Human" {
		t.Errorf("unexpected classification label: %s", res.Label)
	}
	if res.Confidence <= 0.0 || res.Confidence > 1.0 {
		t.Errorf("confidence out of bounds [0, 1]: %f", res.Confidence)
	}
	if len(res.Chunks) == 0 {
		t.Error("expected non-empty chunk details in result")
	}
	if mockRT.InferCount() != 1 {
		t.Errorf("expected 1 infer call to mock runtime, got %d", mockRT.InferCount())
	}

	// 6. Release handle
	handle.Release()

	// 7. Unload model
	if err := reg.Unload(ctx, modelKey); err != nil {
		t.Fatalf("unload failed: %v", err)
	}

	if !mockRT.IsClosed() {
		t.Error("expected mock runtime to be closed after unload")
	}

	descAfterUnload := reg.List()
	if descAfterUnload[0].State != domain.StateUnloaded {
		t.Errorf("expected StateUnloaded, got %s", descAfterUnload[0].State)
	}
}

func TestE2E_ConcurrentPipelineExecution(t *testing.T) {
	tempRoot := t.TempDir()
	_, modelKey := setupFullE2EModelPackage(t, tempRoot)

	mockRT := mock.NewMockRuntime()
	reg := registry.NewDefaultModelRegistry(nil, func(m *registry.ModelManifest, dir string) (domain.Runtime, error) {
		return mockRT, nil
	})

	ctx := context.Background()
	if err := reg.Discover(ctx, tempRoot); err != nil {
		t.Fatal(err)
	}
	if err := reg.Load(ctx, modelKey); err != nil {
		t.Fatal(err)
	}

	handle, err := reg.Acquire(modelKey)
	if err != nil {
		t.Fatal(err)
	}
	defer handle.Release()

	var wg sync.WaitGroup
	concurrentClients := 15
	requestsPerClient := 20

	for i := 0; i < concurrentClients; i++ {
		wg.Add(1)
		go func(clientID int) {
			defer wg.Done()
			for req := 0; req < requestsPerClient; req++ {
				text := fmt.Sprintf("Client %d Request %d: Concurrent verification text", clientID, req)
				res, execErr := handle.Execute(ctx, domain.RawInput{Text: text})
				if execErr != nil {
					t.Errorf("concurrent execution error: %v", execErr)
					return
				}
				if res.ModelID != "academic-detector" {
					t.Errorf("unexpected modelID in concurrent execution: %s", res.ModelID)
				}
			}
		}(i)
	}

	wg.Wait()

	expectedCalls := int64(concurrentClients * requestsPerClient)
	if mockRT.InferCount() != expectedCalls {
		t.Errorf("expected %d runtime infer calls, got %d", expectedCalls, mockRT.InferCount())
	}
}

func TestE2E_ContextCancellationDuringPipeline(t *testing.T) {
	tempRoot := t.TempDir()
	_, modelKey := setupFullE2EModelPackage(t, tempRoot)

	reg := registry.NewDefaultModelRegistry(nil, func(m *registry.ModelManifest, dir string) (domain.Runtime, error) {
		return mock.NewMockRuntime(), nil
	})

	ctx := context.Background()
	_ = reg.Discover(ctx, tempRoot)
	_ = reg.Load(ctx, modelKey)

	handle, err := reg.Acquire(modelKey)
	if err != nil {
		t.Fatal(err)
	}
	defer handle.Release()

	// Pre-cancelled context
	cancelCtx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = handle.Execute(cancelCtx, domain.RawInput{Text: "Cancelled test"})
	if err == nil {
		t.Fatal("expected error on cancelled context, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestE2E_RuntimeFailurePropagation(t *testing.T) {
	tempRoot := t.TempDir()
	_, modelKey := setupFullE2EModelPackage(t, tempRoot)

	simulatedErr := errors.New("simulated accelerator device error")
	faultyRT := mock.NewMockRuntime().WithInferFunc(func(ctx context.Context, inputs domain.TensorMap) (domain.TensorMap, error) {
		return nil, simulatedErr
	})

	reg := registry.NewDefaultModelRegistry(nil, func(m *registry.ModelManifest, dir string) (domain.Runtime, error) {
		return faultyRT, nil
	})

	ctx := context.Background()
	_ = reg.Discover(ctx, tempRoot)
	_ = reg.Load(ctx, modelKey)

	handle, err := reg.Acquire(modelKey)
	if err != nil {
		t.Fatal(err)
	}
	defer handle.Release()

	_, err = handle.Execute(ctx, domain.RawInput{Text: "Test failure"})
	if err == nil {
		t.Fatal("expected pipeline failure, got nil")
	}
	if !errors.Is(err, simulatedErr) {
		t.Errorf("expected error wrapping simulatedErr, got %v", err)
	}
}
