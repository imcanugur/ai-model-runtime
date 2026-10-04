package onnx_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/runtime"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/onnx"
)

func createTempArtifact(t *testing.T, filename string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, filename)
	if err := os.WriteFile(path, []byte("fake-onnx-bytes"), 0644); err != nil {
		t.Fatalf("failed to create temp artifact: %v", err)
	}
	return path
}

func TestONNXRuntime_ValidConstruction(t *testing.T) {
	artifactPath := createTempArtifact(t, "model.onnx")

	rt, err := onnx.NewONNXRuntime(onnx.Config{
		Entrypoint: artifactPath,
		Profile:    "standard_cpu",
	})
	if err != nil {
		t.Fatalf("expected NewONNXRuntime to succeed, got: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	inputTensor, err := domain.NewTensorInt64([]int64{1, 3}, []int64{101, 102, 103})
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	outputs, err := rt.Infer(context.Background(), domain.TensorMap{"input_ids": inputTensor})
	if err != nil {
		t.Fatalf("infer failed: %v", err)
	}

	logits, exists := outputs["logits"]
	if !exists {
		t.Fatal("expected 'logits' tensor in outputs")
	}
	if len(logits.Shape) != 2 || logits.Shape[0] != 1 || logits.Shape[1] != 2 {
		t.Errorf("expected shape [1, 2], got %v", logits.Shape)
	}
}

func TestONNXRuntime_MissingOrInvalidArtifact(t *testing.T) {
	// 1. Empty entrypoint
	_, err := onnx.NewONNXRuntime(onnx.Config{Entrypoint: ""})
	if err == nil {
		t.Fatal("expected error for empty entrypoint, got nil")
	}
	if !errors.Is(err, domain.ErrInvalidArtifactPath) {
		t.Errorf("expected ErrInvalidArtifactPath, got: %v", err)
	}

	// 2. Non-existent file
	_, err = onnx.NewONNXRuntime(onnx.Config{Entrypoint: "/non/existent/model.onnx"})
	if err == nil {
		t.Fatal("expected error for non-existent file, got nil")
	}
	if !errors.Is(err, domain.ErrInvalidArtifactPath) {
		t.Errorf("expected ErrInvalidArtifactPath, got: %v", err)
	}

	// 3. Directory instead of file
	tempDir := t.TempDir()
	_, err = onnx.NewONNXRuntime(onnx.Config{Entrypoint: tempDir})
	if err == nil {
		t.Fatal("expected error for directory entrypoint, got nil")
	}
	if !errors.Is(err, domain.ErrInvalidArtifactPath) {
		t.Errorf("expected ErrInvalidArtifactPath, got: %v", err)
	}
}

func TestONNXRuntime_ClosePreventsInfer(t *testing.T) {
	artifactPath := createTempArtifact(t, "model.onnx")

	rt, err := onnx.NewONNXRuntime(onnx.Config{Entrypoint: artifactPath})
	if err != nil {
		t.Fatalf("failed to construct runtime: %v", err)
	}

	ctx := context.Background()
	if err := rt.Close(ctx); err != nil {
		t.Fatalf("close failed: %v", err)
	}

	inputTensor, _ := domain.NewTensorInt64([]int64{1, 1}, []int64{101})
	_, err = rt.Infer(ctx, domain.TensorMap{"input_ids": inputTensor})
	if err == nil {
		t.Fatal("expected error on infer after close, got nil")
	}
	if !errors.Is(err, domain.ErrRuntimeClosed) {
		t.Errorf("expected ErrRuntimeClosed, got: %v", err)
	}

	// Calling Close again should be a no-op
	if err := rt.Close(ctx); err != nil {
		t.Errorf("idempotent close failed: %v", err)
	}
}

func TestONNXRuntime_ContextCancellation(t *testing.T) {
	artifactPath := createTempArtifact(t, "model.onnx")

	rt, err := onnx.NewONNXRuntime(onnx.Config{Entrypoint: artifactPath})
	if err != nil {
		t.Fatalf("failed to construct runtime: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel context immediately

	inputTensor, _ := domain.NewTensorInt64([]int64{1, 1}, []int64{101})
	_, err = rt.Infer(ctx, domain.TensorMap{"input_ids": inputTensor})
	if err == nil {
		t.Fatal("expected error on cancelled context, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got: %v", err)
	}
}

func TestONNXRuntime_InvalidInputTensor(t *testing.T) {
	artifactPath := createTempArtifact(t, "model.onnx")

	rt, err := onnx.NewONNXRuntime(onnx.Config{Entrypoint: artifactPath})
	if err != nil {
		t.Fatalf("failed to construct runtime: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	// Invalid tensor with mismatched shape and data length
	invalidTensor := domain.Tensor{
		Type:    domain.DataTypeInt64,
		Shape:   []int64{2, 2},
		DataI64: []int64{1, 2}, // expected 4 elements, provided 2
	}

	_, err = rt.Infer(context.Background(), domain.TensorMap{"input_ids": invalidTensor})
	if err == nil {
		t.Fatal("expected error for invalid input tensor, got nil")
	}
	if !errors.Is(err, domain.ErrInvalidTensor) {
		t.Errorf("expected ErrInvalidTensor, got: %v", err)
	}
}

type fakeCustomBackend struct {
	runErr   error
	closeErr error
	closed   bool
	mu       sync.Mutex
}

func (b *fakeCustomBackend) Run(ctx context.Context, inputs domain.TensorMap) (domain.TensorMap, error) {
	if b.runErr != nil {
		return nil, b.runErr
	}
	return domain.TensorMap{
		"logits": inputs["input_ids"],
	}, nil
}

func (b *fakeCustomBackend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.closed = true
	return b.closeErr
}

func TestONNXRuntime_CustomBackendIntegration(t *testing.T) {
	artifactPath := createTempArtifact(t, "model.onnx")
	customBackend := &fakeCustomBackend{
		runErr: errors.New("underlying onnx engine failed"),
	}

	rt, err := onnx.NewONNXRuntime(onnx.Config{
		Entrypoint: artifactPath,
		Backend:    customBackend,
	})
	if err != nil {
		t.Fatalf("failed to construct runtime: %v", err)
	}

	inputTensor, _ := domain.NewTensorInt64([]int64{1, 1}, []int64{101})
	_, err = rt.Infer(context.Background(), domain.TensorMap{"input_ids": inputTensor})
	if err == nil {
		t.Fatal("expected custom backend error, got nil")
	}

	if err := rt.Close(context.Background()); err != nil {
		t.Fatalf("close failed: %v", err)
	}
	if !customBackend.closed {
		t.Error("expected backend to be closed")
	}
}

func TestONNXRuntime_ConcurrentInference(t *testing.T) {
	artifactPath := createTempArtifact(t, "model.onnx")

	rt, err := onnx.NewONNXRuntime(onnx.Config{Entrypoint: artifactPath})
	if err != nil {
		t.Fatalf("failed to construct runtime: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	inputTensor, _ := domain.NewTensorInt64([]int64{2, 4}, []int64{1, 2, 3, 4, 5, 6, 7, 8})
	inputs := domain.TensorMap{"input_ids": inputTensor}

	var wg sync.WaitGroup
	workers := 16
	callsPerWorker := 30

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < callsPerWorker; j++ {
				out, err := rt.Infer(context.Background(), inputs)
				if err != nil {
					t.Errorf("concurrent infer failed: %v", err)
					return
				}
				if _, ok := out["logits"]; !ok {
					t.Errorf("missing logits output")
					return
				}
			}
		}()
	}

	wg.Wait()
}

func TestONNXProvider_CreateRuntime(t *testing.T) {
	artifactPath := createTempArtifact(t, "model.onnx")
	prov := onnx.NewProvider(nil)

	spec := runtime.Spec{
		Provider:   "onnx",
		Profile:    "standard_cpu",
		Entrypoint: artifactPath,
	}

	rt, err := prov.CreateRuntime(context.Background(), spec)
	if err != nil {
		t.Fatalf("failed to create runtime via provider: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	inputTensor, _ := domain.NewTensorInt64([]int64{1, 2}, []int64{1, 2})
	outputs, err := rt.Infer(context.Background(), domain.TensorMap{"input_ids": inputTensor})
	if err != nil {
		t.Fatalf("infer failed: %v", err)
	}
	if len(outputs) == 0 {
		t.Errorf("expected non-empty output")
	}
}

func TestONNXRuntime_SlowInferenceCancellation(t *testing.T) {
	artifactPath := createTempArtifact(t, "model.onnx")

	slowBackend := &blockingBackend{
		sleepDuration: 200 * time.Millisecond,
	}

	rt, err := onnx.NewONNXRuntime(onnx.Config{
		Entrypoint: artifactPath,
		Backend:    slowBackend,
	})
	if err != nil {
		t.Fatalf("failed to construct runtime: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	inputTensor, _ := domain.NewTensorInt64([]int64{1, 1}, []int64{101})
	_, err = rt.Infer(ctx, domain.TensorMap{"input_ids": inputTensor})
	if err == nil {
		t.Fatal("expected timeout/cancellation error, got nil")
	}
}

type blockingBackend struct {
	sleepDuration time.Duration
}

func (b *blockingBackend) Run(ctx context.Context, inputs domain.TensorMap) (domain.TensorMap, error) {
	select {
	case <-time.After(b.sleepDuration):
		return inputs, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (b *blockingBackend) Close() error {
	return nil
}
