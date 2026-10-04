package mock_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/mock"
)

func TestMockRuntime_DeterministicInference(t *testing.T) {
	rt := mock.NewMockRuntime()

	inputTensor, err := domain.NewTensorInt64([]int64{2, 3}, []int64{101, 102, 103, 104, 105, 106})
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	inputs := domain.TensorMap{"input_ids": inputTensor}

	outputs, err := rt.Infer(context.Background(), inputs)
	if err != nil {
		t.Fatalf("infer failed: %v", err)
	}

	logits, exists := outputs["logits"]
	if !exists {
		t.Fatal("expected 'logits' tensor in outputs")
	}

	if len(logits.Shape) != 2 || logits.Shape[0] != 2 || logits.Shape[1] != 2 {
		t.Errorf("expected logits shape [2, 2], got %v", logits.Shape)
	}

	if rt.InferCount() != 1 {
		t.Errorf("expected inferCount 1, got %d", rt.InferCount())
	}
}

func TestMockRuntime_ClosePreventsInfer(t *testing.T) {
	rt := mock.NewMockRuntime()
	ctx := context.Background()

	if err := rt.Close(ctx); err != nil {
		t.Fatalf("close failed: %v", err)
	}

	if !rt.IsClosed() {
		t.Error("expected runtime to be closed")
	}

	inputTensor, _ := domain.NewTensorInt64([]int64{1, 1}, []int64{101})
	_, err := rt.Infer(ctx, domain.TensorMap{"input_ids": inputTensor})
	if err == nil {
		t.Fatal("expected error on infer after close, got nil")
	}
	if !errors.Is(err, domain.ErrRuntimeClosed) {
		t.Errorf("expected ErrRuntimeClosed, got %v", err)
	}
}

func TestMockRuntime_ContextCancelled(t *testing.T) {
	rt := mock.NewMockRuntime()
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	inputTensor, _ := domain.NewTensorInt64([]int64{1, 1}, []int64{101})
	_, err := rt.Infer(ctx, domain.TensorMap{"input_ids": inputTensor})
	if err == nil {
		t.Fatal("expected error on cancelled context, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestMockRuntime_CustomHooks(t *testing.T) {
	customErr := errors.New("simulated hardware error")
	rt := mock.NewMockRuntime().
		WithInferFunc(func(ctx context.Context, inputs domain.TensorMap) (domain.TensorMap, error) {
			return nil, customErr
		}).
		WithCloseFunc(func(ctx context.Context) error {
			return errors.New("custom close error")
		})

	inputTensor, _ := domain.NewTensorInt64([]int64{1, 1}, []int64{101})
	_, err := rt.Infer(context.Background(), domain.TensorMap{"input_ids": inputTensor})
	if !errors.Is(err, customErr) {
		t.Errorf("expected customErr, got %v", err)
	}

	if err := rt.Close(context.Background()); err == nil {
		t.Error("expected custom close error, got nil")
	}
}

func TestMockRuntime_ConcurrentInference(t *testing.T) {
	rt := mock.NewMockRuntime()
	inputTensor, _ := domain.NewTensorInt64([]int64{1, 2}, []int64{1, 2})
	inputs := domain.TensorMap{"input_ids": inputTensor}

	var wg sync.WaitGroup
	workers := 16
	callsPerWorker := 25

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < callsPerWorker; j++ {
				_, err := rt.Infer(context.Background(), inputs)
				if err != nil {
					t.Errorf("concurrent infer failed: %v", err)
				}
			}
		}()
	}

	wg.Wait()

	expectedCalls := int64(workers * callsPerWorker)
	if rt.InferCount() != expectedCalls {
		t.Errorf("expected %d total infer calls, got %d", expectedCalls, rt.InferCount())
	}
}
