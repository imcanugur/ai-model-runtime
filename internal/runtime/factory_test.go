package runtime_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/runtime"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/mock"
)

func TestFactory_RegisterAndCreateMock(t *testing.T) {
	f := runtime.NewDefaultFactory()

	mockProv := mock.NewProvider()
	if err := f.RegisterProvider("mock", mockProv); err != nil {
		t.Fatalf("failed to register mock provider: %v", err)
	}

	if !f.Supports("mock") {
		t.Errorf("expected factory to support 'mock'")
	}
	if !f.Supports("MOCK") {
		t.Errorf("expected case-insensitive support check")
	}

	spec := runtime.Spec{
		Provider: "mock",
		Profile:  "standard_cpu",
	}

	rt, err := f.Create(context.Background(), spec)
	if err != nil {
		t.Fatalf("expected create to succeed, got: %v", err)
	}
	if rt == nil {
		t.Fatal("expected non-nil runtime")
	}

	defer func() { _ = rt.Close(context.Background()) }()

	// Test inference
	inputTensor, err := domain.NewTensorInt64([]int64{1, 2}, []int64{101, 102})
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}
	out, err := rt.Infer(context.Background(), domain.TensorMap{"input_ids": inputTensor})
	if err != nil {
		t.Fatalf("inference failed: %v", err)
	}
	if _, exists := out["logits"]; !exists {
		t.Errorf("expected 'logits' tensor in output")
	}
}

func TestFactory_UnsupportedProvider(t *testing.T) {
	f := runtime.NewDefaultFactory()

	spec := runtime.Spec{
		Provider: "unsupported_engine_xyz",
	}

	_, err := f.Create(context.Background(), spec)
	if err == nil {
		t.Fatal("expected error for unsupported provider, got nil")
	}
	if !errors.Is(err, domain.ErrUnsupportedRuntime) {
		t.Errorf("expected ErrUnsupportedRuntime, got: %v", err)
	}
}

func TestFactory_EmptyProvider(t *testing.T) {
	f := runtime.NewDefaultFactory()

	spec := runtime.Spec{
		Provider: "   ",
	}

	_, err := f.Create(context.Background(), spec)
	if err == nil {
		t.Fatal("expected error for empty provider, got nil")
	}
	if !errors.Is(err, domain.ErrUnsupportedRuntime) {
		t.Errorf("expected ErrUnsupportedRuntime, got: %v", err)
	}
}

func TestFactory_RegisterNilOrEmpty(t *testing.T) {
	f := runtime.NewDefaultFactory()

	if err := f.RegisterProvider("", mock.NewProvider()); err == nil {
		t.Error("expected error when registering empty provider name")
	}

	if err := f.RegisterProvider("valid_name", nil); err == nil {
		t.Error("expected error when registering nil provider")
	}
}

func TestFactory_ConcurrentAccess(t *testing.T) {
	f := runtime.NewDefaultFactory()
	_ = f.RegisterProvider("mock", mock.NewProvider())

	var wg sync.WaitGroup
	workers := 16
	iterations := 50

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				if !f.Supports("mock") {
					t.Errorf("expected mock provider to be supported")
				}
				rt, err := f.Create(context.Background(), runtime.Spec{Provider: "mock"})
				if err != nil {
					t.Errorf("concurrent create failed: %v", err)
					continue
				}
				_ = rt.Close(context.Background())
			}
		}()
	}

	wg.Wait()
}
