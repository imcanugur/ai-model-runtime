package sidecar_test

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/runtime"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/sidecar"
)

func createTempSafetensors(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	if err := os.WriteFile(path, []byte("safetensors-dummy-header-and-weights"), 0644); err != nil {
		t.Fatalf("failed to create dummy safetensors: %v", err)
	}
	return path
}

func TestSidecarRuntime_RoundTripSuccess(t *testing.T) {
	server, err := sidecar.NewMockServer()
	if err != nil {
		t.Fatalf("failed to start mock sidecar server: %v", err)
	}
	defer func() { _ = server.Close() }()

	artifactPath := createTempSafetensors(t)

	rt, err := sidecar.NewSidecarRuntime(sidecar.Config{
		Endpoint:   server.Endpoint(),
		Entrypoint: artifactPath,
		Profile:    "python_torch",
	})
	if err != nil {
		t.Fatalf("failed to construct sidecar runtime: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	inputTensor, err := domain.NewTensorInt64([]int64{2, 3}, []int64{101, 102, 103, 104, 105, 106})
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	outputs, err := rt.Infer(context.Background(), domain.TensorMap{"input_ids": inputTensor})
	if err != nil {
		t.Fatalf("sidecar infer failed: %v", err)
	}

	logits, exists := outputs["logits"]
	if !exists {
		t.Fatal("expected 'logits' in outputs")
	}
	if len(logits.Shape) != 2 || logits.Shape[0] != 2 || logits.Shape[1] != 2 {
		t.Errorf("expected logits shape [2, 2], got %v", logits.Shape)
	}

	if server.RequestCount() != 1 {
		t.Errorf("expected 1 request on server, got %d", server.RequestCount())
	}
}

func TestSidecarRuntime_TensorSerializationAllTypes(t *testing.T) {
	server, err := sidecar.NewMockServer()
	if err != nil {
		t.Fatalf("failed to start mock server: %v", err)
	}
	defer func() { _ = server.Close() }()

	// Echo back whatever tensors are sent
	server.SetHandler(func(req *sidecar.InferenceRequest) (*sidecar.InferenceResponse, error) {
		return &sidecar.InferenceResponse{
			RequestID: req.RequestID,
			Outputs:   req.Inputs,
		}, nil
	})

	artifactPath := createTempSafetensors(t)
	rt, err := sidecar.NewSidecarRuntime(sidecar.Config{
		Endpoint:   server.Endpoint(),
		Entrypoint: artifactPath,
	})
	if err != nil {
		t.Fatalf("failed to construct runtime: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	f32Tensor, _ := domain.NewTensorFloat32([]int64{1, 2}, []float32{1.5, 2.5})
	i64Tensor, _ := domain.NewTensorInt64([]int64{1, 2}, []int64{10, 20})
	i32Tensor, _ := domain.NewTensorInt32([]int64{1, 2}, []int32{30, 40})

	inputs := domain.TensorMap{
		"f32": f32Tensor,
		"i64": i64Tensor,
		"i32": i32Tensor,
	}

	outputs, err := rt.Infer(context.Background(), inputs)
	if err != nil {
		t.Fatalf("infer failed: %v", err)
	}

	if len(outputs) != 3 {
		t.Fatalf("expected 3 outputs, got %d", len(outputs))
	}
	if outputs["f32"].DataF32[0] != 1.5 || outputs["f32"].DataF32[1] != 2.5 {
		t.Errorf("mismatched float32 data")
	}
	if outputs["i64"].DataI64[0] != 10 || outputs["i64"].DataI64[1] != 20 {
		t.Errorf("mismatched int64 data")
	}
	if outputs["i32"].DataI32[0] != 30 || outputs["i32"].DataI32[1] != 40 {
		t.Errorf("mismatched int32 data")
	}
}

func TestSidecarRuntime_TimeoutAndCancellation(t *testing.T) {
	server, err := sidecar.NewMockServer()
	if err != nil {
		t.Fatalf("failed to start server: %v", err)
	}
	defer func() { _ = server.Close() }()

	// Sleep 500ms in handler
	server.SetHandler(func(req *sidecar.InferenceRequest) (*sidecar.InferenceResponse, error) {
		time.Sleep(300 * time.Millisecond)
		return &sidecar.InferenceResponse{RequestID: req.RequestID, Outputs: req.Inputs}, nil
	})

	artifactPath := createTempSafetensors(t)
	rt, err := sidecar.NewSidecarRuntime(sidecar.Config{
		Endpoint:   server.Endpoint(),
		Entrypoint: artifactPath,
		Timeout:    50 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("failed to construct runtime: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	inputTensor, _ := domain.NewTensorInt64([]int64{1, 1}, []int64{101})

	// Test context timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	_, err = rt.Infer(ctx, domain.TensorMap{"input_ids": inputTensor})
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}

	// Test context cancel
	ctx2, cancel2 := context.WithCancel(context.Background())
	cancel2() // cancel immediately
	_, err = rt.Infer(ctx2, domain.TensorMap{"input_ids": inputTensor})
	if err == nil {
		t.Fatal("expected cancelled error, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestSidecarRuntime_UnavailableServer(t *testing.T) {
	artifactPath := createTempSafetensors(t)

	// Port 59999 where nothing is listening
	rt, err := sidecar.NewSidecarRuntime(sidecar.Config{
		Endpoint:   "tcp://127.0.0.1:59999",
		Entrypoint: artifactPath,
	})
	if err != nil {
		t.Fatalf("failed to construct runtime: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	inputTensor, _ := domain.NewTensorInt64([]int64{1, 1}, []int64{101})
	_, err = rt.Infer(context.Background(), domain.TensorMap{"input_ids": inputTensor})
	if err == nil {
		t.Fatal("expected connection refused error, got nil")
	}
	if !errors.Is(err, sidecar.ErrSidecarUnavailable) {
		t.Errorf("expected ErrSidecarUnavailable, got: %v", err)
	}
}

func TestSidecarRuntime_MalformedServerResponse(t *testing.T) {
	server, err := sidecar.NewMockServer()
	if err != nil {
		t.Fatalf("failed to start server: %v", err)
	}
	defer func() { _ = server.Close() }()

	server.SetRawWriter(func(w io.Writer) error {
		return sidecar.WriteCorruptedFrame(w)
	})

	artifactPath := createTempSafetensors(t)
	rt, err := sidecar.NewSidecarRuntime(sidecar.Config{
		Endpoint:   server.Endpoint(),
		Entrypoint: artifactPath,
	})
	if err != nil {
		t.Fatalf("failed to construct runtime: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	inputTensor, _ := domain.NewTensorInt64([]int64{1, 1}, []int64{101})
	_, err = rt.Infer(context.Background(), domain.TensorMap{"input_ids": inputTensor})
	if err == nil {
		t.Fatal("expected error for malformed response, got nil")
	}
}

func TestSidecarRuntime_ServerExecutionError(t *testing.T) {
	server, err := sidecar.NewMockServer()
	if err != nil {
		t.Fatalf("failed to start server: %v", err)
	}
	defer func() { _ = server.Close() }()

	server.SetHandler(func(req *sidecar.InferenceRequest) (*sidecar.InferenceResponse, error) {
		return &sidecar.InferenceResponse{
			RequestID: req.RequestID,
			Error:     "CUDA out of memory in PyTorch worker",
		}, nil
	})

	artifactPath := createTempSafetensors(t)
	rt, err := sidecar.NewSidecarRuntime(sidecar.Config{
		Endpoint:   server.Endpoint(),
		Entrypoint: artifactPath,
	})
	if err != nil {
		t.Fatalf("failed to construct runtime: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	inputTensor, _ := domain.NewTensorInt64([]int64{1, 1}, []int64{101})
	_, err = rt.Infer(context.Background(), domain.TensorMap{"input_ids": inputTensor})
	if err == nil {
		t.Fatal("expected server execution error, got nil")
	}
}

func TestSidecarRuntime_ClosePreventsInfer(t *testing.T) {
	server, err := sidecar.NewMockServer()
	if err != nil {
		t.Fatalf("failed to start server: %v", err)
	}
	defer func() { _ = server.Close() }()

	artifactPath := createTempSafetensors(t)
	rt, err := sidecar.NewSidecarRuntime(sidecar.Config{
		Endpoint:   server.Endpoint(),
		Entrypoint: artifactPath,
	})
	if err != nil {
		t.Fatalf("failed to construct runtime: %v", err)
	}

	if err := rt.Close(context.Background()); err != nil {
		t.Fatalf("close failed: %v", err)
	}

	inputTensor, _ := domain.NewTensorInt64([]int64{1, 1}, []int64{101})
	_, err = rt.Infer(context.Background(), domain.TensorMap{"input_ids": inputTensor})
	if err == nil {
		t.Fatal("expected error on infer after close, got nil")
	}
	if !errors.Is(err, domain.ErrRuntimeClosed) {
		t.Errorf("expected ErrRuntimeClosed, got: %v", err)
	}
}

func TestSidecarRuntime_ConcurrentInference(t *testing.T) {
	server, err := sidecar.NewMockServer()
	if err != nil {
		t.Fatalf("failed to start server: %v", err)
	}
	defer func() { _ = server.Close() }()

	artifactPath := createTempSafetensors(t)
	rt, err := sidecar.NewSidecarRuntime(sidecar.Config{
		Endpoint:   server.Endpoint(),
		Entrypoint: artifactPath,
	})
	if err != nil {
		t.Fatalf("failed to construct runtime: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	inputTensor, _ := domain.NewTensorInt64([]int64{2, 3}, []int64{1, 2, 3, 4, 5, 6})
	inputs := domain.TensorMap{"input_ids": inputTensor}

	var wg sync.WaitGroup
	workers := 12
	callsPerWorker := 20

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

	expectedTotal := int64(workers * callsPerWorker)
	if server.RequestCount() != expectedTotal {
		t.Errorf("expected %d total requests, got %d", expectedTotal, server.RequestCount())
	}
}

func TestSidecarProvider_CreateRuntime(t *testing.T) {
	server, err := sidecar.NewMockServer()
	if err != nil {
		t.Fatalf("failed to start server: %v", err)
	}
	defer func() { _ = server.Close() }()

	artifactPath := createTempSafetensors(t)
	prov := sidecar.NewProvider(server.Endpoint(), nil)

	spec := runtime.Spec{
		Provider:   "sidecar",
		Profile:    "python_torch",
		Entrypoint: artifactPath,
	}

	rt, err := prov.CreateRuntime(context.Background(), spec)
	if err != nil {
		t.Fatalf("provider failed to create runtime: %v", err)
	}
	defer func() { _ = rt.Close(context.Background()) }()

	inputTensor, _ := domain.NewTensorInt64([]int64{1, 2}, []int64{101, 102})
	outputs, err := rt.Infer(context.Background(), domain.TensorMap{"input_ids": inputTensor})
	if err != nil {
		t.Fatalf("infer failed: %v", err)
	}
	if _, ok := outputs["logits"]; !ok {
		t.Errorf("missing logits in output")
	}
}
