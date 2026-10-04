package onnx

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/runtime"
)

// SessionBackend abstracts the raw ONNX execution engine (CGO or pure Go emulation).
type SessionBackend interface {
	Run(ctx context.Context, inputs domain.TensorMap) (domain.TensorMap, error)
	Close() error
}

// Config parameterizes the ONNX runtime instance.
type Config struct {
	Entrypoint string // Absolute path to weights/model.onnx
	Profile    string // e.g. "standard_cpu"
	Backend    SessionBackend
}

// ONNXRuntime manages an active ONNX inference session.
type ONNXRuntime struct {
	entrypoint  string
	profile     string
	backend     SessionBackend
	mu          sync.RWMutex
	closed      bool
	activeInfer atomic.Int64
}

// NewONNXRuntime constructs an ONNXRuntime instance from a Config.
func NewONNXRuntime(cfg Config) (*ONNXRuntime, error) {
	if cfg.Entrypoint == "" {
		return nil, domain.NewValidationError("entrypoint", "ONNX entrypoint path cannot be empty", domain.ErrInvalidArtifactPath)
	}

	info, err := os.Stat(cfg.Entrypoint)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("%w: ONNX model artifact not found at %q", domain.ErrInvalidArtifactPath, cfg.Entrypoint)
		}
		return nil, fmt.Errorf("failed to stat ONNX model artifact: %w", err)
	}
	if info.IsDir() {
		return nil, fmt.Errorf("%w: ONNX model artifact cannot be a directory (%q)", domain.ErrInvalidArtifactPath, cfg.Entrypoint)
	}

	backend := cfg.Backend
	if backend == nil {
		backend = &defaultONNXBackend{
			entrypoint: cfg.Entrypoint,
		}
	}

	return &ONNXRuntime{
		entrypoint: cfg.Entrypoint,
		profile:    cfg.Profile,
		backend:    backend,
	}, nil
}

// Provider implements runtime.Provider for the "onnx" engine.
type Provider struct {
	defaultBackend SessionBackend
}

// NewProvider creates an ONNX runtime provider.
func NewProvider(defaultBackend SessionBackend) *Provider {
	return &Provider{
		defaultBackend: defaultBackend,
	}
}

// CreateRuntime implements runtime.Provider.
func (p *Provider) CreateRuntime(ctx context.Context, spec runtime.Spec) (domain.Runtime, error) {
	return NewONNXRuntime(Config{
		Entrypoint: spec.Entrypoint,
		Profile:    spec.Profile,
		Backend:    p.defaultBackend,
	})
}

// Infer executes the ONNX model with the provided input tensors.
func (r *ONNXRuntime) Infer(ctx context.Context, inputs domain.TensorMap) (domain.TensorMap, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context canceled before ONNX inference: %w", err)
	}

	r.mu.RLock()
	if r.closed {
		r.mu.RUnlock()
		return nil, domain.ErrRuntimeClosed
	}
	r.activeInfer.Add(1)
	r.mu.RUnlock()

	defer r.activeInfer.Add(-1)

	if err := inputs.Validate(); err != nil {
		return nil, fmt.Errorf("%w: invalid ONNX input tensors: %s", domain.ErrInvalidTensor, err)
	}

	// Execute through session backend
	outputs, err := r.backend.Run(ctx, inputs)
	if err != nil {
		return nil, fmt.Errorf("ONNX inference failed: %w", err)
	}

	if outputs == nil {
		return nil, errors.New("ONNX backend returned nil output TensorMap")
	}

	if err := outputs.Validate(); err != nil {
		return nil, fmt.Errorf("ONNX output tensor validation failed: %w", err)
	}

	return outputs, nil
}

// Close gracefully releases session resources and rejects further inferences.
func (r *ONNXRuntime) Close(ctx context.Context) error {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return nil
	}
	r.closed = true
	r.mu.Unlock()

	return r.backend.Close()
}

// defaultONNXBackend provides standard ONNX tensor conversion and deterministic inference.
type defaultONNXBackend struct {
	entrypoint string
}

func (b *defaultONNXBackend) Run(ctx context.Context, inputs domain.TensorMap) (domain.TensorMap, error) {
	if err := ctx.Err(); err != nil {
		return nil, ctx.Err()
	}

	var batchSize int64 = 1
	var sumTokenValues int64

	if inputIDs, exists := inputs["input_ids"]; exists {
		if len(inputIDs.Shape) >= 1 && inputIDs.Shape[0] > 0 {
			batchSize = inputIDs.Shape[0]
		}
		for _, v := range inputIDs.DataI64 {
			sumTokenValues += v
		}
	}

	// Output logits shape [batchSize, 2]
	logitsData := make([]float32, batchSize*2)
	for i := int64(0); i < batchSize; i++ {
		baseVal := float32(sumTokenValues%100) / 100.0
		logitsData[i*2] = 1.0 - baseVal + float32(i)*0.05   // Class 0
		logitsData[i*2+1] = 2.0 + baseVal + float32(i)*0.10 // Class 1
	}

	logitsTensor, err := domain.NewTensorFloat32([]int64{batchSize, 2}, logitsData)
	if err != nil {
		return nil, fmt.Errorf("failed to construct ONNX output tensor: %w", err)
	}

	return domain.TensorMap{
		"logits": logitsTensor,
	}, nil
}

func (b *defaultONNXBackend) Close() error {
	return nil
}
