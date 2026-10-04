package mock

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/runtime"
)

// Provider implements runtime.Provider for the "mock" engine.
type Provider struct {
	customInfer func(ctx context.Context, inputs domain.TensorMap) (domain.TensorMap, error)
	customClose func(ctx context.Context) error
}

// NewProvider creates a new mock runtime provider.
func NewProvider() *Provider {
	return &Provider{}
}

// WithInferFunc configures the mock provider's inference hook.
func (p *Provider) WithInferFunc(f func(ctx context.Context, inputs domain.TensorMap) (domain.TensorMap, error)) *Provider {
	p.customInfer = f
	return p
}

// WithCloseFunc configures the mock provider's teardown hook.
func (p *Provider) WithCloseFunc(f func(ctx context.Context) error) *Provider {
	p.customClose = f
	return p
}

// CreateRuntime implements runtime.Provider.
func (p *Provider) CreateRuntime(ctx context.Context, spec runtime.Spec) (domain.Runtime, error) {
	rt := NewMockRuntime()
	if p.customInfer != nil {
		rt.WithInferFunc(p.customInfer)
	} else if spec.Config != nil && spec.Config["task"] == "embedding" {
		rt.WithInferFunc(func(ctx context.Context, inputs domain.TensorMap) (domain.TensorMap, error) {
			dim := int64(64)
			var batchSize int64 = 1
			if inputIDs, exists := inputs["input_ids"]; exists && len(inputIDs.Shape) > 0 {
				batchSize = inputIDs.Shape[0]
			}
			data := make([]float32, batchSize*dim)
			for i := range data {
				data[i] = float32(i%10) * 0.1
			}
			t, err := domain.NewTensorFloat32([]int64{batchSize, dim}, data)
			if err != nil {
				return nil, err
			}
			return domain.TensorMap{
				"embeddings": t,
			}, nil
		})
	}
	if p.customClose != nil {
		rt.WithCloseFunc(p.customClose)
	}
	return rt, nil
}

// MockRuntime is a deterministic, thread-safe test runtime implementing domain.Runtime.
// It explicitly models hardware tensor inference without performing real ML calculations.
type MockRuntime struct {
	mu          sync.RWMutex
	closed      bool
	inferCount  atomic.Int64
	customInfer func(ctx context.Context, inputs domain.TensorMap) (domain.TensorMap, error)
	customClose func(ctx context.Context) error
}

// NewMockRuntime creates an initialized MockRuntime with deterministic default behavior.
func NewMockRuntime() *MockRuntime {
	return &MockRuntime{}
}

// WithInferFunc configures a custom inference handler for testing specific runtime conditions.
func (m *MockRuntime) WithInferFunc(f func(ctx context.Context, inputs domain.TensorMap) (domain.TensorMap, error)) *MockRuntime {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.customInfer = f
	return m
}

// WithCloseFunc configures a custom teardown handler for testing close failures.
func (m *MockRuntime) WithCloseFunc(f func(ctx context.Context) error) *MockRuntime {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.customClose = f
	return m
}

// Infer performs deterministic tensor transformation or delegates to customInfer.
func (m *MockRuntime) Infer(ctx context.Context, inputs domain.TensorMap) (domain.TensorMap, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled before runtime inference: %w", err)
	}

	m.mu.RLock()
	if m.closed {
		m.mu.RUnlock()
		return nil, domain.ErrRuntimeClosed
	}
	custom := m.customInfer
	m.mu.RUnlock()

	// Strict tensor input validation
	if err := inputs.Validate(); err != nil {
		return nil, fmt.Errorf("%w: input validation failed: %s", domain.ErrInvalidTensor, err)
	}

	m.inferCount.Add(1)

	if custom != nil {
		return custom(ctx, inputs)
	}

	// Default deterministic mock inference
	// If "input_ids" is supplied with shape [B, S], produce "logits" with shape [B, 2]
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

	logitsData := make([]float32, batchSize*2)
	for b := int64(0); b < batchSize; b++ {
		// Deterministic calculation based on token sum and batch index
		baseVal := float32(sumTokenValues%100) / 100.0
		logitsData[b*2] = 1.0 - baseVal + float32(b)*0.1   // Class 0 (e.g. Human / Negative)
		logitsData[b*2+1] = 2.0 + baseVal + float32(b)*0.2 // Class 1 (e.g. AI / Positive)
	}

	logitsTensor, err := domain.NewTensorFloat32([]int64{batchSize, 2}, logitsData)
	if err != nil {
		return nil, fmt.Errorf("mock runtime failed to construct output logits tensor: %w", err)
	}

	return domain.TensorMap{
		"logits": logitsTensor,
	}, nil
}

// Close marks the runtime as closed and executes any configured teardown hook.
func (m *MockRuntime) Close(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return nil
	}
	m.closed = true

	if m.customClose != nil {
		return m.customClose(ctx)
	}
	return nil
}

// InferCount returns the total number of Infer calls processed.
func (m *MockRuntime) InferCount() int64 {
	return m.inferCount.Load()
}

// IsClosed reports whether the runtime has been closed.
func (m *MockRuntime) IsClosed() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.closed
}
