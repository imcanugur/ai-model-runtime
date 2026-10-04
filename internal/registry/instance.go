package registry

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

// ModelInstance encapsulates a fully initialized, runnable model package and its lifecycle.
type ModelInstance struct {
	Key      domain.ModelKey
	Metadata domain.ModelMetadata
	ModelDir string
	Pipeline domain.Pipeline
	Runtime  domain.Runtime
	Manifest *ModelManifest

	mu        sync.RWMutex
	state     domain.ModelState
	refCount  atomic.Int64
	semaphore chan struct{}
	lastError string
	zeroRefCh chan struct{}
	onceZero  sync.Once
}

// NewModelInstance creates an initialized ModelInstance in StateDiscovered.
func NewModelInstance(key domain.ModelKey, meta domain.ModelMetadata, dir string, manifest *ModelManifest, pipeline domain.Pipeline, rt domain.Runtime) *ModelInstance {
	maxConcurrency := 4
	if manifest != nil && manifest.Resources.MaxConcurrency > 0 {
		maxConcurrency = manifest.Resources.MaxConcurrency
	}

	return &ModelInstance{
		Key:       key,
		Metadata:  meta,
		ModelDir:  dir,
		Manifest:  manifest,
		Pipeline:  pipeline,
		Runtime:   rt,
		state:     domain.StateDiscovered,
		semaphore: make(chan struct{}, maxConcurrency),
		zeroRefCh: make(chan struct{}),
	}
}

// State returns the current lifecycle state.
func (m *ModelInstance) State() domain.ModelState {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.state
}

// SetState updates the lifecycle state.
func (m *ModelInstance) SetState(s domain.ModelState) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.state = s
}

// SetError updates the state to StateFailed and stores the error message.
func (m *ModelInstance) SetError(err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.state = domain.StateFailed
	if err != nil {
		m.lastError = err.Error()
	}
}

// LastError returns the error that caused StateFailed.
func (m *ModelInstance) LastError() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.lastError
}

// ActiveLeases returns the number of handles currently acquired.
func (m *ModelInstance) ActiveLeases() int64 {
	return m.refCount.Load()
}

// Acquire requests a reference-counted lease on the model instance.
func (m *ModelInstance) Acquire() (domain.ModelHandle, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	switch m.state {
	case domain.StateReady, domain.StateBusy:
		m.refCount.Add(1)
		return newModelHandle(m), nil
	case domain.StateUnloading, domain.StateUnloaded:
		return nil, domain.ErrModelUnloading
	default:
		return nil, domain.ErrModelNotReady
	}
}

// release decrements the reference count and signals if unloading.
func (m *ModelInstance) release() {
	newCount := m.refCount.Add(-1)
	if newCount <= 0 {
		m.mu.RLock()
		unloading := m.state == domain.StateUnloading
		m.mu.RUnlock()
		if unloading {
			m.onceZero.Do(func() {
				close(m.zeroRefCh)
			})
		}
	}
}

// MarkUnloading transitions the model into StateUnloading, preventing new acquires.
func (m *ModelInstance) MarkUnloading() error {
	m.mu.Lock()
	if m.state == domain.StateUnloaded {
		m.mu.Unlock()
		return nil
	}
	if m.state == domain.StateUnloading {
		m.mu.Unlock()
		return nil
	}
	m.state = domain.StateUnloading
	m.mu.Unlock()

	// If there are zero active leases right now, signal immediately
	if m.refCount.Load() <= 0 {
		m.onceZero.Do(func() {
			close(m.zeroRefCh)
		})
	}
	return nil
}

// WaitAndClose waits for active leases to drain or context timeout, then closes the runtime.
func (m *ModelInstance) WaitAndClose(ctx context.Context) error {
	if err := m.MarkUnloading(); err != nil {
		return err
	}

	// Wait for active leases to hit zero
	select {
	case <-m.zeroRefCh:
	case <-ctx.Done():
		return fmt.Errorf("timeout draining active leases for model %s: %w", m.Key, ctx.Err())
	}

	// Safely close the runtime
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.Runtime != nil {
		if err := m.Runtime.Close(ctx); err != nil {
			m.state = domain.StateFailed
			m.lastError = fmt.Sprintf("failed to close runtime: %s", err)
			return err
		}
	}

	m.state = domain.StateUnloaded
	return nil
}

// Execute enforces model-level concurrency limits and runs the pipeline.
func (m *ModelInstance) Execute(ctx context.Context, input domain.RawInput) (*domain.PredictionResult, error) {
	// 1. Guard concurrency with model semaphore
	select {
	case m.semaphore <- struct{}{}:
		defer func() { <-m.semaphore }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// 2. Execute pipeline
	if m.Pipeline == nil {
		return nil, domain.NewValidationError("pipeline", "pipeline is not initialized", domain.ErrModelNotReady)
	}

	return m.Pipeline.Execute(ctx, input)
}

// Descriptor formats the instance status for external health and listing endpoints.
func (m *ModelInstance) Descriptor() domain.ModelDescriptor {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return domain.ModelDescriptor{
		Key:          m.Key,
		Metadata:     m.Metadata,
		State:        m.state,
		ActiveLeases: m.refCount.Load(),
		LastError:    m.lastError,
	}
}
