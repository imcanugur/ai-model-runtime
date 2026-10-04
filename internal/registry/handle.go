package registry

import (
	"context"
	"sync/atomic"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

// modelHandle provides a reference-counted lease on an active ModelInstance.
type modelHandle struct {
	instance *ModelInstance
	released atomic.Bool
}

func newModelHandle(instance *ModelInstance) *modelHandle {
	return &modelHandle{
		instance: instance,
	}
}

func (h *modelHandle) ID() domain.ModelID {
	return h.instance.Metadata.ID
}

func (h *modelHandle) Version() domain.ModelVersion {
	return h.instance.Metadata.Version
}

func (h *modelHandle) Key() domain.ModelKey {
	return h.instance.Key
}

func (h *modelHandle) Execute(ctx context.Context, input domain.RawInput) (*domain.PredictionResult, error) {
	if h.released.Load() {
		return nil, domain.ErrHandleAlreadyReleased
	}
	return h.instance.Execute(ctx, input)
}

// Release returns the lease back to the ModelInstance. It is safe for concurrent or repeated calls.
func (h *modelHandle) Release() {
	if h.released.CompareAndSwap(false, true) {
		h.instance.release()
	}
}
