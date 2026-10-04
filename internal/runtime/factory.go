package runtime

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

// Spec specifies the parameters required to instantiate a domain.Runtime.
type Spec struct {
	Provider   string            `json:"provider"`   // "mock", "onnx", "sidecar"
	Profile    string            `json:"profile"`    // "standard_cpu", "python_torch", etc.
	Entrypoint string            `json:"entrypoint"` // Resolved absolute path to model artifact
	ModelDir   string            `json:"model_dir"`  // Root directory of model package
	Config     map[string]string `json:"config,omitempty"`
}

// Provider instantiates a specific domain.Runtime given a Spec.
type Provider interface {
	CreateRuntime(ctx context.Context, spec Spec) (domain.Runtime, error)
}

// ProviderFunc allows using a plain function as a Provider.
type ProviderFunc func(ctx context.Context, spec Spec) (domain.Runtime, error)

// CreateRuntime implements Provider for ProviderFunc.
func (f ProviderFunc) CreateRuntime(ctx context.Context, spec Spec) (domain.Runtime, error) {
	return f(ctx, spec)
}

// Factory defines the runtime provider registry and instantiation coordinator.
type Factory interface {
	RegisterProvider(name string, p Provider) error
	Create(ctx context.Context, spec Spec) (domain.Runtime, error)
	Supports(name string) bool
}

// DefaultFactory is the thread-safe standard implementation of Factory.
type DefaultFactory struct {
	mu        sync.RWMutex
	providers map[string]Provider
}

// NewDefaultFactory constructs a Factory with the given providers pre-registered.
func NewDefaultFactory() *DefaultFactory {
	return &DefaultFactory{
		providers: make(map[string]Provider),
	}
}

// RegisterProvider associates a provider name with a Provider implementation.
func (f *DefaultFactory) RegisterProvider(name string, p Provider) error {
	normalized := strings.ToLower(strings.TrimSpace(name))
	if normalized == "" {
		return errors.New("provider name cannot be empty")
	}
	if p == nil {
		return errors.New("provider cannot be nil")
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	f.providers[normalized] = p
	return nil
}

// Create resolves and instantiates a domain.Runtime based on the spec's Provider.
func (f *DefaultFactory) Create(ctx context.Context, spec Spec) (domain.Runtime, error) {
	normalized := strings.ToLower(strings.TrimSpace(spec.Provider))
	if normalized == "" {
		return nil, domain.NewValidationError("runtime.provider", "runtime provider must be specified", domain.ErrUnsupportedRuntime)
	}

	f.mu.RLock()
	provider, exists := f.providers[normalized]
	f.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("%w: unsupported runtime provider %q", domain.ErrUnsupportedRuntime, spec.Provider)
	}

	return provider.CreateRuntime(ctx, spec)
}

// Supports reports whether a given provider name is registered.
func (f *DefaultFactory) Supports(name string) bool {
	normalized := strings.ToLower(strings.TrimSpace(name))
	f.mu.RLock()
	defer f.mu.RUnlock()
	_, exists := f.providers[normalized]
	return exists
}
