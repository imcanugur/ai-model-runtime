package sidecar

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/runtime"
)

// DefaultSidecarEndpoint is the standard fallback endpoint when none is configured.
const DefaultSidecarEndpoint = "tcp://127.0.0.1:50051"

// Config parameterizes a SidecarRuntime instance.
type Config struct {
	Endpoint   string          // Socket address (unix:// or tcp://)
	ModelKey   string          // Identifier passed to sidecar
	Profile    string          // e.g. "python_torch"
	Entrypoint string          // Model weight artifact path (e.g. weights/model.safetensors)
	Task       string          // e.g. "question-answering", "text-classification"
	Arch       string          // e.g. "RobertaForQuestionAnswering"
	Timeout    time.Duration   // Request timeout
	Transport  ClientTransport // Custom or pre-connected transport
}

// SidecarRuntime manages communication with an external Python/PyTorch/SafeTensors sidecar process.
type SidecarRuntime struct {
	modelKey       string
	profile        string
	entrypoint     string
	task           string
	arch           string
	transport      ClientTransport
	mu             sync.RWMutex
	closed         bool
	requestCounter atomic.Uint64
	activeInfer    atomic.Int64
}

// NewSidecarRuntime constructs an initialized SidecarRuntime.
func NewSidecarRuntime(cfg Config) (*SidecarRuntime, error) {
	if cfg.Entrypoint == "" {
		return nil, domain.NewValidationError("entrypoint", "sidecar entrypoint path cannot be empty", domain.ErrInvalidArtifactPath)
	}

	// Verify artifact existence if specified
	if info, err := os.Stat(cfg.Entrypoint); err == nil {
		if info.IsDir() {
			return nil, fmt.Errorf("%w: entrypoint cannot be a directory (%q)", domain.ErrInvalidArtifactPath, cfg.Entrypoint)
		}
	}

	transport := cfg.Transport
	if transport == nil {
		endpoint := cfg.Endpoint
		if endpoint == "" {
			endpoint = DefaultSidecarEndpoint
		}
		st, err := NewSocketTransport(endpoint, cfg.Timeout)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize sidecar socket transport: %w", err)
		}
		transport = st
	}

	return &SidecarRuntime{
		modelKey:   cfg.ModelKey,
		profile:    cfg.Profile,
		entrypoint: cfg.Entrypoint,
		task:       cfg.Task,
		arch:       cfg.Arch,
		transport:  transport,
	}, nil
}

// Provider implements runtime.Provider for the "sidecar" engine.
type Provider struct {
	defaultEndpoint  string
	defaultTransport ClientTransport
}

// NewProvider creates a sidecar runtime provider.
func NewProvider(defaultEndpoint string, defaultTransport ClientTransport) *Provider {
	if defaultEndpoint == "" {
		defaultEndpoint = DefaultSidecarEndpoint
	}
	return &Provider{
		defaultEndpoint:  defaultEndpoint,
		defaultTransport: defaultTransport,
	}
}

// CreateRuntime implements runtime.Provider.
func (p *Provider) CreateRuntime(ctx context.Context, spec runtime.Spec) (domain.Runtime, error) {
	endpoint := p.defaultEndpoint
	task := ""
	arch := ""
	if spec.Config != nil {
		if spec.Config["endpoint"] != "" {
			endpoint = spec.Config["endpoint"]
		}
		task = spec.Config["task"]
		arch = spec.Config["architecture"]
	}

	return NewSidecarRuntime(Config{
		Endpoint:   endpoint,
		ModelKey:   spec.ModelDir,
		Profile:    spec.Profile,
		Entrypoint: spec.Entrypoint,
		Task:       task,
		Arch:       arch,
		Transport:  p.defaultTransport,
	})
}

// Infer transmits inputs to the sidecar process and receives computed tensors.
func (r *SidecarRuntime) Infer(ctx context.Context, inputs domain.TensorMap) (domain.TensorMap, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled before sidecar inference: %w", err)
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
		return nil, fmt.Errorf("%w: invalid sidecar input tensors: %s", domain.ErrInvalidTensor, err)
	}

	params := map[string]string{
		"profile":    r.profile,
		"entrypoint": r.entrypoint,
	}
	if r.task != "" {
		params["task"] = r.task
	}
	if r.arch != "" {
		params["architecture"] = r.arch
	}

	if qaInput, ok := domain.GetQAContext(ctx); ok {
		if qaInput.Text != "" {
			params["text"] = qaInput.Text
		}
		if qaInput.Context != "" {
			params["context"] = qaInput.Context
		}
		if qaInput.Question != "" {
			params["question"] = qaInput.Question
		}
		for k, v := range qaInput.Parameters {
			params[k] = v
		}
	}

	reqID := fmt.Sprintf("req-%d-%d", time.Now().UnixNano(), r.requestCounter.Add(1))
	req := &InferenceRequest{
		RequestID:  reqID,
		ModelKey:   r.modelKey,
		Inputs:     inputs,
		Parameters: params,
	}

	resp, err := r.transport.RoundTrip(ctx, req)
	if err != nil {
		return nil, err
	}

	if resp == nil {
		return nil, errors.New("sidecar returned nil response")
	}

	if resp.Outputs == nil {
		return nil, errors.New("sidecar returned nil tensor outputs")
	}

	if err := resp.Outputs.Validate(); err != nil {
		return nil, fmt.Errorf("sidecar output tensor validation failed: %w", err)
	}

	return resp.Outputs, nil
}

// Close gracefully terminates the runtime and releases transport connections.
func (r *SidecarRuntime) Close(ctx context.Context) error {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return nil
	}
	r.closed = true
	r.mu.Unlock()

	return r.transport.Close()
}
