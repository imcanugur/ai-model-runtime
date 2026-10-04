package sidecar

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"sync"
	"time"
)

// Default transport parameters
const (
	DefaultConnectTimeout = 3 * time.Second
	DefaultRequestTimeout = 10 * time.Second
)

var (
	ErrSidecarUnavailable = errors.New("sidecar inference daemon unavailable")
	ErrTransportClosed    = errors.New("sidecar transport is closed")
)

// ClientTransport defines the bidirectional RPC channel to the inference process.
type ClientTransport interface {
	RoundTrip(ctx context.Context, req *InferenceRequest) (*InferenceResponse, error)
	Close() error
}

// SocketTransport executes requests over Unix domain socket or TCP socket connections.
type SocketTransport struct {
	network        string
	address        string
	connectTimeout time.Duration
	requestTimeout time.Duration
	mu             sync.RWMutex
	closed         bool
}

// ParseEndpoint resolves the network protocol and socket address from an endpoint URI.
func ParseEndpoint(endpoint string) (network, address string, err error) {
	trimmed := strings.TrimSpace(endpoint)
	if trimmed == "" {
		return "", "", errors.New("endpoint cannot be empty")
	}

	if strings.HasPrefix(trimmed, "unix://") {
		return "unix", strings.TrimPrefix(trimmed, "unix://"), nil
	}
	if strings.HasPrefix(trimmed, "tcp://") {
		return "tcp", strings.TrimPrefix(trimmed, "tcp://"), nil
	}

	// Heuristic defaults
	if strings.HasPrefix(trimmed, "/") || strings.HasPrefix(trimmed, "./") || strings.HasPrefix(trimmed, `\\`) {
		return "unix", trimmed, nil
	}
	if strings.Contains(trimmed, ":") {
		return "tcp", trimmed, nil
	}

	return "", "", fmt.Errorf("unrecognized endpoint format %q (expected unix://... or tcp://...)", endpoint)
}

// NewSocketTransport constructs a configured SocketTransport.
func NewSocketTransport(endpoint string, reqTimeout time.Duration) (*SocketTransport, error) {
	network, address, err := ParseEndpoint(endpoint)
	if err != nil {
		return nil, err
	}

	if reqTimeout <= 0 {
		reqTimeout = DefaultRequestTimeout
	}

	return &SocketTransport{
		network:        network,
		address:        address,
		connectTimeout: DefaultConnectTimeout,
		requestTimeout: reqTimeout,
	}, nil
}

// RoundTrip dials the sidecar, transmits the encoded request frame, and reads the response frame.
func (t *SocketTransport) RoundTrip(ctx context.Context, req *InferenceRequest) (*InferenceResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	t.mu.RLock()
	if t.closed {
		t.mu.RUnlock()
		return nil, ErrTransportClosed
	}
	t.mu.RUnlock()

	dialer := &net.Dialer{
		Timeout: t.connectTimeout,
	}

	conn, err := dialer.DialContext(ctx, t.network, t.address)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrSidecarUnavailable, err)
	}
	defer func() { _ = conn.Close() }()

	// Apply context deadline or fallback timeout
	deadline, hasDeadline := ctx.Deadline()
	if !hasDeadline {
		deadline = time.Now().Add(t.requestTimeout)
	}
	_ = conn.SetDeadline(deadline)

	// Cancellation listener
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			_ = conn.Close()
		case <-done:
		}
	}()

	// Transmit request
	if err := EncodeRequest(conn, req); err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("failed to encode sidecar request: %w", err)
	}

	// Receive response
	resp, err := DecodeResponse(conn)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("failed to decode sidecar response: %w", err)
	}

	if resp.Error != "" {
		return nil, fmt.Errorf("sidecar execution error: %s", resp.Error)
	}

	return resp, nil
}

// Close closes the transport.
func (t *SocketTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.closed = true
	return nil
}

// InMemoryTransport provides a zero-network transport for fast isolated unit tests.
type InMemoryTransport struct {
	handler func(ctx context.Context, req *InferenceRequest) (*InferenceResponse, error)
	mu      sync.RWMutex
	closed  bool
}

// NewInMemoryTransport creates an InMemoryTransport with the specified handler.
func NewInMemoryTransport(handler func(ctx context.Context, req *InferenceRequest) (*InferenceResponse, error)) *InMemoryTransport {
	return &InMemoryTransport{handler: handler}
}

// RoundTrip invokes the in-memory handler directly.
func (m *InMemoryTransport) RoundTrip(ctx context.Context, req *InferenceRequest) (*InferenceResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	m.mu.RLock()
	if m.closed {
		m.mu.RUnlock()
		return nil, ErrTransportClosed
	}
	h := m.handler
	m.mu.RUnlock()

	if h == nil {
		return nil, errors.New("no in-memory handler configured")
	}

	resp, err := h(ctx, req)
	if err != nil {
		return nil, err
	}
	if resp != nil && resp.Error != "" {
		return nil, fmt.Errorf("sidecar execution error: %s", resp.Error)
	}
	return resp, nil
}

// Close closes the in-memory transport.
func (m *InMemoryTransport) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return nil
}
