package sidecar

import (
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"sync"
	"sync/atomic"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

// MockServer provides an in-process socket listener simulating a Python/PyTorch sidecar daemon.
type MockServer struct {
	listener     net.Listener
	network      string
	address      string
	handler      func(req *InferenceRequest) (*InferenceResponse, error)
	rawWriter    func(w io.Writer) error // If non-nil, overrides standard response encoding (for malformed response tests)
	mu           sync.RWMutex
	closed       bool
	requestCount atomic.Int64
	done         chan struct{}
}

// NewMockServer starts a mock sidecar server on loopback TCP.
func NewMockServer() (*MockServer, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("failed to listen on loopback: %w", err)
	}

	s := &MockServer{
		listener: ln,
		network:  "tcp",
		address:  ln.Addr().String(),
		done:     make(chan struct{}),
	}

	// Default deterministic handler
	s.handler = s.defaultHandler

	go s.serve()
	return s, nil
}

// Endpoint returns the connection URI for clients.
func (s *MockServer) Endpoint() string {
	return fmt.Sprintf("tcp://%s", s.address)
}

// SetHandler configures a custom inference request handler.
func (s *MockServer) SetHandler(h func(req *InferenceRequest) (*InferenceResponse, error)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.handler = h
}

// SetRawWriter overrides normal response writing to simulate malformed frames.
func (s *MockServer) SetRawWriter(w func(w io.Writer) error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.rawWriter = w
}

// RequestCount returns the total number of requests received.
func (s *MockServer) RequestCount() int64 {
	return s.requestCount.Load()
}

// Close gracefully stops the listener and terminates server goroutines.
func (s *MockServer) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	close(s.done)
	s.mu.Unlock()

	return s.listener.Close()
}

func (s *MockServer) serve() {
	for {
		conn, err := s.listener.Accept()
		if err != nil {
			select {
			case <-s.done:
				return
			default:
				return
			}
		}

		go s.handleConn(conn)
	}
}

func (s *MockServer) handleConn(conn net.Conn) {
	defer func() { _ = conn.Close() }()

	req, err := DecodeRequest(conn)
	if err != nil {
		return
	}

	s.requestCount.Add(1)

	s.mu.RLock()
	rawWriter := s.rawWriter
	handler := s.handler
	s.mu.RUnlock()

	if rawWriter != nil {
		_ = rawWriter(conn)
		return
	}

	var resp *InferenceResponse
	if handler != nil {
		r, err := handler(req)
		if err != nil {
			resp = &InferenceResponse{
				RequestID: req.RequestID,
				Error:     err.Error(),
			}
		} else {
			resp = r
		}
	} else {
		resp, _ = s.defaultHandler(req)
	}

	_ = EncodeResponse(conn, resp)
}

func (s *MockServer) defaultHandler(req *InferenceRequest) (*InferenceResponse, error) {
	var batchSize int64 = 1
	var sumTokens int64

	if inputIDs, exists := req.Inputs["input_ids"]; exists {
		if len(inputIDs.Shape) >= 1 && inputIDs.Shape[0] > 0 {
			batchSize = inputIDs.Shape[0]
		}
		for _, v := range inputIDs.DataI64 {
			sumTokens += v
		}
	}

	logitsData := make([]float32, batchSize*2)
	for i := int64(0); i < batchSize; i++ {
		base := float32(sumTokens%100) / 100.0
		logitsData[i*2] = 1.0 - base + float32(i)*0.05
		logitsData[i*2+1] = 2.0 + base + float32(i)*0.10
	}

	logitsTensor, err := domain.NewTensorFloat32([]int64{batchSize, 2}, logitsData)
	if err != nil {
		return &InferenceResponse{
			RequestID: req.RequestID,
			Error:     fmt.Sprintf("failed to construct tensor: %v", err),
		}, nil
	}

	return &InferenceResponse{
		RequestID: req.RequestID,
		Outputs: domain.TensorMap{
			"logits": logitsTensor,
		},
	}, nil
}

// WriteCorruptedFrame writes an invalid frame for testing decoder error handling.
func WriteCorruptedFrame(w io.Writer) error {
	// Write 4 bytes length indicating 10 bytes, but send only 3 garbage bytes then close
	_ = binary.Write(w, binary.BigEndian, uint32(10))
	_, err := w.Write([]byte{0xFF, 0x00, 0xAA})
	return err
}
