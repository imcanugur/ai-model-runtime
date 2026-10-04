package http

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/imcanugur/ai-model-runtime/internal/queue"
	"github.com/imcanugur/ai-model-runtime/internal/service"
	"github.com/imcanugur/ai-model-runtime/internal/worker"
)

// Server coordinates HTTP routing, middlewares, and unified lifecycle shutdown.
type Server struct {
	httpServer *http.Server
	service    service.InferenceService
	workerPool *worker.WorkerPool
	queue      queue.JobQueue
	logger     *slog.Logger
	handler    http.Handler
}

// ServerDependencies groups optional background execution services for lifecycle management.
type ServerDependencies struct {
	WorkerPool *worker.WorkerPool
	Queue      queue.JobQueue
}

// NewRouter constructs and registers all API routes and middlewares.
func NewRouter(svc service.InferenceService, cfg ServerConfig) http.Handler {
	mux := http.NewServeMux()

	// 1. Synchronous Inference
	mux.HandleFunc("POST /v1/models/{modelID}/predict", HandleSyncPredict(svc))
	mux.HandleFunc("POST /v1/models/{modelID}/versions/{version}/predict", HandleSyncPredict(svc))

	// 2. Asynchronous Jobs
	mux.HandleFunc("POST /v1/jobs", HandleCreateJob(svc))
	mux.HandleFunc("GET /v1/jobs/{jobID}", HandleGetJob(svc))

	// 3. Model Discovery
	mux.HandleFunc("GET /v1/models", HandleListModels(svc))
	mux.HandleFunc("GET /v1/models/{modelID}", HandleGetModel(svc))

	// 4. Health & Readiness
	mux.HandleFunc("GET /health", HandleHealth())
	mux.HandleFunc("GET /ready", HandleReady(svc))

	// 5. Backwards Compatibility
	mux.HandleFunc("POST /predict", HandleLegacyPredict(svc, cfg.DefaultModelID))

	// 6. OpenAPI & Swagger UI
	RegisterSwaggerRoutes(mux)

	// Middleware chain: Recovery -> RequestID -> Timeout -> MaxBody -> ContentType -> Routes
	var handler http.Handler = mux
	handler = ContentTypeJSONMiddleware(handler)
	handler = MaxBodyBytesMiddleware(cfg.MaxBodyBytes)(handler)
	handler = TimeoutMiddleware(cfg.RequestTimeout)(handler)
	handler = RequestIDMiddleware(handler)
	handler = RecoveryMiddleware(cfg.Logger)(handler)

	return handler
}

// NewServer creates an HTTP Server instance.
func NewServer(cfg ServerConfig, svc service.InferenceService, deps ServerDependencies) (*Server, error) {
	if svc == nil {
		return nil, errors.New("inference service cannot be nil")
	}

	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}

	router := NewRouter(svc, cfg)

	httpSrv := &http.Server{
		Addr:         cfg.Address,
		Handler:      router,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
		IdleTimeout:  cfg.IdleTimeout,
	}

	return &Server{
		httpServer: httpSrv,
		service:    svc,
		workerPool: deps.WorkerPool,
		queue:      deps.Queue,
		logger:     logger,
		handler:    router,
	}, nil
}

// Handler returns the underlying router with all middlewares applied (useful for httptest).
func (s *Server) Handler() http.Handler {
	return s.handler
}

// Start runs the HTTP listener blocking until closed.
func (s *Server) Start() error {
	s.logger.Info("starting HTTP server", "address", s.httpServer.Addr)
	if err := s.httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("HTTP server listen error: %w", err)
	}
	return nil
}

// Shutdown executes graceful shutdown according to the required lifecycle sequence:
// 1. Stop accepting new HTTP requests and drain active requests
// 2. Stop worker pool and wait for in-flight tasks to finish
// 3. Close queue connections
func (s *Server) Shutdown(ctx context.Context) error {
	s.logger.Info("initiating graceful server shutdown")

	var errs []error

	// 1. Drain HTTP server requests
	if err := s.httpServer.Shutdown(ctx); err != nil {
		errs = append(errs, fmt.Errorf("HTTP server shutdown error: %w", err))
	}

	// 2. Drain Worker Pool
	if s.workerPool != nil {
		if err := s.workerPool.Stop(ctx); err != nil {
			errs = append(errs, fmt.Errorf("worker pool graceful stop error: %w", err))
		}
	}

	// 3. Close Queue
	if s.queue != nil {
		if err := s.queue.Close(); err != nil {
			errs = append(errs, fmt.Errorf("queue close error: %w", err))
		}
	}

	if len(errs) > 0 {
		return errors.Join(errs...)
	}

	s.logger.Info("server graceful shutdown completed successfully")
	return nil
}
