package http

import (
	"log/slog"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

// ServerConfig parameterizes the HTTP server.
type ServerConfig struct {
	Address        string
	ReadTimeout    time.Duration
	WriteTimeout   time.Duration
	IdleTimeout    time.Duration
	RequestTimeout time.Duration
	MaxBodyBytes   int64
	DefaultModelID domain.ModelID
	Logger         *slog.Logger
}

// DefaultServerConfig returns production-grade server defaults.
func DefaultServerConfig() ServerConfig {
	return ServerConfig{
		Address:        ":8080",
		ReadTimeout:    10 * time.Second,
		WriteTimeout:   30 * time.Second,
		IdleTimeout:    60 * time.Second,
		RequestTimeout: 30 * time.Second,
		MaxBodyBytes:   4 * 1024 * 1024, // 4 MB
		Logger:         slog.Default(),
	}
}
