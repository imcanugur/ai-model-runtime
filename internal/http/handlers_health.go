package http

import (
	"net/http"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/service"
)

// HandleHealth handles GET /health (liveness probe).
func HandleHealth() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		WriteJSON(w, http.StatusOK, HealthResponse{
			Status:    "ok",
			Timestamp: time.Now().UTC().Format(time.RFC3339),
		})
	}
}

// HandleReady handles GET /ready (readiness probe).
func HandleReady(svc service.InferenceService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ready, deps := svc.CheckReadiness(r.Context())
		resp := ReadyResponse{
			Status:       "ready",
			Dependencies: deps,
		}

		if !ready {
			resp.Status = "not_ready"
			WriteJSON(w, http.StatusServiceUnavailable, resp)
			return
		}

		WriteJSON(w, http.StatusOK, resp)
	}
}
