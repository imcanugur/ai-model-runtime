package http

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

// APIErrorDetail holds the structured error body.
type APIErrorDetail struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	RequestID string `json:"request_id,omitempty"`
}

// ErrorMeta holds error-level telemetry metadata.
type ErrorMeta struct {
	RequestID string `json:"request_id"`
}

// ErrorResponse is the standardized JSON error envelope.
type ErrorResponse struct {
	Error APIErrorDetail `json:"error"`
	Meta  ErrorMeta      `json:"meta"`
}

// WriteJSON sends a JSON response with status code.
func WriteJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(data)
}

// WriteError translates an error into a structured JSON error response.
func WriteError(w http.ResponseWriter, r *http.Request, err error) {
	reqID := GetRequestID(r.Context())
	status, code, message := mapDomainErrorToHTTP(err)

	if status >= 500 {
		slog.Error("HTTP internal error", "method", r.Method, "path", r.URL.Path, "status", status, "code", code, "request_id", reqID, "error", err)
	} else {
		slog.Warn("HTTP client error", "method", r.Method, "path", r.URL.Path, "status", status, "code", code, "request_id", reqID, "error", err)
	}

	WriteJSON(w, status, ErrorResponse{
		Error: APIErrorDetail{
			Code:    code,
			Message: message,
		},
		Meta: ErrorMeta{
			RequestID: reqID,
		},
	})
}

// WriteCustomError sends an explicit HTTP status and code.
func WriteCustomError(w http.ResponseWriter, r *http.Request, status int, code, message string) {
	reqID := GetRequestID(r.Context())
	WriteJSON(w, status, ErrorResponse{
		Error: APIErrorDetail{
			Code:    code,
			Message: message,
		},
		Meta: ErrorMeta{
			RequestID: reqID,
		},
	})
}

func mapDomainErrorToHTTP(err error) (int, string, string) {
	if err == nil {
		return http.StatusOK, "OK", ""
	}

	// 1. Validation Errors
	var valErr *domain.ValidationError
	if errors.As(err, &valErr) {
		return http.StatusUnprocessableEntity, "VALIDATION_ERROR", valErr.Error()
	}

	// 2. Not Found Errors
	if errors.Is(err, domain.ErrModelNotFound) || errors.Is(err, domain.ErrNoReadyVersion) {
		return http.StatusNotFound, "MODEL_NOT_FOUND", "requested model not found in registry or has no ready versions"
	}
	if errors.Is(err, domain.ErrJobNotFound) {
		return http.StatusNotFound, "JOB_NOT_FOUND", "requested job not found"
	}

	// 3. Model Lifecycle & Readiness Errors
	if errors.Is(err, domain.ErrModelNotReady) || errors.Is(err, domain.ErrModelUnloading) {
		return http.StatusServiceUnavailable, "MODEL_NOT_READY", "requested model is not currently ready to serve traffic"
	}

	// 3. Bad Request Errors
	if errors.Is(err, domain.ErrInvalidModelID) {
		return http.StatusBadRequest, "INVALID_MODEL_ID", err.Error()
	}
	if errors.Is(err, domain.ErrInvalidModelVersion) {
		return http.StatusBadRequest, "INVALID_MODEL_VERSION", err.Error()
	}
	if errors.Is(err, domain.ErrInvalidManifest) {
		return http.StatusBadRequest, "INVALID_MANIFEST", err.Error()
	}

	// 4. Conflict Errors
	if errors.Is(err, domain.ErrInvalidJobStateTransition) {
		return http.StatusConflict, "INVALID_STATE_TRANSITION", err.Error()
	}
	if errors.Is(err, domain.ErrJobAlreadyClaimed) {
		return http.StatusConflict, "JOB_ALREADY_CLAIMED", err.Error()
	}

	// 5. Dependency / Unavailable Errors
	if errors.Is(err, domain.ErrRuntimeClosed) {
		return http.StatusServiceUnavailable, "RUNTIME_CLOSED", "inference runtime is closed"
	}

	// 6. Generic Internal Error (do not leak internal system details)
	return http.StatusInternalServerError, "INTERNAL_ERROR", "an unexpected error occurred during processing"
}
