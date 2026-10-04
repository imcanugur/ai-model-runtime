package http

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/service"
)

// HandleCreateJob handles POST /v1/jobs.
func HandleCreateJob(svc service.InferenceService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		idempKey := strings.TrimSpace(r.Header.Get("Idempotency-Key"))

		var req CreateJobRequest
		dec := json.NewDecoder(r.Body)
		if err := dec.Decode(&req); err != nil {
			if errors.Is(err, io.EOF) {
				WriteCustomError(w, r, http.StatusBadRequest, "EMPTY_BODY", "request body cannot be empty")
				return
			}
			var maxBytesErr *http.MaxBytesError
			if errors.As(err, &maxBytesErr) {
				WriteCustomError(w, r, http.StatusRequestEntityTooLarge, "PAYLOAD_TOO_LARGE", "request payload exceeds maximum allowed size")
				return
			}
			WriteCustomError(w, r, http.StatusBadRequest, "MALFORMED_JSON", "failed to parse JSON request body: "+err.Error())
			return
		}

		modelID := strings.TrimSpace(req.ResolveModelID())
		if modelID == "" {
			WriteCustomError(w, r, http.StatusUnprocessableEntity, "VALIDATION_ERROR", "model_id field is required")
			return
		}

		input := req.ResolveInput()
		if strings.TrimSpace(input.Text) == "" && strings.TrimSpace(input.Context) == "" {
			WriteCustomError(w, r, http.StatusUnprocessableEntity, "VALIDATION_ERROR", "input text or context cannot be empty")
			return
		}

		job, wasExisting, err := svc.SubmitJob(
			r.Context(),
			domain.ModelID(modelID),
			domain.ModelVersion(req.Version),
			input,
			req.MaxAttempts,
			idempKey,
		)
		if err != nil {
			WriteError(w, r, err)
			return
		}

		resp := CreateJobResponse{
			JobID:     string(job.ID),
			State:     string(job.State),
			CreatedAt: job.CreatedAt.Format(time.RFC3339),
		}

		status := http.StatusCreated
		if wasExisting {
			status = http.StatusOK
		}

		WriteJSON(w, status, resp)
	}
}

// HandleGetJob handles GET /v1/jobs/{jobID}.
func HandleGetJob(svc service.InferenceService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		jobIDStr := strings.TrimSpace(r.PathValue("jobID"))
		if jobIDStr == "" {
			WriteCustomError(w, r, http.StatusBadRequest, "INVALID_JOB_ID", "job ID path parameter is required")
			return
		}

		job, err := svc.GetJob(r.Context(), domain.JobID(jobIDStr))
		if err != nil {
			WriteError(w, r, err)
			return
		}

		resp := MapJobStatus(job)
		WriteJSON(w, http.StatusOK, resp)
	}
}
