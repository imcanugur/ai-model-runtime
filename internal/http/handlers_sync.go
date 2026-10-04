package http

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/service"
)

// HandleSyncPredict handles POST /v1/models/{modelID}/predict and POST /v1/models/{modelID}/versions/{version}/predict.
func HandleSyncPredict(svc service.InferenceService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		modelIDStr := strings.TrimSpace(r.PathValue("modelID"))
		if modelIDStr == "" {
			WriteCustomError(w, r, http.StatusBadRequest, "INVALID_MODEL_ID", "model ID path parameter is required")
			return
		}

		versionStr := strings.TrimSpace(r.PathValue("version"))

		// Decode request payload
		var req PredictRequest
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

		input := req.ResolveInput()
		if strings.TrimSpace(input.Text) == "" && strings.TrimSpace(input.Context) == "" {
			WriteCustomError(w, r, http.StatusUnprocessableEntity, "VALIDATION_ERROR", "input text or context cannot be empty")
			return
		}

		// Execute through Application Service
		result, resolvedKey, err := svc.SyncPredict(r.Context(), domain.ModelID(modelIDStr), domain.ModelVersion(versionStr), input)
		if err != nil {
			WriteError(w, r, err)
			return
		}

		modelID, respVer, _ := resolvedKey.Split()
		result.Sync()
		if result.Execution.DurationMs > 0 {
			w.Header().Set("X-Execution-Duration-Ms", strconv.FormatInt(result.Execution.DurationMs, 10))
		}
		if respVer != "" {
			w.Header().Set("X-Model-Version", string(respVer))
		}

		outputVal := result.Output.Value
		if outputVal == nil {
			outputVal = map[string]any{}
		}

		reqID := GetRequestID(r.Context())
		resp := InferenceResponse{
			Data: InferenceData{
				Value: outputVal,
			},
			Meta: ResponseMeta{
				RequestID: reqID,
				Model: ModelMetaDTO{
					ID:      string(modelID),
					Version: string(respVer),
				},
				DurationMs: result.Execution.DurationMs,
			},
		}

		WriteJSON(w, http.StatusOK, resp)
	}
}
