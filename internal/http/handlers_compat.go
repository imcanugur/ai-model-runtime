package http

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/service"
)

// HandleLegacyPredict provides backwards compatibility for POST /predict.
func HandleLegacyPredict(svc service.InferenceService, defaultModelID domain.ModelID) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req LegacyPredictRequest
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

		if strings.TrimSpace(req.Text) == "" {
			WriteCustomError(w, r, http.StatusUnprocessableEntity, "VALIDATION_ERROR", "text field cannot be empty")
			return
		}

		targetModel := domain.ModelID(strings.TrimSpace(req.Model))
		if targetModel == "" {
			targetModel = defaultModelID
		}

		input := domain.RawInput{
			Text:     req.Text,
			Language: req.Language,
		}

		result, resolvedKey, err := svc.SyncPredict(r.Context(), targetModel, domain.ModelVersion(req.Version), input)
		if err != nil {
			WriteError(w, r, err)
			return
		}

		respID, respVer, _ := resolvedKey.Split()
		result.Sync()
		resMap := map[string]any{
			"primary_score":     result.PrimaryScore,
			"label":             result.Label,
			"confidence":        result.Confidence,
			"execution_time_ms": result.Execution.DurationMs,
		}
		if result.Answer != "" {
			resMap["answer"] = result.Answer
		}
		resp := LegacyPredictResponse{
			Model: ModelMetaDTO{
				ID:      string(respID),
				Version: string(respVer),
			},
			Result:    resMap,
			Output:    result.Output,
			Execution: result.Execution,
		}

		WriteJSON(w, http.StatusOK, resp)
	}
}
