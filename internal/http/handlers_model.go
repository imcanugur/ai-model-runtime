package http

import (
	"net/http"

	"github.com/imcanugur/ai-model-runtime/internal/service"
)

// HandleListModels handles GET /v1/models.
func HandleListModels(svc service.InferenceService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		descriptors := svc.ListModels(r.Context())

		models := make([]ModelInfoDTO, len(descriptors))
		for i, d := range descriptors {
			models[i] = ModelInfoDTO{
				ID:           string(d.Metadata.ID),
				Version:      string(d.Metadata.Version),
				Name:         d.Metadata.Name,
				State:        string(d.State),
				Status:       string(d.State),
				Task:         d.Metadata.Task,
				Languages:    d.Metadata.Languages,
				ActiveLeases: d.ActiveLeases,
			}
		}

		WriteJSON(w, http.StatusOK, ListModelsResponse{Models: models})
	}
}

// HandleGetModel handles GET /v1/models/{modelID}.
func HandleGetModel(svc service.InferenceService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		modelID := r.PathValue("modelID")
		if modelID == "" {
			WriteCustomError(w, r, http.StatusBadRequest, "INVALID_MODEL_ID", "modelID path parameter is required")
			return
		}

		descriptors := svc.ListModels(r.Context())
		for _, d := range descriptors {
			if string(d.Metadata.ID) == modelID {
				WriteJSON(w, http.StatusOK, ModelInfoDTO{
					ID:           string(d.Metadata.ID),
					Version:      string(d.Metadata.Version),
					Name:         d.Metadata.Name,
					State:        string(d.State),
					Status:       string(d.State),
					Task:         d.Metadata.Task,
					Languages:    d.Metadata.Languages,
					ActiveLeases: d.ActiveLeases,
				})
				return
			}
		}

		WriteCustomError(w, r, http.StatusNotFound, "MODEL_NOT_FOUND", "model not found: "+modelID)
	}
}
