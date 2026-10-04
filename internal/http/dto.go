package http

import (
	"fmt"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

// RawInputDTO represents the input payload in HTTP requests.
type RawInputDTO struct {
	Text       string                 `json:"text,omitempty"`
	Context    string                 `json:"context,omitempty"`
	Question   string                 `json:"question,omitempty"`
	Language   string                 `json:"language,omitempty"`
	Documents  []domain.InputDocument `json:"documents,omitempty"`
	Parameters map[string]string      `json:"parameters,omitempty"`
}

// ToDomain maps the DTO to domain.RawInput.
func (d *RawInputDTO) ToDomain() domain.RawInput {
	if d == nil {
		return domain.RawInput{}
	}
	text := d.Text
	if text == "" && d.Context != "" {
		text = d.Context
	}
	if text == "" && len(d.Documents) > 0 {
		text = d.Documents[0].Text
	}
	return domain.RawInput{
		Text:       text,
		Context:    d.Context,
		Question:   d.Question,
		Language:   d.Language,
		Parameters: d.Parameters,
	}
}

// InferenceRequest represents the canonical universal payload for inference requests.
type InferenceRequest struct {
	Input   *RawInputDTO   `json:"input,omitempty"`
	Text    string         `json:"text,omitempty"` // Shorthand convenience
	Options map[string]any `json:"options,omitempty"`
}

// PredictRequest is an alias to InferenceRequest for backward compatibility.
type PredictRequest = InferenceRequest

// ResolveInput extracts domain.RawInput from InferenceRequest.
func (r *InferenceRequest) ResolveInput() domain.RawInput {
	var input domain.RawInput
	if r.Input != nil {
		input = r.Input.ToDomain()
	} else if r.Text != "" {
		input = domain.RawInput{Text: r.Text}
	}
	if len(r.Options) > 0 {
		if input.Parameters == nil {
			input.Parameters = make(map[string]string)
		}
		for k, v := range r.Options {
			input.Parameters[k] = fmt.Sprintf("%v", v)
		}
	}
	return input
}

// InferenceData encapsulates model output payload.
type InferenceData struct {
	Value any `json:"value"`
}

// ModelMetaDTO identifies the model answering the request.
type ModelMetaDTO struct {
	ID      string `json:"id"`
	Version string `json:"version"`
}

// ResponseMeta provides standardized operational telemetry.
type ResponseMeta struct {
	RequestID  string       `json:"request_id"`
	Model      ModelMetaDTO `json:"model"`
	DurationMs int64        `json:"duration_ms"`
}

// InferenceResponse represents the canonical universal envelope returned by all models.
type InferenceResponse struct {
	Data InferenceData `json:"data"`
	Meta ResponseMeta  `json:"meta"`
}

// PredictResponse is an alias to InferenceResponse.
type PredictResponse = InferenceResponse

// CreateJobRequest represents the payload for POST /v1/jobs.
type CreateJobRequest struct {
	ModelID     string         `json:"model_id,omitempty"`
	Model       string         `json:"model,omitempty"` // Backwards-compatible alias
	Version     string         `json:"version,omitempty"`
	Input       *RawInputDTO   `json:"input,omitempty"`
	Text        string         `json:"text,omitempty"`
	Options     map[string]any `json:"options,omitempty"`
	MaxAttempts int            `json:"max_attempts,omitempty"`
}

// ResolveModelID returns model_id, falling back to model.
func (r *CreateJobRequest) ResolveModelID() string {
	if r.ModelID != "" {
		return r.ModelID
	}
	return r.Model
}

// ResolveInput extracts domain.RawInput from CreateJobRequest.
func (r *CreateJobRequest) ResolveInput() domain.RawInput {
	var input domain.RawInput
	if r.Input != nil && (r.Input.Text != "" || r.Input.Context != "") {
		input = r.Input.ToDomain()
	} else if r.Text != "" {
		input = domain.RawInput{Text: r.Text}
	} else if r.Input != nil {
		input = r.Input.ToDomain()
	}
	if len(r.Options) > 0 {
		if input.Parameters == nil {
			input.Parameters = make(map[string]string)
		}
		for k, v := range r.Options {
			input.Parameters[k] = fmt.Sprintf("%v", v)
		}
	}
	return input
}

// CreateJobResponse represents the response envelope for POST /v1/jobs.
type CreateJobResponse struct {
	JobID     string `json:"job_id"`
	State     string `json:"state"`
	CreatedAt string `json:"created_at"`
}

// JobStatusResponse represents the canonical response for GET /v1/jobs/{jobID}.
type JobStatusResponse struct {
	JobID        string         `json:"job_id"`
	ModelID      string         `json:"model_id"`
	ModelVersion string         `json:"model_version,omitempty"`
	State        string         `json:"state"`
	Attempt      int            `json:"attempt"`
	MaxAttempts  int            `json:"max_attempts"`
	CreatedAt    string         `json:"created_at"`
	StartedAt    *string        `json:"started_at,omitempty"`
	CompletedAt  *string        `json:"completed_at,omitempty"`
	Data         *InferenceData `json:"data,omitempty"`
	Meta         *ResponseMeta  `json:"meta,omitempty"`
	Error        string         `json:"error,omitempty"`
}

// MapJobStatus converts domain.Job to JobStatusResponse.
func MapJobStatus(job *domain.Job) JobStatusResponse {
	resp := JobStatusResponse{
		JobID:       string(job.ID),
		ModelID:     string(job.ModelID),
		State:       string(job.State),
		Attempt:     job.Attempt,
		MaxAttempts: job.MaxAttempts,
		CreatedAt:   job.CreatedAt.Format(time.RFC3339),
		Error:       job.Error,
	}

	if job.ModelKey != "" {
		_, ver, err := job.ModelKey.Split()
		if err == nil {
			resp.ModelVersion = string(ver)
		}
	}

	if job.StartedAt != nil {
		t := job.StartedAt.Format(time.RFC3339)
		resp.StartedAt = &t
	}
	if job.CompletedAt != nil {
		t := job.CompletedAt.Format(time.RFC3339)
		resp.CompletedAt = &t
	}
	if job.Result != nil {
		job.Result.Sync()
		resp.Data = &InferenceData{Value: job.Result.Output.Value}
		resp.Meta = &ResponseMeta{
			RequestID: string(job.ID),
			Model: ModelMetaDTO{
				ID:      string(job.ModelID),
				Version: resp.ModelVersion,
			},
			DurationMs: job.Result.Execution.DurationMs,
		}
	}

	return resp
}

// ModelInfoDTO represents public model descriptor information.
type ModelInfoDTO struct {
	ID           string   `json:"id"`
	Version      string   `json:"version"`
	Name         string   `json:"name"`
	State        string   `json:"state"`
	Status       string   `json:"status,omitempty"`
	Task         string   `json:"task,omitempty"`
	Languages    []string `json:"languages,omitempty"`
	ActiveLeases int64    `json:"active_leases"`
}

// ListModelsResponse represents GET /v1/models response.
type ListModelsResponse struct {
	Models []ModelInfoDTO `json:"models"`
}

// HealthResponse represents GET /health response.
type HealthResponse struct {
	Status    string `json:"status"`
	Timestamp string `json:"timestamp"`
}

// ReadyResponse represents GET /ready response.
type ReadyResponse struct {
	Status       string            `json:"status"`
	Dependencies map[string]string `json:"dependencies"`
}

// LegacyPredictRequest represents legacy Python POST /predict payload.
type LegacyPredictRequest struct {
	Text     string `json:"text"`
	Language string `json:"language,omitempty"`
	Model    string `json:"model,omitempty"`
	Version  string `json:"version,omitempty"`
}

// LegacyPredictResponse represents legacy Python POST /predict response envelope.
type LegacyPredictResponse struct {
	Model     ModelMetaDTO            `json:"model"`
	Result    map[string]any          `json:"result,omitempty"`
	Output    domain.PredictionOutput `json:"output"`
	Execution domain.ExecutionInfo    `json:"execution"`
}
