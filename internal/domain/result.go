package domain

// Classification represents a predicted class label with associated calibrated probability.
type Classification struct {
	Label       string  `json:"label"`
	Probability float64 `json:"probability"`
}

// ChunkDetail represents fine-grained scoring per chunk or sentence window.
type ChunkDetail struct {
	ID         int     `json:"id"`
	Text       string  `json:"text"`
	Score      float64 `json:"score"`
	TokenCount int     `json:"token_count"`
}

// ModelInfo captures discovery and specification metadata of the executing model.
type ModelInfo struct {
	ID        ModelID      `json:"id"`
	Version   ModelVersion `json:"version"`
	Task      string       `json:"task,omitempty"`
	Framework string       `json:"framework,omitempty"`
}

// ExecutionInfo captures performance, queuing, and profiling telemetry.
type ExecutionInfo struct {
	DurationMs       int64            `json:"duration_ms"`
	QueueWaitMs      int64            `json:"queue_wait_ms,omitempty"`
	InferenceMs      int64            `json:"inference_ms,omitempty"`
	StageDurationsMs map[string]int64 `json:"stage_durations_ms,omitempty"`
	RuntimeProvider  string           `json:"runtime_provider,omitempty"`
}

// PredictionOutput is the universal task-agnostic output envelope.
type PredictionOutput struct {
	Value any `json:"value"`
}

// InferenceData encapsulates model output payload in domain.
type InferenceData struct {
	Value any `json:"value"`
}

// ResponseMeta captures model and execution metadata in domain.
type ResponseMeta struct {
	RequestID  string    `json:"request_id"`
	Model      ModelInfo `json:"model"`
	DurationMs int64     `json:"duration_ms"`
}

// InferenceResponse represents the canonical universal envelope.
type InferenceResponse struct {
	Data InferenceData `json:"data"`
	Meta ResponseMeta  `json:"meta"`
}

// PredictionResult represents the finalized, strongly typed domain result returned by a pipeline.
type PredictionResult struct {
	Model     ModelInfo        `json:"model"`
	Output    PredictionOutput `json:"output"`
	Execution ExecutionInfo    `json:"execution"`

	// Backward-compatibility fields (kept synchronized for existing consumers)
	ModelID         ModelID           `json:"model_id,omitempty"`
	ModelVersion    ModelVersion      `json:"model_version,omitempty"`
	PrimaryScore    float64           `json:"primary_score,omitempty"`
	Label           string            `json:"label,omitempty"`
	Confidence      float64           `json:"confidence,omitempty"`
	Answer          string            `json:"answer,omitempty"`
	Classifications []Classification  `json:"classifications,omitempty"`
	Chunks          []ChunkDetail     `json:"chunks,omitempty"`
	ExecutionTimeMs int64             `json:"execution_time_ms,omitempty"`
	Metadata        map[string]string `json:"metadata,omitempty"`
}

// Sync ensures bidirectional synchronization between universal PredictionOutput and legacy fields.
func (r *PredictionResult) Sync() {
	if r.ModelID != "" && r.Model.ID == "" {
		r.Model.ID = r.ModelID
	}
	if r.Model.ID != "" && r.ModelID == "" {
		r.ModelID = r.Model.ID
	}
	if r.ModelVersion != "" && r.Model.Version == "" {
		r.Model.Version = r.ModelVersion
	}
	if r.Model.Version != "" && r.ModelVersion == "" {
		r.ModelVersion = r.Model.Version
	}
	if r.ExecutionTimeMs != 0 && r.Execution.DurationMs == 0 {
		r.Execution.DurationMs = r.ExecutionTimeMs
	}
	if r.Execution.DurationMs != 0 && r.ExecutionTimeMs == 0 {
		r.ExecutionTimeMs = r.Execution.DurationMs
	}

	if r.Output.Value == nil {
		if r.Answer != "" {
			r.Output.Value = map[string]any{
				"answer":     r.Answer,
				"confidence": r.Confidence,
			}
		} else if r.Label != "" {
			r.Output.Value = map[string]any{
				"label":      r.Label,
				"score":      r.PrimaryScore,
				"confidence": r.Confidence,
			}
		}
	}
}
