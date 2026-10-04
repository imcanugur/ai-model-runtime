package domain

import (
	"context"
	"fmt"
	"regexp"
	"strings"
)

// ModelID represents the unique slug name of a model family (e.g. "text-detector-tr").
type ModelID string

// ModelVersion represents a semver-compatible or incremental model revision (e.g. "2.1.0", "v1").
type ModelVersion string

// ModelKey is the immutable composite identifier for a loaded model instance: "<id>:<version>".
type ModelKey string

// NewModelKey creates a composite ModelKey from an ID and a Version.
func NewModelKey(id ModelID, version ModelVersion) ModelKey {
	return ModelKey(fmt.Sprintf("%s:%s", id, version))
}

// Split unpacks a ModelKey into its ModelID and ModelVersion parts.
func (k ModelKey) Split() (ModelID, ModelVersion, error) {
	parts := strings.Split(string(k), ":")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", "", fmt.Errorf("%w: malformed model key %q (expected 'id:version')", ErrInvalidManifest, string(k))
	}
	return ModelID(parts[0]), ModelVersion(parts[1]), nil
}

// ModelState represents the lifecycle state of a model instance.
type ModelState string

const (
	StateDiscovered ModelState = "DISCOVERED"
	StateValidating ModelState = "VALIDATING"
	StateLoading    ModelState = "LOADING"
	StateReady      ModelState = "READY"
	StateBusy       ModelState = "BUSY"
	StateUnloading  ModelState = "UNLOADING"
	StateUnloaded   ModelState = "UNLOADED"
	StateFailed     ModelState = "FAILED"
)

// ModelHandle provides temporary, reference-counted access to an acquired model instance.
type ModelHandle interface {
	ID() ModelID
	Version() ModelVersion
	Key() ModelKey
	Execute(ctx context.Context, input RawInput) (*PredictionResult, error)
	Release()
}

// Runtime defines the hardware execution boundary: pure tensors in, pure tensors out.
type Runtime interface {
	Infer(ctx context.Context, inputs TensorMap) (TensorMap, error)
	Close(ctx context.Context) error
}

// Pipeline orchestrates sequential stages on a PipelineContext to produce a PredictionResult.
type Pipeline interface {
	Execute(ctx context.Context, input RawInput) (*PredictionResult, error)
}

// ModelMetadata provides high-level descriptive information about a model package.
type ModelMetadata struct {
	ID          ModelID           `json:"id"`
	Name        string            `json:"name"`
	Version     ModelVersion      `json:"version"`
	Task        string            `json:"task,omitempty"`
	Languages   []string          `json:"languages,omitempty"`
	Description string            `json:"description,omitempty"`
	Author      string            `json:"author,omitempty"`
	Tags        map[string]string `json:"tags,omitempty"`
}

// ModelDescriptor represents the public health and status of a model in the registry.
type ModelDescriptor struct {
	Key          ModelKey      `json:"key"`
	Metadata     ModelMetadata `json:"metadata"`
	State        ModelState    `json:"state"`
	ActiveLeases int64         `json:"active_leases"`
	LastError    string        `json:"last_error,omitempty"`
}

// ModelFilter provides criteria for discovering or querying models in the registry.
type ModelFilter struct {
	Task     string
	Language string
	TagKey   string
	TagValue string
}

var (
	modelIDRegex      = regexp.MustCompile(`^[a-z0-9]+(-[a-z0-9]+)*$`)
	modelVersionRegex = regexp.MustCompile(`^v?[0-9]+(\.[0-9]+)*(-[a-zA-Z0-9.]+)?$`)
)

// ValidateModelID ensures the model ID contains only lowercase alphanumeric characters and hyphens.
func ValidateModelID(id ModelID) error {
	s := strings.TrimSpace(string(id))
	if s == "" {
		return NewValidationError("id", "model id cannot be empty", ErrInvalidModelID)
	}
	if len(s) > 64 {
		return NewValidationError("id", "model id exceeds maximum length of 64 characters", ErrInvalidModelID)
	}
	if !modelIDRegex.MatchString(s) {
		return NewValidationError("id", fmt.Sprintf("model id %q must be lowercase alphanumeric with hyphens (e.g. 'text-detector-tr')", s), ErrInvalidModelID)
	}
	return nil
}

// ValidateModelVersion ensures the version string is a valid semver or clean version tag.
func ValidateModelVersion(v ModelVersion) error {
	s := strings.TrimSpace(string(v))
	if s == "" {
		return NewValidationError("version", "model version cannot be empty", ErrInvalidModelVersion)
	}
	if len(s) > 32 {
		return NewValidationError("version", "model version exceeds maximum length of 32 characters", ErrInvalidModelVersion)
	}
	if !modelVersionRegex.MatchString(s) {
		return NewValidationError("version", fmt.Sprintf("model version %q must be valid version string (e.g. '1.0.0', 'v2', '2.1.0-rc1')", s), ErrInvalidModelVersion)
	}
	return nil
}
