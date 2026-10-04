package domain

import (
	"errors"
	"fmt"
)

// Sentinel domain errors for machine-checkable error classification.
var (
	ErrInvalidManifest       = errors.New("invalid model manifest")
	ErrInvalidModelID        = errors.New("invalid model id")
	ErrInvalidModelVersion   = errors.New("invalid model version")
	ErrInvalidArtifactPath   = errors.New("invalid artifact path")
	ErrArtifactNotFound      = errors.New("artifact not found")
	ErrPathTraversal         = errors.New("path traversal detected")
	ErrInvalidTensor         = errors.New("invalid tensor")
	ErrUnsupportedRuntime    = errors.New("unsupported runtime provider")
	ErrEmptyPipeline         = errors.New("pipeline must contain at least one stage")
	ErrDuplicateStage        = errors.New("duplicate pipeline stage name")
	ErrModelNotFound         = errors.New("model not found")
	ErrModelNotReady         = errors.New("model is not ready")
	ErrModelUnloading        = errors.New("model is currently unloading")
	ErrModelAlreadyLoaded    = errors.New("model is already loaded")
	ErrDuplicateModelKey     = errors.New("duplicate model key")
	ErrNoReadyVersion        = errors.New("no ready version found for model")
	ErrUnknownStageType      = errors.New("unknown pipeline stage type")
	ErrDuplicateStageType    = errors.New("duplicate stage type registration")
	ErrHandleAlreadyReleased = errors.New("model handle already released")
	ErrRuntimeClosed         = errors.New("runtime is closed")
)

// ValidationError provides structured, machine-inspectable details about a validation failure.
type ValidationError struct {
	Field  string
	Reason string
	Err    error
}

func (e *ValidationError) Error() string {
	if e.Field != "" {
		return fmt.Sprintf("validation failed on field '%s': %s", e.Field, e.Reason)
	}
	return fmt.Sprintf("validation failed: %s", e.Reason)
}

func (e *ValidationError) Unwrap() error {
	return e.Err
}

// NewValidationError creates a new ValidationError wrapping an underlying sentinel error.
func NewValidationError(field string, reason string, err error) *ValidationError {
	return &ValidationError{
		Field:  field,
		Reason: reason,
		Err:    err,
	}
}
