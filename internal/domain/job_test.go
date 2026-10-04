package domain_test

import (
	"errors"
	"testing"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

type customRetryableError struct {
	retryable bool
	msg       string
}

func (e *customRetryableError) Error() string   { return e.msg }
func (e *customRetryableError) Retryable() bool { return e.retryable }

func TestJob_StateTransitions(t *testing.T) {
	tests := []struct {
		from  domain.JobState
		to    domain.JobState
		valid bool
	}{
		// Valid transitions
		{domain.JobQueued, domain.JobRunning, true},
		{domain.JobRunning, domain.JobCompleted, true},
		{domain.JobRunning, domain.JobFailed, true},
		{domain.JobRunning, domain.JobQueued, true}, // Retry / Recovery

		// Invalid transitions from terminal states
		{domain.JobCompleted, domain.JobRunning, false},
		{domain.JobCompleted, domain.JobFailed, false},
		{domain.JobFailed, domain.JobRunning, false},
		{domain.JobFailed, domain.JobQueued, false},

		// Invalid skipping transitions
		{domain.JobQueued, domain.JobCompleted, false},
		{domain.JobQueued, domain.JobFailed, false},

		// Unknown states
		{"unknown", domain.JobRunning, false},
		{domain.JobRunning, "unknown", false},
	}

	for _, tt := range tests {
		t.Run(string(tt.from)+"->"+string(tt.to), func(t *testing.T) {
			err := domain.ValidateTransition(tt.from, tt.to)
			if tt.valid && err != nil {
				t.Errorf("expected transition %s -> %s to be valid, got: %v", tt.from, tt.to, err)
			}
			if !tt.valid && err == nil {
				t.Errorf("expected transition %s -> %s to be invalid, got nil", tt.from, tt.to)
			}
		})
	}
}

func TestJob_IsRetryable(t *testing.T) {
	// 1. Nil error
	if domain.IsRetryable(nil) {
		t.Error("expected nil error to not be retryable")
	}

	// 2. Permanent domain errors
	if domain.IsRetryable(domain.ErrModelNotFound) {
		t.Error("expected ErrModelNotFound to be non-retryable")
	}
	if domain.IsRetryable(domain.ErrInvalidTensor) {
		t.Error("expected ErrInvalidTensor to be non-retryable")
	}

	// 3. Custom retryable error
	retErr := &customRetryableError{retryable: true, msg: "transient network glitch"}
	if !domain.IsRetryable(retErr) {
		t.Error("expected custom retryable error to be retryable")
	}

	nonRetErr := &customRetryableError{retryable: false, msg: "bad request payload"}
	if domain.IsRetryable(nonRetErr) {
		t.Error("expected custom non-retryable error to not be retryable")
	}

	// 4. Default unknown transient error
	unknownErr := errors.New("temporary database connection refused")
	if !domain.IsRetryable(unknownErr) {
		t.Error("expected unknown error to be retryable by default")
	}
}
