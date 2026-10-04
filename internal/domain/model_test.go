package domain_test

import (
	"errors"
	"testing"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

func TestModelKey_CreationAndSplit(t *testing.T) {
	key := domain.NewModelKey("detector-tr", "2.1.0")
	if key != "detector-tr:2.1.0" {
		t.Errorf("unexpected key format: %s", key)
	}

	id, ver, err := key.Split()
	if err != nil {
		t.Fatalf("unexpected split error: %v", err)
	}
	if id != "detector-tr" {
		t.Errorf("expected id 'detector-tr', got %s", id)
	}
	if ver != "2.1.0" {
		t.Errorf("expected version '2.1.0', got %s", ver)
	}

	// Malformed key
	badKey := domain.ModelKey("invalid_key_without_colon")
	_, _, err = badKey.Split()
	if err == nil {
		t.Error("expected error splitting malformed key, got nil")
	}
}

func TestValidateModelID(t *testing.T) {
	validIDs := []domain.ModelID{
		"text-detector",
		"model1",
		"detector-tr-v2",
		"academic-detector-2026",
	}
	for _, id := range validIDs {
		if err := domain.ValidateModelID(id); err != nil {
			t.Errorf("expected valid id for %q, got error: %v", id, err)
		}
	}

	invalidIDs := []domain.ModelID{
		"",
		" ",
		"Text-Detector", // uppercase rejected
		"text_detector", // underscore rejected
		"-detector",     // leading hyphen
		"detector-",     // trailing hyphen
		"text/detector", // slash rejected
		"model@name",    // special characters rejected
	}
	for _, id := range invalidIDs {
		if err := domain.ValidateModelID(id); err == nil {
			t.Errorf("expected validation failure for id %q, got nil", id)
		} else if !errors.Is(err, domain.ErrInvalidModelID) {
			t.Errorf("expected ErrInvalidModelID for %q, got %v", id, err)
		}
	}
}

func TestValidateModelVersion(t *testing.T) {
	validVersions := []domain.ModelVersion{
		"1.0.0",
		"2.1.4",
		"v1",
		"0.1",
		"1.0.0-rc1",
		"2.0.0-beta.2",
	}
	for _, v := range validVersions {
		if err := domain.ValidateModelVersion(v); err != nil {
			t.Errorf("expected valid version for %q, got error: %v", v, err)
		}
	}

	invalidVersions := []domain.ModelVersion{
		"",
		" ",
		"version 1.0", // spaces rejected
		"1.0.0/prod",  // slashes rejected
		"ver:1",       // colons rejected
	}
	for _, v := range invalidVersions {
		if err := domain.ValidateModelVersion(v); err == nil {
			t.Errorf("expected validation failure for version %q, got nil", v)
		} else if !errors.Is(err, domain.ErrInvalidModelVersion) {
			t.Errorf("expected ErrInvalidModelVersion for %q, got %v", v, err)
		}
	}
}
