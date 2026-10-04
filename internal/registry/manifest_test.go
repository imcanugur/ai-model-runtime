package registry_test

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/pipeline"
	"github.com/imcanugur/ai-model-runtime/internal/registry"
)

const validYAML = `
schema_version: "v1"
id: "text-detector-tr"
name: "Turkish Academic Text Detector"
version: "2.1.0"
metadata:
  task: "text-classification"
  languages: ["tr"]
resources:
  max_concurrency: 4
  timeout_ms: 60000
runtime:
  provider: "onnx"
  profile: "standard_cpu"
  entrypoint: "weights/model.onnx"
pipeline:
  stages:
    - name: "normalizer"
      type: "text_normalizer"
    - name: "chunker"
      type: "sliding_window_chunker"
      config:
        tokenizer_file: "assets/tokenizer.json"
    - name: "infer"
      type: "runtime_infer"
`

func setupValidModelDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	if err := os.MkdirAll(filepath.Join(dir, "weights"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "assets"), 0755); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(dir, "model.yaml"), []byte(validYAML), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "weights", "model.onnx"), []byte("mock-onnx-bytes"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "assets", "tokenizer.json"), []byte("{}"), 0644); err != nil {
		t.Fatal(err)
	}

	return dir
}

func TestManifest_ValidParsingAndValidation(t *testing.T) {
	modelDir := setupValidModelDir(t)
	manifestPath := filepath.Join(modelDir, "model.yaml")

	manifest, err := registry.ParseManifestFile(manifestPath)
	if err != nil {
		t.Fatalf("expected valid manifest parse, got: %v", err)
	}

	validator := registry.NewValidator(true)
	if err := validator.ValidateModelPackage(modelDir, manifest); err != nil {
		t.Fatalf("expected valid model package validation, got: %v", err)
	}

	if manifest.ID != "text-detector-tr" {
		t.Errorf("expected ID 'text-detector-tr', got %s", manifest.ID)
	}
	if manifest.Version != "2.1.0" {
		t.Errorf("expected Version '2.1.0', got %s", manifest.Version)
	}
}

func TestManifest_MissingFile(t *testing.T) {
	_, err := registry.ParseManifestFile("/non/existent/path/model.yaml")
	if err == nil {
		t.Fatal("expected error for non-existent manifest file, got nil")
	}
}

func TestManifest_InvalidYAMLSyntax(t *testing.T) {
	invalidYAML := `
schema_version: "v1"
id: "text-detector"
  invalid_indentation: [
`
	_, err := registry.ParseManifest([]byte(invalidYAML))
	if err == nil {
		t.Fatal("expected error for invalid YAML syntax, got nil")
	}
}

func TestManifest_MissingRequiredFields(t *testing.T) {
	validator := registry.NewValidator(false)

	tests := []struct {
		name     string
		mutate   func(*registry.ModelManifest)
		expected error
	}{
		{
			name: "missing schema_version",
			mutate: func(m *registry.ModelManifest) {
				m.SchemaVersion = ""
			},
			expected: domain.ErrInvalidManifest,
		},
		{
			name: "unsupported schema_version",
			mutate: func(m *registry.ModelManifest) {
				m.SchemaVersion = "v99"
			},
			expected: domain.ErrInvalidManifest,
		},
		{
			name: "missing id",
			mutate: func(m *registry.ModelManifest) {
				m.ID = ""
			},
			expected: domain.ErrInvalidModelID,
		},
		{
			name: "invalid id format",
			mutate: func(m *registry.ModelManifest) {
				m.ID = "UPPERCASE_ID"
			},
			expected: domain.ErrInvalidModelID,
		},
		{
			name: "missing name",
			mutate: func(m *registry.ModelManifest) {
				m.Name = ""
			},
			expected: domain.ErrInvalidManifest,
		},
		{
			name: "missing version",
			mutate: func(m *registry.ModelManifest) {
				m.Version = ""
			},
			expected: domain.ErrInvalidModelVersion,
		},
		{
			name: "missing runtime provider",
			mutate: func(m *registry.ModelManifest) {
				m.Runtime.Provider = ""
			},
			expected: domain.ErrUnsupportedRuntime,
		},
		{
			name: "unsupported runtime provider",
			mutate: func(m *registry.ModelManifest) {
				m.Runtime.Provider = "unsupported_lib"
			},
			expected: domain.ErrUnsupportedRuntime,
		},
		{
			name: "missing runtime profile",
			mutate: func(m *registry.ModelManifest) {
				m.Runtime.Profile = ""
			},
			expected: domain.ErrInvalidManifest,
		},
		{
			name: "missing runtime entrypoint",
			mutate: func(m *registry.ModelManifest) {
				m.Runtime.Entrypoint = ""
			},
			expected: domain.ErrInvalidArtifactPath,
		},
		{
			name: "empty pipeline",
			mutate: func(m *registry.ModelManifest) {
				m.Pipeline.Stages = nil
			},
			expected: domain.ErrEmptyPipeline,
		},
		{
			name: "duplicate stage name",
			mutate: func(m *registry.ModelManifest) {
				m.Pipeline.Stages = []pipeline.StageConfig{
					{Name: "same_name", Type: "type1"},
					{Name: "same_name", Type: "type2"},
				}
			},
			expected: domain.ErrDuplicateStage,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := registry.ParseManifest([]byte(validYAML))
			if err != nil {
				t.Fatalf("failed to parse base valid manifest: %v", err)
			}
			tt.mutate(m)

			err = validator.ValidateManifest(m)
			if err == nil {
				t.Fatalf("expected validation error for %s, got nil", tt.name)
			}
			if !errors.Is(err, tt.expected) {
				t.Errorf("expected error wrapping %v, got %v", tt.expected, err)
			}
		})
	}
}

func TestManifest_MissingEntrypointOnDisk(t *testing.T) {
	modelDir := t.TempDir()
	manifest, err := registry.ParseManifest([]byte(validYAML))
	if err != nil {
		t.Fatal(err)
	}

	// File existence check is enabled, but weights/model.onnx is missing on disk
	validator := registry.NewValidator(true)
	err = validator.ValidateModelPackage(modelDir, manifest)
	if err == nil {
		t.Fatal("expected error for missing entrypoint file on disk, got nil")
	}
	if !errors.Is(err, domain.ErrArtifactNotFound) {
		t.Errorf("expected ErrArtifactNotFound, got %v", err)
	}
}

func TestManifest_PathTraversalRejection(t *testing.T) {
	modelDir := t.TempDir()
	validator := registry.NewValidator(false)

	traversalPaths := []string{
		"../../etc/passwd",
		"../secrets.yaml",
		"/etc/passwd",
		"\\Windows\\System32\\cmd.exe",
		"sub/../../outside",
		"C:/boot.ini",
	}

	for _, badPath := range traversalPaths {
		t.Run(badPath, func(t *testing.T) {
			m, err := registry.ParseManifest([]byte(validYAML))
			if err != nil {
				t.Fatal(err)
			}
			m.Runtime.Entrypoint = badPath

			err = validator.ValidateModelPackage(modelDir, m)
			if err == nil {
				t.Fatalf("expected path traversal rejection for %q, got nil", badPath)
			}
			if !errors.Is(err, domain.ErrPathTraversal) {
				t.Errorf("expected ErrPathTraversal for %q, got %v", badPath, err)
			}
		})
	}
}

func TestManifest_SymlinkEscapeRejection(t *testing.T) {
	tempBase := t.TempDir()
	modelDir := filepath.Join(tempBase, "model")
	outsideDir := filepath.Join(tempBase, "outside")

	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(outsideDir, 0755); err != nil {
		t.Fatal(err)
	}

	secretFile := filepath.Join(outsideDir, "secret.onnx")
	if err := os.WriteFile(secretFile, []byte("secret"), 0644); err != nil {
		t.Fatal(err)
	}

	// Create symlink inside modelDir pointing to secret outside
	symlinkPath := filepath.Join(modelDir, "symlink_escape.onnx")
	err := os.Symlink(secretFile, symlinkPath)
	if err != nil {
		t.Skip("Symlinks not supported in this test environment, skipping")
	}

	validator := registry.NewValidator(false)
	_, err = validator.ValidateArtifactPath(modelDir, "symlink_escape.onnx")
	if err == nil {
		t.Fatal("expected symlink escape rejection, got nil")
	}
	if !errors.Is(err, domain.ErrPathTraversal) {
		t.Errorf("expected ErrPathTraversal, got %v", err)
	}
}

func TestBERTurkRev2_ManifestAndModelPackageValidation(t *testing.T) {
	bestModelDir := os.Getenv("BERTURK_MODEL_DIR")
	if bestModelDir == "" {
		cwd, err := os.Getwd()
		if err == nil {
			parent := filepath.Dir(filepath.Dir(cwd))
			entries, _ := os.ReadDir(parent)
			for _, e := range entries {
				candidate := filepath.Join(parent, e.Name(), "best_model")
				if _, err := os.Stat(filepath.Join(candidate, "model.yaml")); err == nil {
					bestModelDir = candidate
					break
				}
			}
		}
	}

	if bestModelDir == "" {
		t.Skip("best_model/model.yaml not found on disk")
	}

	manifest, err := registry.ParseManifestFile(filepath.Join(bestModelDir, "model.yaml"))
	if err != nil {
		t.Fatalf("failed to parse BERTurk model.yaml: %v", err)
	}

	if manifest.ID != "text-detector-berturk" {
		t.Errorf("expected ID text-detector-berturk, got %q", manifest.ID)
	}
	if manifest.Runtime.Provider != "safetensors" {
		t.Errorf("expected provider safetensors, got %q", manifest.Runtime.Provider)
	}

	validator := registry.NewValidator(true)
	if err := validator.ValidateModelPackage(bestModelDir, manifest); err != nil {
		t.Fatalf("BERTurk model package validation failed: %v", err)
	}
}
