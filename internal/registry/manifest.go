package registry

import (
	"fmt"
	"os"

	"github.com/imcanugur/ai-model-runtime/internal/pipeline"
	"gopkg.in/yaml.v3"
)

// CurrentSupportedSchemaVersion is the required schema version for model manifests.
const CurrentSupportedSchemaVersion = "v1"

// ModelManifest defines the complete declarative specification of a model package.
type ModelManifest struct {
	SchemaVersion string            `yaml:"schema_version"`
	ID            string            `yaml:"id"`
	Name          string            `yaml:"name"`
	Version       string            `yaml:"version"`
	Metadata      ManifestMetadata  `yaml:"metadata"`
	Resources     ManifestResources `yaml:"resources"`
	Runtime       ManifestRuntime   `yaml:"runtime"`
	Pipeline      ManifestPipeline  `yaml:"pipeline"`
}

// ManifestMetadata provides discovery tags and domain categorization.
type ManifestMetadata struct {
	Task         string            `yaml:"task"`
	Languages    []string          `yaml:"languages"`
	Description  string            `yaml:"description"`
	Author       string            `yaml:"author"`
	Framework    string            `yaml:"framework,omitempty"`
	Architecture string            `yaml:"architecture,omitempty"`
	Tags         map[string]string `yaml:"tags"`
}

// ManifestResources defines operational resource constraints.
type ManifestResources struct {
	MaxConcurrency int `yaml:"max_concurrency"`
	TimeoutMs      int `yaml:"timeout_ms"`
	MinMemoryMB    int `yaml:"min_memory_mb"`
}

// ManifestRuntime defines the runtime engine requirement without deployment topology.
type ManifestRuntime struct {
	Provider   string `yaml:"provider"`   // e.g. "onnx", "sidecar_grpc", "mock"
	Profile    string `yaml:"profile"`    // e.g. "standard_cpu", "cuda_gpu"
	Entrypoint string `yaml:"entrypoint"` // Relative path inside model directory, e.g. "weights/model.onnx"
}

// ManifestPipeline defines the sequential pipeline stages.
type ManifestPipeline struct {
	Stages []pipeline.StageConfig `yaml:"stages"`
}

// ParseManifest parses raw YAML bytes into a ModelManifest struct.
func ParseManifest(data []byte) (*ModelManifest, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty manifest content")
	}

	var manifest ModelManifest
	if err := yaml.Unmarshal(data, &manifest); err != nil {
		return nil, fmt.Errorf("failed to parse YAML manifest: %w", err)
	}

	return &manifest, nil
}

// ParseManifestFile reads and parses a model.yaml file from disk.
func ParseManifestFile(filePath string) (*ModelManifest, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read manifest file: %w", err)
	}
	return ParseManifest(data)
}
