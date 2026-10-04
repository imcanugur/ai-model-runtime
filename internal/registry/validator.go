package registry

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

// AllowedRuntimeProviders lists the known supported runtime execution providers.
var AllowedRuntimeProviders = map[string]bool{
	"onnx":         true,
	"safetensors":  true,
	"native":       true,
	"sidecar":      true,
	"sidecar_grpc": true,
	"mock":         true,
}

// Validator validates model manifests and artifact paths against security and correctness constraints.
type Validator struct {
	checkFileExistence bool
}

// NewValidator creates a manifest validator.
// If checkFileExistence is true, entrypoint and declared artifact files must exist on disk.
func NewValidator(checkFileExistence bool) *Validator {
	return &Validator{
		checkFileExistence: checkFileExistence,
	}
}

// ValidateManifest validates the manifest structure and values.
func (v *Validator) ValidateManifest(m *ModelManifest) error {
	if m == nil {
		return domain.NewValidationError("manifest", "manifest is nil", domain.ErrInvalidManifest)
	}

	// 1. Schema version
	if strings.TrimSpace(m.SchemaVersion) == "" {
		return domain.NewValidationError("schema_version", "schema_version is required", domain.ErrInvalidManifest)
	}
	if m.SchemaVersion != CurrentSupportedSchemaVersion {
		return domain.NewValidationError("schema_version", fmt.Sprintf("unsupported schema_version %q (expected %q)", m.SchemaVersion, CurrentSupportedSchemaVersion), domain.ErrInvalidManifest)
	}

	// 2. Model ID
	if err := domain.ValidateModelID(domain.ModelID(m.ID)); err != nil {
		return err
	}

	// 3. Name
	if strings.TrimSpace(m.Name) == "" {
		return domain.NewValidationError("name", "model name is required", domain.ErrInvalidManifest)
	}

	// 4. Version
	if err := domain.ValidateModelVersion(domain.ModelVersion(m.Version)); err != nil {
		return err
	}

	// 5. Runtime Provider
	provider := strings.TrimSpace(m.Runtime.Provider)
	if provider == "" {
		return domain.NewValidationError("runtime.provider", "runtime provider is required", domain.ErrUnsupportedRuntime)
	}
	if !AllowedRuntimeProviders[provider] {
		return domain.NewValidationError("runtime.provider", fmt.Sprintf("unsupported runtime provider %q", provider), domain.ErrUnsupportedRuntime)
	}

	// 6. Runtime Profile
	if strings.TrimSpace(m.Runtime.Profile) == "" {
		return domain.NewValidationError("runtime.profile", "runtime profile is required", domain.ErrInvalidManifest)
	}

	// 7. Entrypoint
	if strings.TrimSpace(m.Runtime.Entrypoint) == "" {
		return domain.NewValidationError("runtime.entrypoint", "runtime entrypoint is required", domain.ErrInvalidArtifactPath)
	}

	// 8. Pipeline stages
	if len(m.Pipeline.Stages) == 0 {
		return domain.NewValidationError("pipeline.stages", "pipeline must contain at least one stage", domain.ErrEmptyPipeline)
	}

	seenStages := make(map[string]bool)
	for i, stage := range m.Pipeline.Stages {
		stageName := strings.TrimSpace(stage.Name)
		if stageName == "" {
			return domain.NewValidationError(fmt.Sprintf("pipeline.stages[%d].name", i), "stage name cannot be empty", domain.ErrInvalidManifest)
		}
		if seenStages[stageName] {
			return domain.NewValidationError(fmt.Sprintf("pipeline.stages[%d].name", i), fmt.Sprintf("duplicate stage name %q", stageName), domain.ErrDuplicateStage)
		}
		seenStages[stageName] = true

		stageType := strings.TrimSpace(stage.Type)
		if stageType == "" {
			return domain.NewValidationError(fmt.Sprintf("pipeline.stages[%d].type", i), fmt.Sprintf("stage %q must define a non-empty type", stageName), domain.ErrInvalidManifest)
		}
	}

	return nil
}

// ValidateModelPackage validates both the manifest and the physical directory contents.
func (v *Validator) ValidateModelPackage(modelDir string, m *ModelManifest) error {
	if err := v.ValidateManifest(m); err != nil {
		return err
	}

	cleanDir, err := filepath.Abs(modelDir)
	if err != nil {
		return domain.NewValidationError("model_dir", fmt.Sprintf("failed to resolve absolute path of model directory: %s", err), domain.ErrInvalidArtifactPath)
	}

	// Validate entrypoint path
	entrypointPath, err := v.ValidateArtifactPath(cleanDir, m.Runtime.Entrypoint)
	if err != nil {
		return fmt.Errorf("invalid runtime entrypoint: %w", err)
	}

	if v.checkFileExistence {
		if _, err := os.Stat(entrypointPath); err != nil {
			if os.IsNotExist(err) {
				return domain.NewValidationError("runtime.entrypoint", fmt.Sprintf("entrypoint file does not exist: %s", m.Runtime.Entrypoint), domain.ErrArtifactNotFound)
			}
			return domain.NewValidationError("runtime.entrypoint", fmt.Sprintf("failed to stat entrypoint file: %s", err), domain.ErrInvalidArtifactPath)
		}
	}

	// Validate any artifact paths referenced in stage configurations
	for i, stage := range m.Pipeline.Stages {
		for key, val := range stage.Config {
			if strings.HasSuffix(key, "_path") || strings.HasSuffix(key, "_file") {
				if strVal, ok := val.(string); ok && strVal != "" {
					artifactPath, err := v.ValidateArtifactPath(cleanDir, strVal)
					if err != nil {
						return fmt.Errorf("invalid artifact path in stage %q field %q: %w", stage.Name, key, err)
					}
					if v.checkFileExistence {
						if _, err := os.Stat(artifactPath); err != nil {
							if os.IsNotExist(err) {
								return domain.NewValidationError(fmt.Sprintf("pipeline.stages[%d].config.%s", i, key), fmt.Sprintf("referenced artifact file does not exist: %s", strVal), domain.ErrArtifactNotFound)
							}
							return domain.NewValidationError(fmt.Sprintf("pipeline.stages[%d].config.%s", i, key), fmt.Sprintf("failed to stat artifact file: %s", err), domain.ErrInvalidArtifactPath)
						}
					}
				}
			}
		}
	}

	// Validate config.json consistency if present in model directory
	configPath := filepath.Join(cleanDir, "config.json")
	if _, err := os.Stat(configPath); err == nil {
		if err := v.validateModelConfig(configPath, m); err != nil {
			return err
		}
	}

	return nil
}

type modelConfigFile struct {
	Architectures []string `json:"architectures"`
	ModelType     string   `json:"model_type"`
}

func (v *Validator) validateModelConfig(configPath string, m *ModelManifest) error {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return domain.NewValidationError("config.json", fmt.Sprintf("failed to read config.json: %s", err), domain.ErrInvalidManifest)
	}

	var cfg modelConfigFile
	if err := json.Unmarshal(data, &cfg); err != nil {
		return domain.NewValidationError("config.json", fmt.Sprintf("invalid JSON in config.json: %s", err), domain.ErrInvalidManifest)
	}

	if len(cfg.Architectures) > 0 {
		configArch := cfg.Architectures[0]

		// If manifest defines architecture, verify strict match with config.json descriptor
		if m.Metadata.Architecture != "" && !strings.EqualFold(m.Metadata.Architecture, configArch) {
			return domain.NewValidationError("metadata.architecture",
				fmt.Sprintf("manifest architecture %q does not match config.json architecture %q", m.Metadata.Architecture, configArch),
				domain.ErrInvalidManifest)
		}
	}

	return nil
}

// ValidateArtifactPath strictly verifies that relativePath resides entirely within modelDir
// without directory traversal (e.g. "../"), absolute paths, or symlink escapes.
func (v *Validator) ValidateArtifactPath(modelDir string, relativePath string) (string, error) {
	rel := strings.TrimSpace(relativePath)
	if rel == "" {
		return "", domain.NewValidationError("path", "artifact path cannot be empty", domain.ErrInvalidArtifactPath)
	}

	// Reject absolute paths
	if filepath.IsAbs(rel) || strings.HasPrefix(rel, "/") || strings.HasPrefix(rel, "\\") {
		return "", domain.NewValidationError("path", fmt.Sprintf("absolute paths are forbidden: %q", rel), domain.ErrPathTraversal)
	}

	// Reject Windows volume paths like C:
	if len(rel) >= 2 && rel[1] == ':' {
		return "", domain.NewValidationError("path", fmt.Sprintf("volume paths are forbidden: %q", rel), domain.ErrPathTraversal)
	}

	// Clean and combine
	cleanedRel := filepath.Clean(rel)
	if strings.HasPrefix(cleanedRel, "..") || strings.Contains(cleanedRel, "/..") || strings.Contains(cleanedRel, "\\..") {
		return "", domain.NewValidationError("path", fmt.Sprintf("path traversal detected: %q", rel), domain.ErrPathTraversal)
	}

	absModelDir, err := filepath.Abs(modelDir)
	if err != nil {
		return "", domain.NewValidationError("model_dir", fmt.Sprintf("invalid model directory: %s", err), domain.ErrInvalidArtifactPath)
	}

	targetPath := filepath.Join(absModelDir, cleanedRel)

	// Ensure relative calculation stays inside modelDir
	relCheck, err := filepath.Rel(absModelDir, targetPath)
	if err != nil || strings.HasPrefix(relCheck, "..") || relCheck == ".." {
		return "", domain.NewValidationError("path", fmt.Sprintf("resolved path escapes model directory: %q", rel), domain.ErrPathTraversal)
	}

	// If target file exists, evaluate symlinks to guarantee no escape
	if fi, err := os.Lstat(targetPath); err == nil {
		if fi.Mode()&os.ModeSymlink != 0 {
			resolvedTarget, err := filepath.EvalSymlinks(targetPath)
			if err != nil {
				return "", domain.NewValidationError("path", fmt.Sprintf("failed to evaluate symlink: %s", err), domain.ErrInvalidArtifactPath)
			}
			resolvedRel, err := filepath.Rel(absModelDir, resolvedTarget)
			if err != nil || strings.HasPrefix(resolvedRel, "..") || resolvedRel == ".." {
				return "", domain.NewValidationError("path", fmt.Sprintf("symlink %q escapes model directory to %q", rel, resolvedTarget), domain.ErrPathTraversal)
			}
		}
	}

	return targetPath, nil
}
