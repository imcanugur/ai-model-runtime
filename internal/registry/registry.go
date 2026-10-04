package registry

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/pipeline"
	"github.com/imcanugur/ai-model-runtime/internal/runtime"
)

// RuntimeFactory instantiates a domain.Runtime for a validated model package.
type RuntimeFactory func(manifest *ModelManifest, modelDir string) (domain.Runtime, error)

// AdaptRuntimeFactory converts a runtime.Factory to a registry.RuntimeFactory.
func AdaptRuntimeFactory(factory runtime.Factory) RuntimeFactory {
	return func(manifest *ModelManifest, modelDir string) (domain.Runtime, error) {
		entrypoint := filepath.Join(modelDir, manifest.Runtime.Entrypoint)
		spec := runtime.Spec{
			Provider:   manifest.Runtime.Provider,
			Profile:    manifest.Runtime.Profile,
			Entrypoint: entrypoint,
			ModelDir:   modelDir,
			Config: map[string]string{
				"task":         manifest.Metadata.Task,
				"architecture": manifest.Metadata.Architecture,
			},
		}
		return factory.Create(context.Background(), spec)
	}
}

// ModelRegistry defines the model discovery, loading, lifecycle, and acquisition contract.
type ModelRegistry interface {
	Discover(ctx context.Context, rootDir string) error
	Load(ctx context.Context, key domain.ModelKey) error
	Unload(ctx context.Context, key domain.ModelKey) error
	Acquire(key domain.ModelKey) (domain.ModelHandle, error)
	AcquireDefault(id domain.ModelID) (domain.ModelHandle, error)
	List() []domain.ModelDescriptor
}

type discoveredEntry struct {
	dir      string
	manifest *ModelManifest
}

// DefaultModelRegistry is the thread-safe standard implementation of ModelRegistry.
type DefaultModelRegistry struct {
	mu             sync.RWMutex
	validator      *Validator
	stageRegistry  pipeline.StageRegistry
	runtimeFactory RuntimeFactory

	instances  map[domain.ModelKey]*ModelInstance
	discovered map[domain.ModelKey]*discoveredEntry
}

// NewDefaultModelRegistry creates an initialized ModelRegistry.
func NewDefaultModelRegistry(stageRegistry pipeline.StageRegistry, rtFactory RuntimeFactory) *DefaultModelRegistry {
	if stageRegistry == nil {
		stageRegistry = pipeline.NewDefaultStageRegistry()
	}
	if rtFactory == nil {
		rtFactory = func(manifest *ModelManifest, modelDir string) (domain.Runtime, error) {
			return &noopRuntime{}, nil
		}
	}

	return &DefaultModelRegistry{
		validator:      NewValidator(true),
		stageRegistry:  stageRegistry,
		runtimeFactory: rtFactory,
		instances:      make(map[domain.ModelKey]*ModelInstance),
		discovered:     make(map[domain.ModelKey]*discoveredEntry),
	}
}

// Discover scans rootDir for model packages containing model.yaml.
func (r *DefaultModelRegistry) Discover(ctx context.Context, rootDir string) error {
	cleanRoot, err := filepath.Abs(rootDir)
	if err != nil {
		return fmt.Errorf("invalid root directory %q: %w", rootDir, err)
	}

	entries, err := os.ReadDir(cleanRoot)
	if err != nil {
		return fmt.Errorf("failed to read model root directory %q: %w", cleanRoot, err)
	}

	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			continue
		}
		isDir := entry.IsDir()
		if !isDir && (info.Mode()&os.ModeSymlink != 0) {
			targetInfo, statErr := os.Stat(filepath.Join(cleanRoot, entry.Name()))
			if statErr == nil && targetInfo.IsDir() {
				isDir = true
			}
		}
		if !isDir {
			continue
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		modelDir := filepath.Join(cleanRoot, entry.Name())
		manifestPath := filepath.Join(modelDir, "model.yaml")

		// Skip directories without model.yaml
		if _, err := os.Stat(manifestPath); err != nil {
			continue
		}

		manifest, parseErr := ParseManifestFile(manifestPath)
		if parseErr != nil {
			// Record broken manifest as FAILED instance so it is observable in List()
			fallbackKey := domain.NewModelKey(domain.ModelID(entry.Name()), "unknown")
			r.recordFailedModel(fallbackKey, entry.Name(), modelDir, fmt.Errorf("parse manifest: %w", parseErr))
			continue
		}

		valErr := r.validator.ValidateModelPackage(modelDir, manifest)
		key := domain.NewModelKey(domain.ModelID(manifest.ID), domain.ModelVersion(manifest.Version))

		if valErr != nil {
			r.recordFailedModel(key, manifest.Name, modelDir, valErr)
			continue
		}

		r.mu.Lock()
		if _, exists := r.discovered[key]; exists {
			r.mu.Unlock()
			return fmt.Errorf("%w: %s discovered multiple times", domain.ErrDuplicateModelKey, key)
		}

		meta := domain.ModelMetadata{
			ID:          domain.ModelID(manifest.ID),
			Name:        manifest.Name,
			Version:     domain.ModelVersion(manifest.Version),
			Task:        manifest.Metadata.Task,
			Languages:   manifest.Metadata.Languages,
			Description: manifest.Metadata.Description,
			Author:      manifest.Metadata.Author,
			Tags:        manifest.Metadata.Tags,
		}

		instance := NewModelInstance(key, meta, modelDir, manifest, nil, nil)
		r.instances[key] = instance
		r.discovered[key] = &discoveredEntry{dir: modelDir, manifest: manifest}
		r.mu.Unlock()
	}

	return nil
}

// Load builds the pipeline and initializes the runtime for a discovered model.
func (r *DefaultModelRegistry) Load(ctx context.Context, key domain.ModelKey) error {
	r.mu.Lock()
	instance, exists := r.instances[key]
	if !exists {
		r.mu.Unlock()
		return fmt.Errorf("%w: %s", domain.ErrModelNotFound, key)
	}

	state := instance.State()
	if state == domain.StateReady || state == domain.StateBusy {
		r.mu.Unlock()
		return fmt.Errorf("%w: %s", domain.ErrModelAlreadyLoaded, key)
	}

	disc, hasDisc := r.discovered[key]
	if !hasDisc || disc.manifest == nil {
		r.mu.Unlock()
		return fmt.Errorf("%w: missing discovered manifest for %s", domain.ErrInvalidManifest, key)
	}

	instance.SetState(domain.StateLoading)
	manifest := disc.manifest
	modelDir := disc.dir
	r.mu.Unlock()

	// Initialize runtime outside global lock
	rt, err := r.runtimeFactory(manifest, modelDir)
	if err != nil {
		instance.SetError(fmt.Errorf("initialize runtime: %w", err))
		return fmt.Errorf("failed to initialize runtime for model %s: %w", key, err)
	}

	// Build declarative pipeline
	pipe, err := pipeline.BuildPipeline(domain.ModelID(manifest.ID), domain.ModelVersion(manifest.Version), manifest.Pipeline.Stages, r.stageRegistry, rt)
	if err != nil {
		if rt != nil {
			_ = rt.Close(ctx)
		}
		instance.SetError(fmt.Errorf("build pipeline: %w", err))
		return fmt.Errorf("failed to build pipeline for model %s: %w", key, err)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	instance.Pipeline = pipe
	instance.Runtime = rt
	instance.SetState(domain.StateReady)
	return nil
}

// Unload drains active leases and terminates a model instance.
func (r *DefaultModelRegistry) Unload(ctx context.Context, key domain.ModelKey) error {
	r.mu.RLock()
	instance, exists := r.instances[key]
	r.mu.RUnlock()

	if !exists {
		return fmt.Errorf("%w: %s", domain.ErrModelNotFound, key)
	}

	return instance.WaitAndClose(ctx)
}

// Acquire returns a reference-counted handle to a specific ModelKey.
func (r *DefaultModelRegistry) Acquire(key domain.ModelKey) (domain.ModelHandle, error) {
	r.mu.RLock()
	instance, exists := r.instances[key]
	r.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("%w: %s", domain.ErrModelNotFound, key)
	}

	return instance.Acquire()
}

// AcquireDefault selects the highest semver READY version for a ModelID deterministically.
func (r *DefaultModelRegistry) AcquireDefault(id domain.ModelID) (domain.ModelHandle, error) {
	r.mu.RLock()
	var candidates []*ModelInstance
	var modelExists bool
	for _, inst := range r.instances {
		if inst.Metadata.ID == id {
			modelExists = true
			if inst.State() == domain.StateReady {
				candidates = append(candidates, inst)
			}
		}
	}
	r.mu.RUnlock()

	if !modelExists {
		return nil, fmt.Errorf("%w: %s", domain.ErrModelNotFound, id)
	}
	if len(candidates) == 0 {
		return nil, fmt.Errorf("%w: id %q has no ready versions", domain.ErrNoReadyVersion, id)
	}

	// Sort candidates by version descending
	sort.Slice(candidates, func(i, j int) bool {
		return compareVersions(candidates[i].Metadata.Version, candidates[j].Metadata.Version) > 0
	})

	return candidates[0].Acquire()
}

// List returns descriptors for all registered models.
func (r *DefaultModelRegistry) List() []domain.ModelDescriptor {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make([]domain.ModelDescriptor, 0, len(r.instances))
	for _, inst := range r.instances {
		result = append(result, inst.Descriptor())
	}

	// Sort deterministically by key
	sort.Slice(result, func(i, j int) bool {
		return result[i].Key < result[j].Key
	})
	return result
}

func (r *DefaultModelRegistry) recordFailedModel(key domain.ModelKey, name string, dir string, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	id, ver, _ := key.Split()
	meta := domain.ModelMetadata{
		ID:      id,
		Name:    name,
		Version: ver,
	}
	inst := NewModelInstance(key, meta, dir, nil, nil, nil)
	inst.SetError(err)
	r.instances[key] = inst
}

// compareVersions implements deterministic semver-aware comparison.
// Returns: 1 if v1 > v2, -1 if v1 < v2, 0 if v1 == v2.
func compareVersions(v1, v2 domain.ModelVersion) int {
	s1 := strings.TrimPrefix(string(v1), "v")
	s2 := strings.TrimPrefix(string(v2), "v")

	p1 := strings.Split(s1, ".")
	p2 := strings.Split(s2, ".")

	maxParts := len(p1)
	if len(p2) > maxParts {
		maxParts = len(p2)
	}

	for i := 0; i < maxParts; i++ {
		var n1, n2 int
		var err1, err2 error
		if i < len(p1) {
			n1, err1 = strconv.Atoi(p1[i])
		}
		if i < len(p2) {
			n2, err2 = strconv.Atoi(p2[i])
		}

		// If both are numbers, compare numerically
		if err1 == nil && err2 == nil {
			if n1 > n2 {
				return 1
			}
			if n1 < n2 {
				return -1
			}
			continue
		}

		// Fallback to string comparison for pre-release tags
		var str1, str2 string
		if i < len(p1) {
			str1 = p1[i]
		}
		if i < len(p2) {
			str2 = p2[i]
		}
		if str1 > str2 {
			return 1
		}
		if str1 < str2 {
			return -1
		}
	}
	return 0
}

// noopRuntime provides a safe placeholder runtime.
type noopRuntime struct{}

func (n *noopRuntime) Infer(ctx context.Context, inputs domain.TensorMap) (domain.TensorMap, error) {
	return domain.TensorMap{}, nil
}

func (n *noopRuntime) Close(ctx context.Context) error {
	return nil
}
