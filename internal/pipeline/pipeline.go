package pipeline

import (
	"context"
	"fmt"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

// StageConfig defines a single declarative stage configuration parsed from model.yaml.
type StageConfig struct {
	Name   string         `yaml:"name"`
	Type   string         `yaml:"type"`
	Config map[string]any `yaml:"config"`
}

// DefaultPipeline executes a list of sequential stages on a PipelineContext.
type DefaultPipeline struct {
	modelID      domain.ModelID
	modelVersion domain.ModelVersion
	stages       []PipelineStage
}

// NewDefaultPipeline constructs a DefaultPipeline instance.
func NewDefaultPipeline(id domain.ModelID, version domain.ModelVersion, stages []PipelineStage) *DefaultPipeline {
	return &DefaultPipeline{
		modelID:      id,
		modelVersion: version,
		stages:       stages,
	}
}

// Stages returns the list of stages configured in this pipeline.
func (p *DefaultPipeline) Stages() []PipelineStage {
	return p.stages
}

// Execute orchestrates sequential stage execution with context propagation and latency recording.
func (p *DefaultPipeline) Execute(ctx context.Context, input domain.RawInput) (*domain.PredictionResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	start := time.Now()
	pCtx := domain.NewPipelineContext(ctx, input)
	pCtx.Result.ModelID = p.modelID
	pCtx.Result.ModelVersion = p.modelVersion
	pCtx.Result.Model.ID = p.modelID
	pCtx.Result.Model.Version = p.modelVersion

	stageDurations := make(map[string]int64)
	for _, stage := range p.stages {
		// Respect context deadlines and cancellations before running each stage
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("pipeline cancelled before stage %q: %w", stage.Name(), err)
		}

		stageStart := time.Now()
		if err := stage.Process(pCtx); err != nil {
			return nil, fmt.Errorf("pipeline stage %q failed: %w", stage.Name(), err)
		}
		stageDurations[stage.Name()] = time.Since(stageStart).Milliseconds()
	}

	totalMs := time.Since(start).Milliseconds()
	pCtx.Result.ExecutionTimeMs = totalMs
	pCtx.Result.Execution.DurationMs = totalMs
	pCtx.Result.Execution.StageDurationsMs = stageDurations
	pCtx.Result.Sync()
	return &pCtx.Result, nil
}

// BuildPipeline constructs a validated domain.Pipeline from stage configurations and StageRegistry.
func BuildPipeline(id domain.ModelID, version domain.ModelVersion, stageConfigs []StageConfig, stageRegistry StageRegistry, rt domain.Runtime) (domain.Pipeline, error) {
	if stageRegistry == nil {
		return nil, domain.NewValidationError("stage_registry", "stage registry cannot be nil", domain.ErrInvalidManifest)
	}

	stages := make([]PipelineStage, 0, len(stageConfigs))

	for _, stageCfg := range stageConfigs {
		stage, err := stageRegistry.Create(stageCfg.Type, stageCfg.Name, stageCfg.Config, rt)
		if err != nil {
			return nil, fmt.Errorf("failed to create stage %q (type: %s): %w", stageCfg.Name, stageCfg.Type, err)
		}
		stages = append(stages, stage)
	}

	return NewDefaultPipeline(id, version, stages), nil
}
