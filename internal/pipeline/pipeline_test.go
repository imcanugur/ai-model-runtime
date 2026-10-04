package pipeline_test

import (
	"context"
	"errors"
	"testing"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/pipeline"
)

// mockRuntime records inference calls and returns predefined output tensors.
type mockRuntime struct {
	called bool
	closed bool
}

func (m *mockRuntime) Infer(ctx context.Context, inputs domain.TensorMap) (domain.TensorMap, error) {
	m.called = true
	t, _ := domain.NewTensorFloat32([]int64{1, 2}, []float32{0.1, 0.9})
	return domain.TensorMap{"logits": t}, nil
}

func (m *mockRuntime) Close(ctx context.Context) error {
	m.closed = true
	return nil
}

func TestStageRegistry_RegisterAndCreate(t *testing.T) {
	r := pipeline.NewDefaultStageRegistry()

	// Builtin stages should exist
	if !r.Has("text_normalizer") {
		t.Error("expected builtin 'text_normalizer' to exist")
	}

	// Register custom stage
	err := r.Register("custom_stage", func(name string, config map[string]any, rt domain.Runtime) (pipeline.PipelineStage, error) {
		return &testStage{name: name}, nil
	})
	if err != nil {
		t.Fatalf("failed to register custom stage: %v", err)
	}

	// Duplicate registration must fail
	err = r.Register("custom_stage", func(name string, config map[string]any, rt domain.Runtime) (pipeline.PipelineStage, error) {
		return &testStage{name: name}, nil
	})
	if !errors.Is(err, domain.ErrDuplicateStageType) {
		t.Errorf("expected ErrDuplicateStageType, got %v", err)
	}

	// Create registered stage
	stage, err := r.Create("custom_stage", "step1", nil, nil)
	if err != nil {
		t.Fatalf("failed to create stage: %v", err)
	}
	if stage.Name() != "step1" {
		t.Errorf("expected stage name 'step1', got %s", stage.Name())
	}

	// Unknown stage type must fail
	_, err = r.Create("unknown_type", "step2", nil, nil)
	if !errors.Is(err, domain.ErrUnknownStageType) {
		t.Errorf("expected ErrUnknownStageType, got %v", err)
	}
}

func TestPipeline_ExecutionOrderAndContext(t *testing.T) {
	var executionOrder []string

	stage1 := &trackingStage{
		name: "stage1",
		onProcess: func(ctx *domain.PipelineContext) error {
			executionOrder = append(executionOrder, "stage1")
			ctx.Document.NormalizedText = "normalized:" + ctx.Input.Text
			return nil
		},
	}

	stage2 := &trackingStage{
		name: "stage2",
		onProcess: func(ctx *domain.PipelineContext) error {
			executionOrder = append(executionOrder, "stage2")
			ctx.Result.Label = "AI"
			ctx.Result.PrimaryScore = 95.5
			return nil
		},
	}

	pipe := pipeline.NewDefaultPipeline("test-model", "1.0.0", []pipeline.PipelineStage{stage1, stage2})

	ctx := context.Background()
	input := domain.RawInput{Text: "sample input"}

	res, err := pipe.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected pipeline error: %v", err)
	}

	if len(executionOrder) != 2 || executionOrder[0] != "stage1" || executionOrder[1] != "stage2" {
		t.Errorf("unexpected execution order: %v", executionOrder)
	}
	if res.ModelID != "test-model" || res.ModelVersion != "1.0.0" {
		t.Errorf("unexpected model identity: %s:%s", res.ModelID, res.ModelVersion)
	}
	if res.Label != "AI" || res.PrimaryScore != 95.5 {
		t.Errorf("unexpected result: %+v", res)
	}
}

func TestPipeline_StageErrorPropagation(t *testing.T) {
	expectedErr := errors.New("underlying failure")

	failingStage := &trackingStage{
		name: "bad_stage",
		onProcess: func(ctx *domain.PipelineContext) error {
			return expectedErr
		},
	}

	pipe := pipeline.NewDefaultPipeline("test-model", "1.0.0", []pipeline.PipelineStage{failingStage})

	_, err := pipe.Execute(context.Background(), domain.RawInput{Text: "test"})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, expectedErr) {
		t.Errorf("expected error wrapping %v, got %v", expectedErr, err)
	}
}

func TestPipeline_BuildPipelineFromManifestWithRuntime(t *testing.T) {
	stageConfigs := []pipeline.StageConfig{
		{Name: "norm", Type: "text_normalizer"},
		{Name: "chunk", Type: "sliding_window_chunker"},
		{Name: "infer", Type: "runtime_infer"},
	}

	rt := &mockRuntime{}
	stageReg := pipeline.NewDefaultStageRegistry()

	pipe, err := pipeline.BuildPipeline("sentiment", "1.0.0", stageConfigs, stageReg, rt)
	if err != nil {
		t.Fatalf("failed to build pipeline: %v", err)
	}

	res, err := pipe.Execute(context.Background(), domain.RawInput{Text: "hello"})
	if err != nil {
		t.Fatalf("pipeline execution failed: %v", err)
	}

	if !rt.called {
		t.Error("expected mock runtime to be called by runtime_infer stage")
	}
	if res.ModelID != "sentiment" {
		t.Errorf("expected modelID 'sentiment', got %s", res.ModelID)
	}
}

// Helper test stages
type testStage struct{ name string }

func (s *testStage) Name() string                              { return s.name }
func (s *testStage) Process(ctx *domain.PipelineContext) error { return nil }

type trackingStage struct {
	name      string
	onProcess func(ctx *domain.PipelineContext) error
}

func (s *trackingStage) Name() string                              { return s.name }
func (s *trackingStage) Process(ctx *domain.PipelineContext) error { return s.onProcess(ctx) }
