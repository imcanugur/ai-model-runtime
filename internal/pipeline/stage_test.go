package pipeline_test

import (
	"context"
	"errors"
	"testing"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/pipeline"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/mock"
)

func TestGenericNormalizerStage(t *testing.T) {
	stageReg := pipeline.NewDefaultStageRegistry()
	cfg := map[string]any{
		"strip_whitespace": true,
		"lowercase":        true,
		"remove_pattern":   `\[cite[^\]]*\]`,
	}

	stage, err := stageReg.Create("text_normalizer", "norm", cfg, nil)
	if err != nil {
		t.Fatalf("failed to create normalizer stage: %v", err)
	}

	pCtx := domain.NewPipelineContext(context.Background(), domain.RawInput{
		Text: "  Hello [cite:123]   WORLD! \n This is a TEST.  ",
	})

	if err := stage.Process(pCtx); err != nil {
		t.Fatalf("process failed: %v", err)
	}

	expected := "hello world! this is a test."
	if pCtx.Document.NormalizedText != expected {
		t.Errorf("expected %q, got %q", expected, pCtx.Document.NormalizedText)
	}
}

func TestGenericChunkerStage(t *testing.T) {
	stageReg := pipeline.NewDefaultStageRegistry()
	cfg := map[string]any{
		"max_tokens": 4,
		"stride":     0,
	}

	stage, err := stageReg.Create("sliding_window_chunker", "chunker", cfg, nil)
	if err != nil {
		t.Fatalf("failed to create chunker stage: %v", err)
	}

	pCtx := domain.NewPipelineContext(context.Background(), domain.RawInput{})
	// 9 words with max_tokens=4 should yield 3 chunks: [4 words, 4 words, 1 word]
	pCtx.Document.NormalizedText = "w1 w2 w3 w4 w5 w6 w7 w8 w9"

	if err := stage.Process(pCtx); err != nil {
		t.Fatalf("process failed: %v", err)
	}

	if pCtx.Tokens.BatchSize != 3 {
		t.Fatalf("expected 3 chunks, got %d", pCtx.Tokens.BatchSize)
	}
	if pCtx.Tokens.SequenceLength != 4 {
		t.Errorf("expected sequence length 4, got %d", pCtx.Tokens.SequenceLength)
	}

	inputIDs, exists := pCtx.Inputs["input_ids"]
	if !exists {
		t.Fatal("expected 'input_ids' tensor in pCtx.Inputs")
	}
	if len(inputIDs.Shape) != 2 || inputIDs.Shape[0] != 3 || inputIDs.Shape[1] != 4 {
		t.Errorf("expected input_ids shape [3, 4], got %v", inputIDs.Shape)
	}

	mask, exists := pCtx.Inputs["attention_mask"]
	if !exists {
		t.Fatal("expected 'attention_mask' tensor in pCtx.Inputs")
	}
	if len(mask.Shape) != 2 || mask.Shape[0] != 3 || mask.Shape[1] != 4 {
		t.Errorf("expected attention_mask shape [3, 4], got %v", mask.Shape)
	}

	// Last chunk has 1 token and 3 pads -> mask should be [1, 0, 0, 0] for last row
	lastRowMask := mask.DataI64[8:12]
	if lastRowMask[0] != 1 || lastRowMask[1] != 0 || lastRowMask[2] != 0 || lastRowMask[3] != 0 {
		t.Errorf("unexpected padding mask for last chunk: %v", lastRowMask)
	}
}

func TestRuntimeInferStage(t *testing.T) {
	rt := mock.NewMockRuntime()
	stageReg := pipeline.NewDefaultStageRegistry()

	stage, err := stageReg.Create("runtime_infer", "infer", nil, rt)
	if err != nil {
		t.Fatalf("failed to create infer stage: %v", err)
	}

	pCtx := domain.NewPipelineContext(context.Background(), domain.RawInput{})
	tIn, _ := domain.NewTensorInt64([]int64{1, 2}, []int64{10, 20})
	pCtx.Inputs["input_ids"] = tIn

	if err := stage.Process(pCtx); err != nil {
		t.Fatalf("process failed: %v", err)
	}

	if _, exists := pCtx.Outputs["logits"]; !exists {
		t.Fatal("expected 'logits' tensor in pCtx.Outputs")
	}
	if rt.InferCount() != 1 {
		t.Errorf("expected 1 infer call, got %d", rt.InferCount())
	}
}

func TestRuntimeInferStage_MissingRuntime(t *testing.T) {
	stageReg := pipeline.NewDefaultStageRegistry()
	stage, err := stageReg.Create("runtime_infer", "infer", nil, nil) // rt is nil
	if err != nil {
		t.Fatal(err)
	}

	pCtx := domain.NewPipelineContext(context.Background(), domain.RawInput{})
	err = stage.Process(pCtx)
	if err == nil {
		t.Fatal("expected error for missing runtime, got nil")
	}
	if !errors.Is(err, domain.ErrModelNotReady) {
		t.Errorf("expected ErrModelNotReady, got %v", err)
	}
}

func TestGenericCalibratorStage(t *testing.T) {
	stageReg := pipeline.NewDefaultStageRegistry()
	cfg := map[string]any{
		"threshold":      2.0,
		"positive_label": "AI",
		"negative_label": "Human",
	}

	stage, err := stageReg.Create("threshold_calibrator", "calibrator", cfg, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Case 1: Score above threshold -> AI
	pCtxPositive := domain.NewPipelineContext(context.Background(), domain.RawInput{})
	pCtxPositive.Result.PrimaryScore = 3.5

	if err := stage.Process(pCtxPositive); err != nil {
		t.Fatal(err)
	}
	if pCtxPositive.Result.Label != "AI" {
		t.Errorf("expected label 'AI', got %s", pCtxPositive.Result.Label)
	}
	if pCtxPositive.Result.Confidence <= 0.5 {
		t.Errorf("expected confidence > 0.5, got %f", pCtxPositive.Result.Confidence)
	}

	// Case 2: Score below threshold -> Human
	pCtxNegative := domain.NewPipelineContext(context.Background(), domain.RawInput{})
	pCtxNegative.Result.PrimaryScore = 0.5

	if err := stage.Process(pCtxNegative); err != nil {
		t.Fatal(err)
	}
	if pCtxNegative.Result.Label != "Human" {
		t.Errorf("expected label 'Human', got %s", pCtxNegative.Result.Label)
	}
	if pCtxNegative.Result.Confidence >= 0.5 {
		t.Errorf("expected confidence < 0.5, got %f", pCtxNegative.Result.Confidence)
	}
}

func TestQAAnswerExtractorStage(t *testing.T) {
	stageReg := pipeline.NewDefaultStageRegistry()
	stage, err := stageReg.Create("qa_answer_extractor", "qa_extractor", map[string]any{"max_answer_length": 10}, nil)
	if err != nil {
		t.Fatal(err)
	}

	pCtx := domain.NewPipelineContext(context.Background(), domain.RawInput{
		Context:  "Runtime is an AI inference platform written in Go.",
		Question: "What language?",
	})

	startLogits, _ := domain.NewTensorFloat32([]int64{1, 5}, []float32{-5.0, -2.0, -1.0, 0.0, 8.5})
	endLogits, _ := domain.NewTensorFloat32([]int64{1, 5}, []float32{-5.0, -2.0, -1.0, 0.0, 8.5})
	offsets, _ := domain.NewTensorInt64([]int64{5, 2}, []int64{
		0, 0,
		0, 7,
		36, 43,
		44, 46,
		47, 49,
	})
	mask, _ := domain.NewTensorInt64([]int64{5}, []int64{0, 1, 1, 1, 1})

	pCtx.Outputs["start_logits"] = startLogits
	pCtx.Outputs["end_logits"] = endLogits
	pCtx.Outputs["offset_mapping"] = offsets
	pCtx.Outputs["context_mask"] = mask

	if err := stage.Process(pCtx); err != nil {
		t.Fatalf("stage execution failed: %v", err)
	}

	if pCtx.Result.Answer != "Go" {
		t.Errorf("expected answer 'Go', got %q", pCtx.Result.Answer)
	}
	if pCtx.Result.Confidence <= 0.8 {
		t.Errorf("expected confidence > 0.8, got %f", pCtx.Result.Confidence)
	}
}
