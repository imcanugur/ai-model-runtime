package domain

import (
	"context"
)

// PipelineContext is the strongly typed data carrier passing through pipeline stages.
// It avoids dynamic map[string]any casting across standard stage boundaries while
// allowing typed inspection and zero-allocation updates.
type PipelineContext struct {
	Ctx            context.Context
	Input          RawInput
	InferenceInput InferenceInput
	Document       TextDocument
	Tokens         TokenizedBatch
	Inputs         TensorMap // Tensors prepared for runtime execution
	Outputs        TensorMap // Raw tensors returned from runtime execution
	Result         PredictionResult
	Metadata       map[string]string
}

// NewPipelineContext creates an initialized PipelineContext.
func NewPipelineContext(ctx context.Context, input RawInput) *PipelineContext {
	if ctx == nil {
		ctx = context.Background()
	}
	infInput := input.ToInferenceInput()
	return &PipelineContext{
		Ctx:            ctx,
		Input:          input,
		InferenceInput: infInput,
		Inputs:         make(TensorMap),
		Outputs:        make(TensorMap),
		Metadata:       make(map[string]string),
		Result: PredictionResult{
			Metadata: make(map[string]string),
		},
	}
}

// NewPipelineContextWithInferenceInput creates an initialized PipelineContext from universal InferenceInput.
func NewPipelineContextWithInferenceInput(ctx context.Context, infInput InferenceInput) *PipelineContext {
	if ctx == nil {
		ctx = context.Background()
	}
	raw := infInput.ToRawInput()
	return &PipelineContext{
		Ctx:            ctx,
		Input:          raw,
		InferenceInput: infInput,
		Inputs:         make(TensorMap),
		Outputs:        make(TensorMap),
		Metadata:       make(map[string]string),
		Result: PredictionResult{
			Metadata: make(map[string]string),
		},
	}
}

type qaContextKeyType struct{}

var qaContextKey = qaContextKeyType{}

// WithQAContext injects QA RawInput into a context.Context.
func WithQAContext(ctx context.Context, input RawInput) context.Context {
	return context.WithValue(ctx, qaContextKey, input)
}

// GetQAContext extracts QA RawInput from a context.Context if present.
func GetQAContext(ctx context.Context) (RawInput, bool) {
	if ctx == nil {
		return RawInput{}, false
	}
	val, ok := ctx.Value(qaContextKey).(RawInput)
	return val, ok
}
