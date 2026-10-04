package pipeline

import (
	"fmt"
	"hash/fnv"
	"math"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

// PipelineStage represents an individual sequential processing step within a pipeline.
type PipelineStage interface {
	Name() string
	Process(ctx *domain.PipelineContext) error
}

// StageFactory instantiates a configured PipelineStage.
// The runtime is passed so that inference-related stages can bind to the model's runtime boundary.
type StageFactory func(name string, config map[string]any, rt domain.Runtime) (PipelineStage, error)

// StageRegistry manages the set of available generic pipeline stage types.
type StageRegistry interface {
	Register(stageType string, factory StageFactory) error
	Create(stageType string, name string, config map[string]any, rt domain.Runtime) (PipelineStage, error)
	Has(stageType string) bool
}

// DefaultStageRegistry is a thread-safe implementation of StageRegistry.
type DefaultStageRegistry struct {
	mu        sync.RWMutex
	factories map[string]StageFactory
}

// NewDefaultStageRegistry creates an initialized StageRegistry with standard generic stage types.
func NewDefaultStageRegistry() *DefaultStageRegistry {
	r := &DefaultStageRegistry{
		factories: make(map[string]StageFactory),
	}
	r.registerBuiltins()
	return r
}

// Register adds a new stage factory. Returns ErrDuplicateStageType if already registered.
func (r *DefaultStageRegistry) Register(stageType string, factory StageFactory) error {
	if stageType == "" {
		return domain.NewValidationError("stage_type", "stage type cannot be empty", domain.ErrInvalidManifest)
	}
	if factory == nil {
		return domain.NewValidationError("stage_factory", "stage factory cannot be nil", domain.ErrInvalidManifest)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.factories[stageType]; exists {
		return fmt.Errorf("%w: stage type %q is already registered", domain.ErrDuplicateStageType, stageType)
	}

	r.factories[stageType] = factory
	return nil
}

// Create instantiates a stage by its registered type. Returns ErrUnknownStageType if not found.
func (r *DefaultStageRegistry) Create(stageType string, name string, config map[string]any, rt domain.Runtime) (PipelineStage, error) {
	r.mu.RLock()
	factory, exists := r.factories[stageType]
	r.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("%w: no factory registered for stage type %q", domain.ErrUnknownStageType, stageType)
	}

	return factory(name, config, rt)
}

// Has checks if a stage type is registered.
func (r *DefaultStageRegistry) Has(stageType string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, exists := r.factories[stageType]
	return exists
}

// registerBuiltins registers standard, generic pipeline stages.
func (r *DefaultStageRegistry) registerBuiltins() {
	// 1. Generic text normalizer stage
	_ = r.Register("text_normalizer", func(name string, config map[string]any, rt domain.Runtime) (PipelineStage, error) {
		stripWhitespace := true
		if val, ok := config["strip_whitespace"].(bool); ok {
			stripWhitespace = val
		}
		lowercase := false
		if val, ok := config["lowercase"].(bool); ok {
			lowercase = val
		}
		var regex *regexp.Regexp
		if pat, ok := config["remove_pattern"].(string); ok && pat != "" {
			compiled, err := regexp.Compile(pat)
			if err != nil {
				return nil, fmt.Errorf("invalid remove_pattern in stage %q: %w", name, err)
			}
			regex = compiled
		}
		return &GenericNormalizerStage{
			name:            name,
			stripWhitespace: stripWhitespace,
			lowercase:       lowercase,
			pattern:         regex,
		}, nil
	})

	// 2. Generic sliding window chunker stage
	_ = r.Register("sliding_window_chunker", func(name string, config map[string]any, rt domain.Runtime) (PipelineStage, error) {
		maxTokens := 512
		if val, ok := config["max_tokens"].(int); ok && val > 0 {
			maxTokens = val
		}
		stride := 0
		if val, ok := config["stride"].(int); ok && val >= 0 {
			stride = val
		}
		return &GenericChunkerStage{
			name:      name,
			maxTokens: maxTokens,
			stride:    stride,
		}, nil
	})

	// 3. Generic runtime infer stage (bridges pipeline context to domain.Runtime)
	_ = r.Register("runtime_infer", func(name string, config map[string]any, rt domain.Runtime) (PipelineStage, error) {
		return &RuntimeInferStage{name: name, runtime: rt}, nil
	})

	// 4. Generic token weighted logits stage
	_ = r.Register("token_weighted_logits", func(name string, config map[string]any, rt domain.Runtime) (PipelineStage, error) {
		humanIdx := 0
		if val, ok := config["human_class_index"].(int); ok {
			humanIdx = val
		}
		aiIdx := 1
		if val, ok := config["ai_class_index"].(int); ok {
			aiIdx = val
		}
		return &GenericLogitReductionStage{
			name:     name,
			humanIdx: humanIdx,
			aiIdx:    aiIdx,
		}, nil
	})

	// 5. Generic threshold calibrator stage
	_ = r.Register("threshold_calibrator", func(name string, config map[string]any, rt domain.Runtime) (PipelineStage, error) {
		threshold := 0.5
		if val, ok := config["threshold"].(float64); ok {
			threshold = val
		}
		posLabel := "POSITIVE"
		if val, ok := config["positive_label"].(string); ok && val != "" {
			posLabel = val
		}
		negLabel := "NEGATIVE"
		if val, ok := config["negative_label"].(string); ok && val != "" {
			negLabel = val
		}
		return &GenericCalibratorStage{
			name:          name,
			threshold:     threshold,
			positiveLabel: posLabel,
			negativeLabel: negLabel,
		}, nil
	})

	// 6. Generic QA Answer Extractor stage
	_ = r.Register("qa_answer_extractor", func(name string, config map[string]any, rt domain.Runtime) (PipelineStage, error) {
		maxAnswerLength := 30
		if val, ok := config["max_answer_length"].(int); ok && val > 0 {
			maxAnswerLength = val
		}
		return &QAAnswerExtractorStage{
			name:            name,
			maxAnswerLength: maxAnswerLength,
		}, nil
	})

	// 7. Generic Embedding Extractor stage
	_ = r.Register("embedding_extractor", func(name string, config map[string]any, rt domain.Runtime) (PipelineStage, error) {
		normalize := true
		if val, ok := config["normalize"].(bool); ok {
			normalize = val
		}
		poolingMode := "mean"
		if val, ok := config["pooling_mode"].(string); ok && val != "" {
			poolingMode = strings.ToLower(val)
		} else if cls, ok := config["pooling_mode_cls_token"].(bool); ok && cls {
			poolingMode = "cls"
		}
		targetTensor := ""
		if val, ok := config["target_tensor"].(string); ok {
			targetTensor = val
		}
		return &EmbeddingExtractorStage{
			name:         name,
			normalize:    normalize,
			poolingMode:  poolingMode,
			targetTensor: targetTensor,
		}, nil
	})

	// 8. Generic Generation Extractor stage
	genStageFactory := func(name string, config map[string]any, rt domain.Runtime) (PipelineStage, error) {
		return &GenerationExtractorStage{
			name: name,
		}, nil
	}
	_ = r.Register("generation_extractor", genStageFactory)
	_ = r.Register("text_generation_extractor", genStageFactory)
}

// --- Built-in Generic Stage Implementations ---

// GenericNormalizerStage normalizes text via whitespace trimming, casing, and pattern replacement.
type GenericNormalizerStage struct {
	name            string
	stripWhitespace bool
	lowercase       bool
	pattern         *regexp.Regexp
}

func (s *GenericNormalizerStage) Name() string { return s.name }
func (s *GenericNormalizerStage) Process(ctx *domain.PipelineContext) error {
	text := ctx.Input.Text
	if text == "" {
		text = ctx.InferenceInput.GetText()
	}
	if text == "" && ctx.Input.Context != "" {
		text = ctx.Input.Context
	}
	if text == "" && ctx.InferenceInput.GetContext() != "" {
		text = ctx.InferenceInput.GetContext()
	}
	ctx.Document.RawText = text

	normalizeStr := func(in string) string {
		out := in
		if s.pattern != nil {
			out = s.pattern.ReplaceAllString(out, " ")
		}
		if s.stripWhitespace {
			out = strings.Join(strings.Fields(out), " ")
		}
		if s.lowercase {
			out = strings.ToLower(out)
		}
		return out
	}

	ctx.Document.NormalizedText = normalizeStr(text)
	if ctx.Input.Context != "" {
		ctx.Input.Context = normalizeStr(ctx.Input.Context)
	} else if ctx.InferenceInput.GetContext() != "" {
		ctx.Input.Context = normalizeStr(ctx.InferenceInput.GetContext())
	}
	if ctx.Input.Question != "" {
		ctx.Input.Question = normalizeStr(ctx.Input.Question)
	} else if ctx.InferenceInput.GetQuestion() != "" {
		ctx.Input.Question = normalizeStr(ctx.InferenceInput.GetQuestion())
	}
	if ctx.Input.Text != "" {
		ctx.Input.Text = ctx.Document.NormalizedText
	}
	return nil
}

// GenericChunkerStage tokenizes normalized text into windows and builds input tensors.
type GenericChunkerStage struct {
	name      string
	maxTokens int
	stride    int
}

func (s *GenericChunkerStage) Name() string { return s.name }
func (s *GenericChunkerStage) Process(ctx *domain.PipelineContext) error {
	words := strings.Fields(ctx.Document.NormalizedText)
	if len(words) == 0 {
		words = []string{""}
	}

	// Calculate deterministic token IDs per word
	allTokens := make([]int64, len(words))
	for i, w := range words {
		h := fnv.New32a()
		_, _ = h.Write([]byte(w))
		allTokens[i] = int64(h.Sum32()%30000) + 100
	}

	// Chunk tokens into slices of size maxTokens
	chunkSize := s.maxTokens
	step := chunkSize
	if s.stride > 0 {
		step = s.stride
	}

	var chunks [][]int64
	for i := 0; i < len(allTokens); i += step {
		end := i + chunkSize
		if end > len(allTokens) {
			end = len(allTokens)
		}
		chunks = append(chunks, allTokens[i:end])
		if end == len(allTokens) {
			break
		}
	}
	if len(chunks) == 0 {
		chunks = [][]int64{{0}}
	}

	batchSize := len(chunks)
	seqLen := s.maxTokens

	flatIDs := make([]int64, batchSize*seqLen)
	flatMask := make([]int64, batchSize*seqLen)
	counts := make([]int, batchSize)

	for b, c := range chunks {
		counts[b] = len(c)
		for j := 0; j < seqLen; j++ {
			idx := b*seqLen + j
			if j < len(c) {
				flatIDs[idx] = c[j]
				flatMask[idx] = 1
			} else {
				flatIDs[idx] = 0 // Pad token
				flatMask[idx] = 0
			}
		}
	}

	ctx.Tokens = domain.TokenizedBatch{
		InputIDs:       flatIDs,
		AttentionMask:  flatMask,
		TokenCounts:    counts,
		BatchSize:      batchSize,
		SequenceLength: seqLen,
	}

	// Pack tensors into ctx.Inputs
	inputIDsTensor, err := domain.NewTensorInt64([]int64{int64(batchSize), int64(seqLen)}, flatIDs)
	if err != nil {
		return fmt.Errorf("failed to build input_ids tensor: %w", err)
	}

	maskTensor, err := domain.NewTensorInt64([]int64{int64(batchSize), int64(seqLen)}, flatMask)
	if err != nil {
		return fmt.Errorf("failed to build attention_mask tensor: %w", err)
	}

	ctx.Inputs["input_ids"] = inputIDsTensor
	ctx.Inputs["attention_mask"] = maskTensor
	return nil
}

// RuntimeInferStage passes prepared tensors to the bound Runtime and collects output tensors.
type RuntimeInferStage struct {
	name    string
	runtime domain.Runtime
}

func (s *RuntimeInferStage) Name() string { return s.name }
func (s *RuntimeInferStage) Process(ctx *domain.PipelineContext) error {
	if s.runtime == nil {
		return domain.NewValidationError("runtime", "runtime is not bound to infer stage", domain.ErrModelNotReady)
	}

	if err := ctx.Inputs.Validate(); err != nil {
		return fmt.Errorf("invalid input tensors: %w", err)
	}

	inferCtx := domain.WithQAContext(ctx.Ctx, ctx.Input)
	outputs, err := s.runtime.Infer(inferCtx, ctx.Inputs)
	if err != nil {
		return fmt.Errorf("runtime infer failure: %w", err)
	}

	if err := outputs.Validate(); err != nil {
		return fmt.Errorf("invalid output tensors from runtime: %w", err)
	}

	ctx.Outputs = outputs

	if textTensor, exists := outputs["generated_text"]; exists && textTensor.Type == domain.DataTypeInt32 {
		bytes := make([]byte, len(textTensor.DataI32))
		for i, b := range textTensor.DataI32 {
			bytes[i] = byte(b)
		}
		str := string(bytes)
		ctx.Result.Output.Value = str
		ctx.Result.Answer = str
		ctx.Result.Sync()
	}
	if tokenTensor, ok := outputs["generated_tokens"]; ok && tokenTensor.Type == domain.DataTypeInt64 {
		if ctx.Result.Metadata == nil {
			ctx.Result.Metadata = make(map[string]string)
		}
		ctx.Result.Metadata["generated_tokens_count"] = strconv.Itoa(len(tokenTensor.DataI64))
	}

	return nil
}

// GenericLogitReductionStage aggregates multi-chunk logits into a primary score and per-chunk details.
type GenericLogitReductionStage struct {
	name     string
	humanIdx int
	aiIdx    int
}

func (s *GenericLogitReductionStage) Name() string { return s.name }
func (s *GenericLogitReductionStage) Process(ctx *domain.PipelineContext) error {
	logitsTensor, exists := ctx.Outputs["logits"]
	if !exists || logitsTensor.Type != domain.DataTypeFloat32 || len(logitsTensor.Shape) != 2 {
		return nil
	}

	batchSize := int(logitsTensor.Shape[0])
	numClasses := int(logitsTensor.Shape[1])
	if numClasses <= s.humanIdx || numClasses <= s.aiIdx {
		return fmt.Errorf("logits numClasses (%d) insufficient for class indices (%d, %d)", numClasses, s.humanIdx, s.aiIdx)
	}

	var totalScore float64
	chunks := make([]domain.ChunkDetail, batchSize)

	for b := 0; b < batchSize; b++ {
		humanLogit := float64(logitsTensor.DataF32[b*numClasses+s.humanIdx])
		aiLogit := float64(logitsTensor.DataF32[b*numClasses+s.aiIdx])
		chunkDiff := aiLogit - humanLogit

		tokenCount := 0
		if b < len(ctx.Tokens.TokenCounts) {
			tokenCount = ctx.Tokens.TokenCounts[b]
		}

		chunks[b] = domain.ChunkDetail{
			ID:         b + 1,
			Text:       fmt.Sprintf("Chunk %d", b+1),
			Score:      math.Round(chunkDiff*100) / 100,
			TokenCount: tokenCount,
		}
		totalScore += chunkDiff
	}

	meanScore := totalScore / float64(batchSize)
	ctx.Result.PrimaryScore = math.Round(meanScore*100) / 100
	ctx.Result.Chunks = chunks
	return nil
}

// GenericCalibratorStage applies decision thresholds to set the label and confidence.
type GenericCalibratorStage struct {
	name          string
	threshold     float64
	positiveLabel string
	negativeLabel string
}

func (s *GenericCalibratorStage) Name() string { return s.name }
func (s *GenericCalibratorStage) Process(ctx *domain.PipelineContext) error {
	score := ctx.Result.PrimaryScore

	if score >= s.threshold {
		ctx.Result.Label = s.positiveLabel
		// Sigmoid-like bounded confidence
		diff := score - s.threshold
		ctx.Result.Confidence = math.Min(0.99, 0.50+0.49*(1.0-math.Exp(-diff/2.0)))
	} else {
		ctx.Result.Label = s.negativeLabel
		diff := s.threshold - score
		ctx.Result.Confidence = math.Max(0.01, 0.50-0.49*(1.0-math.Exp(-diff/2.0)))
	}
	ctx.Result.Confidence = math.Round(ctx.Result.Confidence*1000) / 1000
	ctx.Result.Output.Value = map[string]any{
		"label":      ctx.Result.Label,
		"score":      ctx.Result.PrimaryScore,
		"confidence": ctx.Result.Confidence,
	}
	ctx.Result.Sync()
	return nil
}

// QAAnswerExtractorStage extracts answer text and span probabilities from start_logits and end_logits.
type QAAnswerExtractorStage struct {
	name            string
	maxAnswerLength int
}

func (s *QAAnswerExtractorStage) Name() string { return s.name }

func softmaxFloats(logits []float32) []float64 {
	if len(logits) == 0 {
		return nil
	}
	maxVal := float64(-math.MaxFloat64)
	for _, v := range logits {
		if float64(v) > maxVal {
			maxVal = float64(v)
		}
	}
	probs := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		p := math.Exp(float64(v) - maxVal)
		probs[i] = p
		sum += p
	}
	if sum > 0 {
		for i := range probs {
			probs[i] /= sum
		}
	}
	return probs
}

func (s *QAAnswerExtractorStage) Process(ctx *domain.PipelineContext) error {
	startTensor, hasStart := ctx.Outputs["start_logits"]
	endTensor, hasEnd := ctx.Outputs["end_logits"]
	if !hasStart || !hasEnd {
		return fmt.Errorf("qa_answer_extractor: missing start_logits or end_logits in runtime outputs")
	}

	seqLen := len(startTensor.DataF32)
	if seqLen == 0 || len(endTensor.DataF32) != seqLen {
		return fmt.Errorf("qa_answer_extractor: start_logits and end_logits length mismatch or empty (%d vs %d)", seqLen, len(endTensor.DataF32))
	}

	startProbs := softmaxFloats(startTensor.DataF32)
	endProbs := softmaxFloats(endTensor.DataF32)

	contextMaskTensor, hasMask := ctx.Outputs["context_mask"]

	maxLen := s.maxAnswerLength
	if maxLen <= 0 {
		maxLen = 30
	}

	bestScore := float64(-math.MaxFloat64)
	bestStart, bestEnd := 0, 0

	for i := 0; i < seqLen; i++ {
		if hasMask && i < len(contextMaskTensor.DataI64) && contextMaskTensor.DataI64[i] == 0 {
			continue // Skip tokens outside context
		}
		startVal := float64(startTensor.DataF32[i])
		for j := i; j < seqLen && (j-i) < maxLen; j++ {
			if hasMask && j < len(contextMaskTensor.DataI64) && contextMaskTensor.DataI64[j] == 0 {
				continue
			}
			score := startVal + float64(endTensor.DataF32[j])
			if score > bestScore {
				bestScore = score
				bestStart = i
				bestEnd = j
			}
		}
	}

	// Calculate confidence bounded in [0.0, 1.0]
	var confidence float64
	if len(startProbs) > bestStart && len(endProbs) > bestEnd {
		confidence = startProbs[bestStart] * endProbs[bestEnd]
	}
	if confidence > 1.0 {
		confidence = 1.0
	} else if confidence < 0.0 {
		confidence = 0.0
	}
	confidence = math.Round(confidence*1000) / 1000

	// Extract answer text from offsets and context
	contextText := ctx.Input.Context
	if contextText == "" {
		contextText = ctx.Input.Text
	}

	answer := ""
	offsetTensor, hasOffsets := ctx.Outputs["offset_mapping"]
	if hasOffsets && len(offsetTensor.DataI64) >= (bestEnd+1)*2 {
		startChar := int(offsetTensor.DataI64[bestStart*2])
		endChar := int(offsetTensor.DataI64[bestEnd*2+1])
		if startChar >= 0 && endChar <= len(contextText) && startChar <= endChar {
			answer = strings.TrimSpace(contextText[startChar:endChar])
		}
	}

	if answer == "" && contextText != "" {
		words := strings.Fields(contextText)
		if len(words) > 0 {
			answer = words[0]
		}
	}

	ctx.Result.Answer = answer
	ctx.Result.Label = answer
	ctx.Result.Confidence = confidence
	ctx.Result.PrimaryScore = confidence
	ctx.Result.Output.Value = map[string]any{
		"answer":     answer,
		"confidence": confidence,
	}
	ctx.Result.Sync()
	return nil
}

// EmbeddingExtractorStage extracts dense representation vectors from runtime tensor outputs.
type EmbeddingExtractorStage struct {
	name         string
	normalize    bool
	poolingMode  string
	targetTensor string
}

func (s *EmbeddingExtractorStage) Name() string { return s.name }

func (s *EmbeddingExtractorStage) Process(ctx *domain.PipelineContext) error {
	var targetTensor *domain.Tensor
	// If explicit target tensor configured, check it first
	if s.targetTensor != "" {
		if t, ok := ctx.Outputs[s.targetTensor]; ok {
			targetTensor = &t
		}
	}

	if targetTensor == nil {
		var candidates []string
		if s.poolingMode == "cls" {
			candidates = []string{"pooler_output", "embeddings", "embedding", "last_hidden_state", "sentence_embedding", "logits", "output"}
		} else {
			// "mean" or default prefers last_hidden_state for token-level masked pooling
			candidates = []string{"last_hidden_state", "embeddings", "embedding", "pooler_output", "sentence_embedding", "logits", "output"}
		}

		for _, key := range candidates {
			if t, ok := ctx.Outputs[key]; ok {
				targetTensor = &t
				break
			}
		}
	}

	if targetTensor == nil {
		for _, t := range ctx.Outputs {
			targetTensor = &t
			break
		}
	}

	if targetTensor == nil || len(targetTensor.DataF32) == 0 {
		return fmt.Errorf("embedding_extractor: no float32 tensor found in runtime outputs")
	}

	// Check for attention_mask in Outputs or Inputs for attention-weighted mean pooling
	var maskData []int64
	if maskTensor, ok := ctx.Outputs["attention_mask"]; ok && len(maskTensor.DataI64) > 0 {
		maskData = maskTensor.DataI64
	} else if maskTensor, ok := ctx.Inputs["attention_mask"]; ok && len(maskTensor.DataI64) > 0 {
		maskData = maskTensor.DataI64
	}

	shape := targetTensor.Shape
	var vectors [][]float32
	var dim int

	if len(shape) == 1 {
		dim = int(shape[0])
		vec := make([]float32, dim)
		copy(vec, targetTensor.DataF32)
		if s.normalize {
			normalizeVector(vec)
		}
		vectors = append(vectors, vec)
	} else if len(shape) == 2 {
		batchSize := int(shape[0])
		dim = int(shape[1])
		for b := 0; b < batchSize; b++ {
			vec := make([]float32, dim)
			copy(vec, targetTensor.DataF32[b*dim:(b+1)*dim])
			if s.normalize {
				normalizeVector(vec)
			}
			vectors = append(vectors, vec)
		}
	} else if len(shape) == 3 {
		// [batch_size, seq_len, dim] -> pooled over sequence
		batchSize := int(shape[0])
		seqLen := int(shape[1])
		dim = int(shape[2])

		for b := 0; b < batchSize; b++ {
			vec := make([]float32, dim)
			validCount := 0

			for i := 0; i < seqLen; i++ {
				maskVal := int64(1)
				maskIdx := b*seqLen + i
				if maskIdx < len(maskData) {
					maskVal = maskData[maskIdx]
				}

				if maskVal > 0 {
					validCount++
					offset := b*seqLen*dim + i*dim
					for d := 0; d < dim; d++ {
						vec[d] += targetTensor.DataF32[offset+d]
					}
				}
			}

			if validCount > 0 {
				denom := float32(validCount)
				for d := 0; d < dim; d++ {
					vec[d] /= denom
				}
			} else if seqLen > 0 {
				denom := float32(seqLen)
				for d := 0; d < dim; d++ {
					vec[d] /= denom
				}
			}

			if s.normalize {
				normalizeVector(vec)
			}
			vectors = append(vectors, vec)
		}
	} else {
		dim = len(targetTensor.DataF32)
		vec := make([]float32, dim)
		copy(vec, targetTensor.DataF32)
		if s.normalize {
			normalizeVector(vec)
		}
		vectors = append(vectors, vec)
	}

	if len(vectors) == 1 {
		ctx.Result.Output.Value = vectors[0]
	} else {
		ctx.Result.Output.Value = vectors
	}
	ctx.Result.Sync()
	return nil
}

func normalizeVector(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x * x)
	}
	norm := float32(math.Sqrt(sum))
	if norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
}

// GenerationExtractorStage extracts decoded text from generation tensor outputs.
type GenerationExtractorStage struct {
	name string
}

func (s *GenerationExtractorStage) Name() string { return s.name }
func (s *GenerationExtractorStage) Process(ctx *domain.PipelineContext) error {
	textTensor, exists := ctx.Outputs["generated_text"]
	if !exists {
		return fmt.Errorf("stage %q: missing required output tensor 'generated_text'", s.name)
	}

	var generatedText string
	if textTensor.Type == domain.DataTypeInt32 {
		bytes := make([]byte, len(textTensor.DataI32))
		for i, b := range textTensor.DataI32 {
			bytes[i] = byte(b)
		}
		generatedText = string(bytes)
	} else {
		return fmt.Errorf("stage %q: unsupported tensor type %v for generated_text", s.name, textTensor.Type)
	}

	if tokenTensor, ok := ctx.Outputs["generated_tokens"]; ok && tokenTensor.Type == domain.DataTypeInt64 {
		if ctx.Result.Metadata == nil {
			ctx.Result.Metadata = make(map[string]string)
		}
		ctx.Result.Metadata["generated_tokens_count"] = strconv.Itoa(len(tokenTensor.DataI64))
	}

	ctx.Result.Output.Value = generatedText
	ctx.Result.Answer = generatedText
	ctx.Result.Sync()
	return nil
}
