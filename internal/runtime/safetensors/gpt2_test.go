package safetensors_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/pipeline"
	"github.com/imcanugur/ai-model-runtime/internal/registry"
	"github.com/imcanugur/ai-model-runtime/internal/runtime"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/safetensors"
)

func TestGPT2Tokenizer_TurkishRoundTrip(t *testing.T) {
	root := findRepoRoot(t)
	tokPath := filepath.Join(root, "models", "text-generation-tr-gpt2", "tokenizer.json")
	if _, err := os.Stat(tokPath); err != nil {
		t.Skip("tokenizer.json not found")
	}

	tok, err := safetensors.LoadTokenizer(tokPath)
	if err != nil {
		t.Fatalf("failed to load tokenizer: %v", err)
	}

	testCases := []string{
		"Türkiye'nin başkenti nedir?",
		"İstanbul hakkında kısa bir cümle yaz.",
		"Türk hakkında kısa bir cümle yaz.",
		"Bugün hava çok güzel. Bu cümleyi İngilizceye çevir:",
		"Yapay zeka nedir? Kısaca açıkla.",
		"Kırmızı bir araba hakkında kısa bir hikaye yaz.",
		"Merhaba, nasılsın?",
		"ğüşıöç ĞÜŞİÖÇ",
	}

	for _, text := range testCases {
		encoded := tok.Encode(text)
		decoded := tok.Decode(encoded, false)
		t.Logf("ORIGINAL: %q", text)
		t.Logf("TOKENS  : %v (%d tokens)", encoded, len(encoded))
		t.Logf("DECODED : %q", decoded)
		if decoded != text {
			t.Errorf("Roundtrip mismatch!\nExpected: %q\nGot:      %q", text, decoded)
		}
	}
}

func TestGPT2Model_LoadAndForward(t *testing.T) {
	root := findRepoRoot(t)
	modelPath := filepath.Join(root, "models", "text-generation-tr-gpt2", "model.safetensors")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("model.safetensors not found")
	}

	stFile, err := safetensors.OpenFile(modelPath)
	if err != nil {
		t.Fatalf("failed to open safetensors: %v", err)
	}
	defer stFile.Close()

	t0 := time.Now()
	cfg := safetensors.DefaultGPT2MediumConfig()
	model, err := safetensors.LoadGPT2Model(stFile, cfg)
	if err != nil {
		t.Fatalf("failed to load GPT2 model: %v", err)
	}
	t.Logf("GPT-2 Medium 350M loaded successfully in %v", time.Since(t0))

	// Single step test
	ctx := context.Background()
	promptTokens := []int64{3581, 1340, 1752, 307, 10178, 670, 14} // "İstanbul hakkında kısa bir cümle yaz."
	t1 := time.Now()
	genTokens, err := model.GenerateAutoregressive(ctx, promptTokens, 1)
	if err != nil {
		t.Fatalf("GenerateAutoregressive failed: %v", err)
	}
	t.Logf("Autoregressive 1-token step took %v, generated token: %v", time.Since(t1), genTokens)

	if len(genTokens) != 1 {
		t.Fatalf("expected 1 generated token, got %d", len(genTokens))
	}
}

func TestGPT2_MaxNewTokensProgression(t *testing.T) {
	root := findRepoRoot(t)
	modelDir := filepath.Join(root, "models", "text-generation-tr-gpt2")
	modelPath := filepath.Join(modelDir, "model.safetensors")
	tokPath := filepath.Join(modelDir, "tokenizer.json")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("model.safetensors not found")
	}

	tok, err := safetensors.LoadTokenizer(tokPath)
	if err != nil {
		t.Fatalf("failed to load tokenizer: %v", err)
	}

	stFile, err := safetensors.OpenFile(modelPath)
	if err != nil {
		t.Fatalf("failed to open safetensors: %v", err)
	}
	defer stFile.Close()

	cfg := safetensors.DefaultGPT2MediumConfig()
	model, err := safetensors.LoadGPT2Model(stFile, cfg)
	if err != nil {
		t.Fatalf("failed to load GPT2 model: %v", err)
	}

	prompt := "İstanbul hakkında kısa bir cümle yaz."
	promptTokens := tok.Encode(prompt)
	ctx := context.Background()

	tokenLimits := []int{1, 2, 4, 8, 16}
	for _, maxTokens := range tokenLimits {
		t0 := time.Now()
		gen, err := model.GenerateAutoregressive(ctx, promptTokens, maxTokens)
		if err != nil {
			t.Fatalf("generation failed for max_new_tokens=%d: %v", maxTokens, err)
		}
		duration := time.Since(t0)
		decoded := tok.Decode(gen, false)

		t.Logf("=== max_new_tokens=%d ===", maxTokens)
		t.Logf("  sequence_length    : %d (prompt %d + generated %d)", len(promptTokens)+len(gen), len(promptTokens), len(gen))
		t.Logf("  selected_token_ids : %v", gen)
		t.Logf("  decoded_token_text : %q", decoded)
		t.Logf("  logits_shape       : [%d]", cfg.VocabSize)
		t.Logf("  duration           : %v", duration)

		if len(gen) > maxTokens {
			t.Errorf("expected at most %d tokens, got %d", maxTokens, len(gen))
		}
	}
}

func TestGPT2_ModelRegistryAndPipeline(t *testing.T) {
	root := findRepoRoot(t)
	modelArtifact := filepath.Join(root, "models", "text-generation-tr-gpt2", "model.safetensors")
	if _, err := os.Stat(modelArtifact); err != nil {
		t.Skip("models/text-generation-tr-gpt2/model.safetensors not present")
	}

	rtFactory := runtime.NewDefaultFactory()
	stProv := safetensors.NewProvider()
	if err := rtFactory.RegisterProvider("safetensors", stProv); err != nil {
		t.Fatalf("failed to register safetensors provider: %v", err)
	}

	stageReg := pipeline.NewDefaultStageRegistry()
	modelReg := registry.NewDefaultModelRegistry(stageReg, registry.AdaptRuntimeFactory(rtFactory))

	ctx := context.Background()
	modelsDir := filepath.Join(root, "models")
	if err := modelReg.Discover(ctx, modelsDir); err != nil {
		t.Fatalf("model discovery failed: %v", err)
	}

	gpt2Key := domain.NewModelKey("text-generation-tr-gpt2", "1.0.0")
	if err := modelReg.Load(ctx, gpt2Key); err != nil {
		t.Fatalf("failed to load text-generation-tr-gpt2: %v", err)
	}
	defer func() { _ = modelReg.Unload(ctx, gpt2Key) }()

	handle, err := modelReg.Acquire(gpt2Key)
	if err != nil {
		t.Fatalf("failed to acquire handle: %v", err)
	}
	defer handle.Release()

	// Test inference through pipeline handle.Execute
	res, err := handle.Execute(ctx, domain.RawInput{
		Text: "İstanbul hakkında kısa bir cümle yaz.",
		Parameters: map[string]string{
			"max_new_tokens": "4",
		},
	})
	if err != nil {
		t.Fatalf("handle.Execute failed: %v", err)
	}

	val, ok := res.Output.Value.(string)
	if !ok {
		t.Fatalf("expected string output value, got: %T (%v)", res.Output.Value, res.Output.Value)
	}
	t.Logf("Pipeline Result Output: %q", val)
	if val == "" {
		t.Errorf("pipeline produced empty string output")
	}
}
