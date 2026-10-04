package runtime_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/pipeline"
	"github.com/imcanugur/ai-model-runtime/internal/registry"
	"github.com/imcanugur/ai-model-runtime/internal/runtime"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/mock"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/onnx"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/sidecar"
)

const samplePipelineStagesYAML = `
    - name: "normalizer"
      type: "text_normalizer"
      config:
        strip_citation_tags: true
    - name: "chunker"
      type: "sliding_window_chunker"
      config:
        max_tokens: 100
        stride: 0
    - name: "inference"
      type: "runtime_infer"
    - name: "reduction"
      type: "token_weighted_logits"
      config:
        human_class_index: 0
        ai_class_index: 1
    - name: "calibrator"
      type: "threshold_calibrator"
      config:
        threshold: 0.5
        positive_label: "AI"
        negative_label: "Human"
`

func createModelPackage(t *testing.T, baseDir, id, version, provider, profile, entrypointRel string) string {
	t.Helper()
	modelDir := filepath.Join(baseDir, id)
	weightsDir := filepath.Join(modelDir, filepath.Dir(entrypointRel))
	if err := os.MkdirAll(weightsDir, 0755); err != nil {
		t.Fatalf("failed to create model dirs: %v", err)
	}

	// Create dummy weight file
	entrypointPath := filepath.Join(modelDir, entrypointRel)
	if err := os.WriteFile(entrypointPath, []byte("fake-model-weights-data"), 0644); err != nil {
		t.Fatalf("failed to create dummy weights file: %v", err)
	}

	manifestYAML := `schema_version: "v1"
id: "` + id + `"
name: "Integration Test Model (` + provider + `)"
version: "` + version + `"
metadata:
  task: "text-classification"
  languages: ["tr", "en"]
resources:
  max_concurrency: 4
  timeout_ms: 5000
runtime:
  provider: "` + provider + `"
  profile: "` + profile + `"
  entrypoint: "` + entrypointRel + `"
pipeline:
  stages:` + samplePipelineStagesYAML

	manifestPath := filepath.Join(modelDir, "model.yaml")
	if err := os.WriteFile(manifestPath, []byte(manifestYAML), 0644); err != nil {
		t.Fatalf("failed to write manifest: %v", err)
	}

	return modelDir
}

func TestIntegration_ONNXRuntimeFlow(t *testing.T) {
	tempModelsDir := t.TempDir()
	createModelPackage(t, tempModelsDir, "text-detector-onnx", "1.0.0", "onnx", "standard_cpu", "weights/model.onnx")

	// 1. Initialize Factory with ONNX provider
	rtFactory := runtime.NewDefaultFactory()
	if err := rtFactory.RegisterProvider("onnx", onnx.NewProvider(nil)); err != nil {
		t.Fatalf("failed to register onnx provider: %v", err)
	}

	// 2. Initialize Registry using AdaptRuntimeFactory
	stageReg := pipeline.NewDefaultStageRegistry()
	modelReg := registry.NewDefaultModelRegistry(stageReg, registry.AdaptRuntimeFactory(rtFactory))

	ctx := context.Background()

	// 3. Discover
	if err := modelReg.Discover(ctx, tempModelsDir); err != nil {
		t.Fatalf("discovery failed: %v", err)
	}

	// 4. Load
	modelKey := domain.NewModelKey("text-detector-onnx", "1.0.0")
	if err := modelReg.Load(ctx, modelKey); err != nil {
		t.Fatalf("load failed: %v", err)
	}
	defer func() { _ = modelReg.Unload(ctx, modelKey) }()

	// 5. Acquire
	handle, err := modelReg.Acquire(modelKey)
	if err != nil {
		t.Fatalf("acquire failed: %v", err)
	}
	defer handle.Release()

	// 6. Execute Pipeline
	input := domain.RawInput{
		Text: "Bu metin ONNX runtime üzerinden pipeline aşamalarından geçerek sınıflandırılacaktır.",
	}
	result, err := handle.Execute(ctx, input)
	if err != nil {
		t.Fatalf("pipeline execution failed: %v", err)
	}

	if result.ModelID != domain.ModelID("text-detector-onnx") {
		t.Errorf("expected result model ID 'text-detector-onnx', got %q", result.ModelID)
	}
	if result.ModelVersion != domain.ModelVersion("1.0.0") {
		t.Errorf("expected result model version '1.0.0', got %q", result.ModelVersion)
	}
	if handle.Key() != modelKey {
		t.Errorf("expected handle key %q, got %q", modelKey, handle.Key())
	}
	if result.Label == "" {
		t.Errorf("expected non-empty label")
	}
	if len(result.Chunks) == 0 {
		t.Errorf("expected at least one chunk in result")
	}
}

func TestIntegration_SidecarSafetensorsFlow(t *testing.T) {
	tempModelsDir := t.TempDir()
	createModelPackage(t, tempModelsDir, "text-detector-safetensors", "2.0.0", "sidecar", "python_torch", "weights/model.safetensors")

	// 1. Start Mock Sidecar Server simulating Python/PyTorch daemon
	server, err := sidecar.NewMockServer()
	if err != nil {
		t.Fatalf("failed to start mock sidecar server: %v", err)
	}
	defer func() { _ = server.Close() }()

	// 2. Initialize Factory with Sidecar provider targeting the mock server
	rtFactory := runtime.NewDefaultFactory()
	sidecarProv := sidecar.NewProvider(server.Endpoint(), nil)
	if err := rtFactory.RegisterProvider("sidecar", sidecarProv); err != nil {
		t.Fatalf("failed to register sidecar provider: %v", err)
	}

	// 3. Initialize Registry using AdaptRuntimeFactory
	stageReg := pipeline.NewDefaultStageRegistry()
	modelReg := registry.NewDefaultModelRegistry(stageReg, registry.AdaptRuntimeFactory(rtFactory))

	ctx := context.Background()

	// 4. Discover
	if err := modelReg.Discover(ctx, tempModelsDir); err != nil {
		t.Fatalf("discovery failed: %v", err)
	}

	// 5. Load
	modelKey := domain.NewModelKey("text-detector-safetensors", "2.0.0")
	if err := modelReg.Load(ctx, modelKey); err != nil {
		t.Fatalf("load failed: %v", err)
	}
	defer func() { _ = modelReg.Unload(ctx, modelKey) }()

	// 6. Acquire
	handle, err := modelReg.Acquire(modelKey)
	if err != nil {
		t.Fatalf("acquire failed: %v", err)
	}
	defer handle.Release()

	// 7. Execute Pipeline
	input := domain.RawInput{
		Text: "Bu metin SafeTensors modelini Python sidecar processine tensor olarak gönderip sonuç alacaktır.",
	}
	result, err := handle.Execute(ctx, input)
	if err != nil {
		t.Fatalf("pipeline execution failed: %v", err)
	}

	if result.ModelID != domain.ModelID("text-detector-safetensors") {
		t.Errorf("expected result model ID 'text-detector-safetensors', got %q", result.ModelID)
	}
	if result.ModelVersion != domain.ModelVersion("2.0.0") {
		t.Errorf("expected result model version '2.0.0', got %q", result.ModelVersion)
	}
	if handle.Key() != modelKey {
		t.Errorf("expected handle key %q, got %q", modelKey, handle.Key())
	}
	if result.Label == "" {
		t.Errorf("expected non-empty label")
	}
	if server.RequestCount() == 0 {
		t.Errorf("expected mock sidecar server to receive inference requests")
	}
}

func TestIntegration_MockRuntimeRegression(t *testing.T) {
	tempModelsDir := t.TempDir()
	createModelPackage(t, tempModelsDir, "text-detector-mock", "0.1.0", "mock", "test", "weights/model.bin")

	rtFactory := runtime.NewDefaultFactory()
	_ = rtFactory.RegisterProvider("mock", mock.NewProvider())

	stageReg := pipeline.NewDefaultStageRegistry()
	modelReg := registry.NewDefaultModelRegistry(stageReg, registry.AdaptRuntimeFactory(rtFactory))

	ctx := context.Background()
	if err := modelReg.Discover(ctx, tempModelsDir); err != nil {
		t.Fatalf("discovery failed: %v", err)
	}

	modelKey := domain.NewModelKey("text-detector-mock", "0.1.0")
	if err := modelReg.Load(ctx, modelKey); err != nil {
		t.Fatalf("load failed: %v", err)
	}
	defer func() { _ = modelReg.Unload(ctx, modelKey) }()

	handle, err := modelReg.Acquire(modelKey)
	if err != nil {
		t.Fatalf("acquire failed: %v", err)
	}
	defer handle.Release()

	result, err := handle.Execute(ctx, domain.RawInput{Text: "Regression testing mock runtime pipeline"})
	if err != nil {
		t.Fatalf("execute failed: %v", err)
	}
	if result.ModelID != domain.ModelID("text-detector-mock") {
		t.Errorf("expected result model ID 'text-detector-mock', got %q", result.ModelID)
	}
	if result.ModelVersion != domain.ModelVersion("0.1.0") {
		t.Errorf("expected result model version '0.1.0', got %q", result.ModelVersion)
	}
	if handle.Key() != modelKey {
		t.Errorf("expected handle key %q, got %q", modelKey, handle.Key())
	}
	if result.PrimaryScore == 0 && result.Label == "" {
		t.Errorf("expected valid primary score or label")
	}
}
