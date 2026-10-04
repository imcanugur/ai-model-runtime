package sidecar_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	internalHTTP "github.com/imcanugur/ai-model-runtime/internal/http"
	storeMemory "github.com/imcanugur/ai-model-runtime/internal/jobstore/memory"
	"github.com/imcanugur/ai-model-runtime/internal/pipeline"
	queueMemory "github.com/imcanugur/ai-model-runtime/internal/queue/memory"
	"github.com/imcanugur/ai-model-runtime/internal/registry"
	"github.com/imcanugur/ai-model-runtime/internal/runtime"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/sidecar"
	"github.com/imcanugur/ai-model-runtime/internal/service"
)

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func TestEmbeddingModel_RealTurkishInference(t *testing.T) {
	pythonPath := findPythonBinary(t)
	if pythonPath == "" {
		t.Skip("Python interpreter not found; skipping real Turkish embedding tests")
	}

	repoRoot := getRepoRoot(t)
	modelArtifact := filepath.Join(repoRoot, "models", "embedding-tr-real", "model.safetensors")
	if _, err := filepath.Abs(modelArtifact); err != nil {
		t.Skip("models/embedding-tr-real/model.safetensors not present")
	}

	endpoint, cancelDaemon, killDaemon := startRealSidecarDaemon(t, pythonPath)
	defer killDaemon()
	defer cancelDaemon()

	// 1. Initialize Runtime Factory & Provider
	rtFactory := runtime.NewDefaultFactory()
	sidecarProv := sidecar.NewProvider(endpoint, nil)
	if err := rtFactory.RegisterProvider("sidecar", sidecarProv); err != nil {
		t.Fatalf("failed to register sidecar provider: %v", err)
	}

	// 2. Initialize Registry
	stageReg := pipeline.NewDefaultStageRegistry()
	modelReg := registry.NewDefaultModelRegistry(stageReg, registry.AdaptRuntimeFactory(rtFactory))

	ctx := context.Background()

	// 3. Discover models
	modelsDir := filepath.Join(repoRoot, "models")
	if err := modelReg.Discover(ctx, modelsDir); err != nil {
		t.Fatalf("model discovery failed: %v", err)
	}

	embedKey := domain.NewModelKey("embedding-tr-real", "1.0.0")
	if err := modelReg.Load(ctx, embedKey); err != nil {
		t.Fatalf("failed to load embedding-tr-real model: %v", err)
	}
	defer func() { _ = modelReg.Unload(ctx, embedKey) }()

	// 4. Acquire Handle
	handle, err := modelReg.Acquire(embedKey)
	if err != nil {
		t.Fatalf("failed to acquire embedding-tr-real handle: %v", err)
	}
	defer handle.Release()

	// Test 5 Turkish sentences
	sentences := []string{
		"Merhaba, nasılsın?",
		"Bugün hava çok güzel ve güneşli.",
		"AI Model Runtime, yapay zeka destekli bir metin analiz platformudur.",
		"Bu belge sözleşmenin tarafları ve yükümlülükleri hakkında bilgi içeriyor.",
		"Bugün hava çok güzel ve güneşli.",
	}

	vectors := make([][]float32, len(sentences))
	latencies := make([]int64, len(sentences))

	for i, s := range sentences {
		start := time.Now()
		res, err := handle.Execute(ctx, domain.RawInput{Text: s, Language: "tr"})
		latencies[i] = time.Since(start).Milliseconds()
		if err != nil {
			t.Fatalf("Sentence %d (%q) execution failed: %v", i+1, s, err)
		}

		vec, ok := res.Output.Value.([]float32)
		if !ok {
			t.Fatalf("Sentence %d: expected []float32 in Output.Value, got %T (%+v)", i+1, res.Output.Value, res.Output.Value)
		}
		if len(vec) != 384 {
			t.Errorf("Sentence %d: expected vector dimension 384, got %d", i+1, len(vec))
		}
		vectors[i] = vec
		t.Logf("Sentence %d: %q -> Dim: %d Latency: %dms", i+1, s, len(vec), latencies[i])
	}

	// Sentence 2 and 5 are identical ("Bugün hava çok güzel ve güneşli.")
	sim2and5 := cosineSimilarity(vectors[1], vectors[4])
	t.Logf("Cosine similarity between identical sentences (2 and 5): %.6f", sim2and5)
	if sim2and5 < 0.999 {
		t.Errorf("expected identical sentences to have similarity >= 0.999, got %.6f", sim2and5)
	}

	// Sentence 2 ("hava durumu") and Sentence 4 ("hukuki sözleşme") should have significantly lower similarity
	sim2and4 := cosineSimilarity(vectors[1], vectors[3])
	t.Logf("Cosine similarity between unrelated sentences (2 and 4): %.6f", sim2and4)
	if sim2and4 >= 0.60 {
		t.Errorf("expected unrelated sentences to have similarity < 0.60, got %.6f", sim2and4)
	}

	// Sentence 1 ("selamlama") and Sentence 3 ("teknoloji platformu")
	sim1and3 := cosineSimilarity(vectors[0], vectors[2])
	t.Logf("Cosine similarity between sentences (1 and 3): %.6f", sim1and3)

	// 5. Test HTTP E2E on POST /v1/models/embedding-tr-real/predict
	jobStore := storeMemory.NewMemoryJobStore()
	jobQueue := queueMemory.NewMemoryQueue()
	defer jobQueue.Close()

	inferSvc, err := service.NewInferenceService(service.ServiceConfig{
		Registry: modelReg,
		JobStore: jobStore,
		Queue:    jobQueue,
	})
	if err != nil {
		t.Fatalf("failed to create inference service: %v", err)
	}

	httpCfg := internalHTTP.DefaultServerConfig()
	server, err := internalHTTP.NewServer(httpCfg, inferSvc, internalHTTP.ServerDependencies{
		Queue: jobQueue,
	})
	if err != nil {
		t.Fatalf("failed to create HTTP server: %v", err)
	}

	ts := httptest.NewServer(server.Handler())
	defer ts.Close()

	httpPayload := map[string]any{
		"input": map[string]any{
			"text":     "AI Model Runtime, yapay zeka destekli bir metin analiz platformudur.",
			"language": "tr",
		},
	}
	reqBytes, _ := json.Marshal(httpPayload)

	httpResp, err := http.Post(
		ts.URL+"/v1/models/embedding-tr-real/predict",
		"application/json",
		bytes.NewReader(reqBytes),
	)
	if err != nil {
		t.Fatalf("HTTP POST to embedding-tr-real failed: %v", err)
	}
	defer httpResp.Body.Close()

	bodyBytes, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 OK from embedding-tr-real, got %d: %s", httpResp.StatusCode, string(bodyBytes))
	}

	var resp internalHTTP.InferenceResponse
	if err := json.Unmarshal(bodyBytes, &resp); err != nil {
		t.Fatalf("failed to parse HTTP response as InferenceResponse: %v, body: %s", err, string(bodyBytes))
	}

	dataBytes, _ := json.Marshal(resp.Data.Value)
	var valSlice []float32
	if err := json.Unmarshal(dataBytes, &valSlice); err != nil {
		t.Fatalf("failed to parse data.value as []float32: %v, body: %s", err, string(dataBytes))
	}

	if len(valSlice) != 384 {
		t.Errorf("expected 384 numbers in embedding HTTP response, got %d", len(valSlice))
	}
	if resp.Meta.Model.ID != "embedding-tr-real" {
		t.Errorf("expected model.id embedding-tr-real, got %s", resp.Meta.Model.ID)
	}
	t.Logf("HTTP E2E Successful: Model=%s Dimension=%d Sample=%v", resp.Meta.Model.ID, len(valSlice), valSlice[:3])
}
