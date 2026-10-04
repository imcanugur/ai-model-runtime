package registry_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	"github.com/imcanugur/ai-model-runtime/internal/registry"
)

func createTestModelPackage(t *testing.T, rootDir string, id string, version string, valid bool) string {
	t.Helper()
	modelDir := filepath.Join(rootDir, fmt.Sprintf("%s-%s", id, version))
	if err := os.MkdirAll(filepath.Join(modelDir, "weights"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(modelDir, "assets"), 0755); err != nil {
		t.Fatal(err)
	}

	var manifestContent string
	if valid {
		manifestContent = fmt.Sprintf(`
schema_version: "v1"
id: %q
name: "Model %s"
version: %q
metadata:
  task: "text-classification"
resources:
  max_concurrency: 4
  timeout_ms: 1000
runtime:
  provider: "mock"
  profile: "cpu"
  entrypoint: "weights/model.bin"
pipeline:
  stages:
    - name: "norm"
      type: "text_normalizer"
`, id, id, version)
		if err := os.WriteFile(filepath.Join(modelDir, "weights", "model.bin"), []byte("bin"), 0644); err != nil {
			t.Fatal(err)
		}
	} else {
		// Invalid manifest: missing schema_version and required fields
		manifestContent = `
id: "broken-model"
`
	}

	if err := os.WriteFile(filepath.Join(modelDir, "model.yaml"), []byte(manifestContent), 0644); err != nil {
		t.Fatal(err)
	}

	return modelDir
}

func TestRegistry_DiscoverAndList(t *testing.T) {
	tempRoot := t.TempDir()

	createTestModelPackage(t, tempRoot, "detector-a", "1.0.0", true)
	createTestModelPackage(t, tempRoot, "detector-b", "2.0.0", true)
	createTestModelPackage(t, tempRoot, "broken", "0.0.1", false) // Should be caught and marked FAILED

	reg := registry.NewDefaultModelRegistry(nil, nil)
	err := reg.Discover(context.Background(), tempRoot)
	if err != nil {
		t.Fatalf("unexpected discovery error: %v", err)
	}

	descriptors := reg.List()
	if len(descriptors) != 3 {
		t.Fatalf("expected 3 model descriptors, got %d", len(descriptors))
	}

	// Verify states
	var foundValidA, foundValidB, foundFailed bool
	for _, desc := range descriptors {
		if desc.Key == "detector-a:1.0.0" && desc.State == domain.StateDiscovered {
			foundValidA = true
		}
		if desc.Key == "detector-b:2.0.0" && desc.State == domain.StateDiscovered {
			foundValidB = true
		}
		if desc.State == domain.StateFailed {
			foundFailed = true
		}
	}

	if !foundValidA || !foundValidB {
		t.Errorf("expected both valid models discovered, got: %+v", descriptors)
	}
	if !foundFailed {
		t.Error("expected broken model to be listed in StateFailed")
	}
}

func TestRegistry_LoadAndAcquireLifecycle(t *testing.T) {
	tempRoot := t.TempDir()
	createTestModelPackage(t, tempRoot, "detector", "1.0.0", true)

	reg := registry.NewDefaultModelRegistry(nil, nil)
	if err := reg.Discover(context.Background(), tempRoot); err != nil {
		t.Fatal(err)
	}

	key := domain.NewModelKey("detector", "1.0.0")

	// Acquire before load must fail
	_, err := reg.Acquire(key)
	if !errors.Is(err, domain.ErrModelNotReady) {
		t.Fatalf("expected ErrModelNotReady, got: %v", err)
	}

	// Load model
	ctx := context.Background()
	if err := reg.Load(ctx, key); err != nil {
		t.Fatalf("failed to load model: %v", err)
	}

	// Acquire after load
	handle, err := reg.Acquire(key)
	if err != nil {
		t.Fatalf("failed to acquire model: %v", err)
	}

	if handle.ID() != "detector" || handle.Version() != "1.0.0" {
		t.Errorf("unexpected handle metadata: %s:%s", handle.ID(), handle.Version())
	}

	// Check refCount via List()
	desc := reg.List()
	if len(desc) != 1 || desc[0].ActiveLeases != 1 {
		t.Errorf("expected 1 active lease, got %d", desc[0].ActiveLeases)
	}

	// Execute through handle
	res, err := handle.Execute(ctx, domain.RawInput{Text: "test input"})
	if err != nil {
		t.Fatalf("handle execution failed: %v", err)
	}
	if res.ModelID != "detector" {
		t.Errorf("expected modelID 'detector', got %s", res.ModelID)
	}

	// Release handle
	handle.Release()

	descAfter := reg.List()
	if descAfter[0].ActiveLeases != 0 {
		t.Errorf("expected 0 active leases after release, got %d", descAfter[0].ActiveLeases)
	}

	// Double release is safe and idempotent
	handle.Release()
}

func TestRegistry_AcquireDefault_SemverSelection(t *testing.T) {
	tempRoot := t.TempDir()

	// Create 3 versions of the same model
	createTestModelPackage(t, tempRoot, "multi-ver", "1.0.0", true)
	createTestModelPackage(t, tempRoot, "multi-ver", "2.1.0", true)
	createTestModelPackage(t, tempRoot, "multi-ver", "2.0.4", true)

	reg := registry.NewDefaultModelRegistry(nil, nil)
	if err := reg.Discover(context.Background(), tempRoot); err != nil {
		t.Fatal(err)
	}

	// Load all 3 versions
	ctx := context.Background()
	for _, v := range []string{"1.0.0", "2.1.0", "2.0.4"} {
		if err := reg.Load(ctx, domain.NewModelKey("multi-ver", domain.ModelVersion(v))); err != nil {
			t.Fatal(err)
		}
	}

	// AcquireDefault should deterministically select highest semver: 2.1.0
	handle, err := reg.AcquireDefault("multi-ver")
	if err != nil {
		t.Fatalf("failed to acquire default model: %v", err)
	}
	defer handle.Release()

	if handle.Version() != "2.1.0" {
		t.Fatalf("expected default version '2.1.0', got %q", handle.Version())
	}
}

func TestRegistry_Unload_DrainsActiveLeases(t *testing.T) {
	tempRoot := t.TempDir()
	createTestModelPackage(t, tempRoot, "drain-test", "1.0.0", true)

	reg := registry.NewDefaultModelRegistry(nil, nil)
	if err := reg.Discover(context.Background(), tempRoot); err != nil {
		t.Fatal(err)
	}

	key := domain.NewModelKey("drain-test", "1.0.0")
	ctx := context.Background()
	if err := reg.Load(ctx, key); err != nil {
		t.Fatal(err)
	}

	// Acquire a lease
	handle, err := reg.Acquire(key)
	if err != nil {
		t.Fatal(err)
	}

	unloadStarted := make(chan struct{})
	unloadFinished := make(chan struct{})

	// Start unload in background goroutine while lease is still held
	go func() {
		close(unloadStarted)
		unloadCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = reg.Unload(unloadCtx, key)
		close(unloadFinished)
	}()

	<-unloadStarted
	time.Sleep(50 * time.Millisecond) // Give unloader time to mark UNLOADING

	// New acquire during UNLOADING must be rejected
	_, err = reg.Acquire(key)
	if !errors.Is(err, domain.ErrModelUnloading) {
		t.Errorf("expected ErrModelUnloading during drain, got: %v", err)
	}

	// Unload must not be finished yet because handle is held
	select {
	case <-unloadFinished:
		t.Fatal("unload completed prematurely before lease was released")
	default:
	}

	// Now release the held lease
	handle.Release()

	// Unload should now finish cleanly
	select {
	case <-unloadFinished:
	case <-time.After(1 * time.Second):
		t.Fatal("unload timed out waiting for lease release")
	}

	// State should now be UNLOADED
	descriptors := reg.List()
	if len(descriptors) != 1 || descriptors[0].State != domain.StateUnloaded {
		t.Errorf("expected StateUnloaded, got: %+v", descriptors)
	}
}

func TestRegistry_ConcurrentAcquireRelease(t *testing.T) {
	tempRoot := t.TempDir()
	createTestModelPackage(t, tempRoot, "concurrency-model", "1.0.0", true)

	reg := registry.NewDefaultModelRegistry(nil, nil)
	if err := reg.Discover(context.Background(), tempRoot); err != nil {
		t.Fatal(err)
	}
	key := domain.NewModelKey("concurrency-model", "1.0.0")
	if err := reg.Load(context.Background(), key); err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	workers := 20
	iterations := 50

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				h, err := reg.Acquire(key)
				if err != nil {
					t.Errorf("concurrent acquire failed: %v", err)
					return
				}
				_, execErr := h.Execute(context.Background(), domain.RawInput{Text: "concurrent"})
				if execErr != nil {
					t.Errorf("concurrent execute failed: %v", execErr)
				}
				h.Release()
			}
		}()
	}

	wg.Wait()

	desc := reg.List()
	if desc[0].ActiveLeases != 0 {
		t.Errorf("expected 0 remaining leases, got %d", desc[0].ActiveLeases)
	}
}

func TestRegistry_ZeroCodeModelAddition(t *testing.T) {
	tempRoot := t.TempDir()

	// Step 1: Initialize registry on empty root
	reg := registry.NewDefaultModelRegistry(nil, nil)
	if err := reg.Discover(context.Background(), tempRoot); err != nil {
		t.Fatal(err)
	}
	if len(reg.List()) != 0 {
		t.Fatalf("expected 0 models, got %d", len(reg.List()))
	}

	// Step 2: Dynamically drop a brand new model folder: models/german-detector
	germanDir := filepath.Join(tempRoot, "german-detector")
	if err := os.MkdirAll(filepath.Join(germanDir, "weights"), 0755); err != nil {
		t.Fatal(err)
	}
	germanYAML := `
schema_version: "v1"
id: "german-detector"
name: "German Academic Text Detector"
version: "1.0.0"
metadata:
  task: "text-classification"
  languages: ["de"]
resources:
  max_concurrency: 2
runtime:
  provider: "mock"
  profile: "cpu"
  entrypoint: "weights/model.onnx"
pipeline:
  stages:
    - name: "norm"
      type: "text_normalizer"
    - name: "chunk"
      type: "sliding_window_chunker"
`
	if err := os.WriteFile(filepath.Join(germanDir, "model.yaml"), []byte(germanYAML), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(germanDir, "weights", "model.onnx"), []byte("onnx-bytes"), 0644); err != nil {
		t.Fatal(err)
	}

	// Step 3: Run discovery again (representing reload or startup)
	// ZERO Go code modified!
	if err := reg.Discover(context.Background(), tempRoot); err != nil {
		t.Fatalf("discovery of new model failed: %v", err)
	}

	key := domain.NewModelKey("german-detector", "1.0.0")
	if err := reg.Load(context.Background(), key); err != nil {
		t.Fatalf("loading dynamically added model failed: %v", err)
	}

	handle, err := reg.Acquire(key)
	if err != nil {
		t.Fatalf("failed to acquire new model: %v", err)
	}
	defer handle.Release()

	res, err := handle.Execute(context.Background(), domain.RawInput{Text: "Guten Tag"})
	if err != nil {
		t.Fatalf("execution on new model failed: %v", err)
	}

	if res.ModelID != "german-detector" || res.ModelVersion != "1.0.0" {
		t.Errorf("unexpected model execution result: %+v", res)
	}
}
