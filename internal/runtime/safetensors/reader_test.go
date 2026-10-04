package safetensors_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/imcanugur/ai-model-runtime/internal/runtime/safetensors"
)

func findRepoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("failed to get working dir: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("could not find repo root with go.mod starting from %s", dir)
		}
		dir = parent
	}
}

func TestSafeTensorsReader_RealModel(t *testing.T) {
	root := findRepoRoot(t)
	modelPath := filepath.Join(root, "models", "text-generation-tr", "model.safetensors")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model file not found at %s: skipping", modelPath)
	}

	stFile, err := safetensors.OpenFile(modelPath)
	if err != nil {
		t.Fatalf("failed to open safetensors file: %v", err)
	}
	defer stFile.Close()

	names := stFile.TensorNames()
	if len(names) == 0 {
		t.Fatalf("expected at least one tensor in %s", modelPath)
	}

	t.Logf("Successfully read SafeTensors index in pure Go! Found %d tensors.", len(names))
	for i := 0; i < 5 && i < len(names); i++ {
		info, _ := stFile.GetTensorInfo(names[i])
		t.Logf("  Tensor [%s]: DType=%s Shape=%v Offsets=%v", names[i], info.DType, info.Shape, info.DataOffsets)
	}

	// Read first tensor to verify conversion
	firstTensorName := names[0]
	tensor, err := stFile.ReadTensor(firstTensorName)
	if err != nil {
		t.Fatalf("failed to read tensor %q: %v", firstTensorName, err)
	}
	t.Logf("Successfully read and decoded tensor %q: Type=%s Shape=%v", firstTensorName, tensor.Type, tensor.Shape)
}
