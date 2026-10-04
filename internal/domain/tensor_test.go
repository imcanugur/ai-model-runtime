package domain_test

import (
	"errors"
	"testing"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

func TestTensorFloat32_Valid(t *testing.T) {
	shape := []int64{2, 3}
	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

	tensor, err := domain.NewTensorFloat32(shape, data)
	if err != nil {
		t.Fatalf("expected valid tensor, got error: %v", err)
	}

	if tensor.Type != domain.DataTypeFloat32 {
		t.Errorf("expected type %s, got %s", domain.DataTypeFloat32, tensor.Type)
	}
	if tensor.TotalElements() != 6 {
		t.Errorf("expected 6 total elements, got %d", tensor.TotalElements())
	}
}

func TestTensorInt64_Valid(t *testing.T) {
	shape := []int64{1, 4}
	data := []int64{10, 20, 30, 40}

	tensor, err := domain.NewTensorInt64(shape, data)
	if err != nil {
		t.Fatalf("expected valid tensor, got error: %v", err)
	}

	if tensor.Type != domain.DataTypeInt64 {
		t.Errorf("expected type %s, got %s", domain.DataTypeInt64, tensor.Type)
	}
	if tensor.TotalElements() != 4 {
		t.Errorf("expected 4 total elements, got %d", tensor.TotalElements())
	}
}

func TestTensorInt32_Valid(t *testing.T) {
	shape := []int64{3}
	data := []int32{100, 200, 300}

	tensor, err := domain.NewTensorInt32(shape, data)
	if err != nil {
		t.Fatalf("expected valid tensor, got error: %v", err)
	}

	if tensor.Type != domain.DataTypeInt32 {
		t.Errorf("expected type %s, got %s", domain.DataTypeInt32, tensor.Type)
	}
}

func TestTensor_ShapeCapacityMismatch(t *testing.T) {
	// Shape says 2x3 = 6 elements, but provided data has 5 elements
	shape := []int64{2, 3}
	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0}

	_, err := domain.NewTensorFloat32(shape, data)
	if err == nil {
		t.Fatal("expected error on shape/capacity mismatch, got nil")
	}

	if !errors.Is(err, domain.ErrInvalidTensor) {
		t.Errorf("expected ErrInvalidTensor, got %v", err)
	}
}

func TestTensor_EmptyShape(t *testing.T) {
	var shape []int64
	data := []float32{1.0}

	_, err := domain.NewTensorFloat32(shape, data)
	if err == nil {
		t.Fatal("expected error on empty shape, got nil")
	}
	if !errors.Is(err, domain.ErrInvalidTensor) {
		t.Errorf("expected ErrInvalidTensor, got %v", err)
	}
}

func TestTensor_NegativeDimension(t *testing.T) {
	shape := []int64{2, -3}
	data := []float32{1.0, 2.0}

	_, err := domain.NewTensorFloat32(shape, data)
	if err == nil {
		t.Fatal("expected error on negative dimension, got nil")
	}
	if !errors.Is(err, domain.ErrInvalidTensor) {
		t.Errorf("expected ErrInvalidTensor, got %v", err)
	}
}

func TestTensor_UnsupportedDataType(t *testing.T) {
	tensor := domain.Tensor{
		Type:  domain.DataType("FLOAT64"),
		Shape: []int64{1},
	}
	err := tensor.Validate()
	if err == nil {
		t.Fatal("expected error on unsupported type, got nil")
	}
	if !errors.Is(err, domain.ErrInvalidTensor) {
		t.Errorf("expected ErrInvalidTensor, got %v", err)
	}
}

func TestTensorMap_ValidAndInvalid(t *testing.T) {
	t1, err := domain.NewTensorFloat32([]int64{1, 2}, []float32{1.0, 2.0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tm := domain.TensorMap{
		"input_ids": t1,
	}
	if err := tm.Validate(); err != nil {
		t.Errorf("expected valid tensor map, got %v", err)
	}

	// Empty name is invalid
	tmInvalid := domain.TensorMap{
		"": t1,
	}
	if err := tmInvalid.Validate(); err == nil {
		t.Error("expected error for empty tensor name in map, got nil")
	}
}
