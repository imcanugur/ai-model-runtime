package domain

import (
	"fmt"
)

// DataType represents supported tensor primitive types.
type DataType string

const (
	DataTypeFloat32 DataType = "FLOAT32"
	DataTypeInt64   DataType = "INT64"
	DataTypeInt32   DataType = "INT32"
)

// Tensor represents an n-dimensional contiguous array with strict shape validation.
type Tensor struct {
	Type    DataType  `json:"type"`
	Shape   []int64   `json:"shape"`
	DataF32 []float32 `json:"data_f32,omitempty"`
	DataI64 []int64   `json:"data_i64,omitempty"`
	DataI32 []int32   `json:"data_i32,omitempty"`
}

// NewTensorFloat32 constructs a validated FLOAT32 tensor.
func NewTensorFloat32(shape []int64, data []float32) (Tensor, error) {
	t := Tensor{
		Type:    DataTypeFloat32,
		Shape:   shape,
		DataF32: data,
	}
	if err := t.Validate(); err != nil {
		return Tensor{}, err
	}
	return t, nil
}

// NewTensorInt64 constructs a validated INT64 tensor.
func NewTensorInt64(shape []int64, data []int64) (Tensor, error) {
	t := Tensor{
		Type:    DataTypeInt64,
		Shape:   shape,
		DataI64: data,
	}
	if err := t.Validate(); err != nil {
		return Tensor{}, err
	}
	return t, nil
}

// NewTensorInt32 constructs a validated INT32 tensor.
func NewTensorInt32(shape []int64, data []int32) (Tensor, error) {
	t := Tensor{
		Type:    DataTypeInt32,
		Shape:   shape,
		DataI32: data,
	}
	if err := t.Validate(); err != nil {
		return Tensor{}, err
	}
	return t, nil
}

// TotalElements calculates the product of all dimensions in the shape.
func (t Tensor) TotalElements() int64 {
	if len(t.Shape) == 0 {
		return 0
	}
	var total int64 = 1
	for _, dim := range t.Shape {
		if dim <= 0 {
			return 0
		}
		total *= dim
	}
	return total
}

// Validate checks that the tensor has a valid type, non-empty positive shape,
// and the data slice length exactly matches the shape's expected element count.
func (t Tensor) Validate() error {
	switch t.Type {
	case DataTypeFloat32, DataTypeInt64, DataTypeInt32:
	case "":
		return NewValidationError("tensor.type", "tensor type is required", ErrInvalidTensor)
	default:
		return NewValidationError("tensor.type", fmt.Sprintf("unsupported tensor data type %q", t.Type), ErrInvalidTensor)
	}

	if len(t.Shape) == 0 {
		return NewValidationError("tensor.shape", "tensor shape cannot be empty", ErrInvalidTensor)
	}

	for i, dim := range t.Shape {
		if dim <= 0 {
			return NewValidationError("tensor.shape", fmt.Sprintf("shape dimension at index %d must be greater than 0, got %d", i, dim), ErrInvalidTensor)
		}
	}

	expected := t.TotalElements()

	switch t.Type {
	case DataTypeFloat32:
		actual := int64(len(t.DataF32))
		if actual != expected {
			return NewValidationError("tensor.data", fmt.Sprintf("FLOAT32 data length (%d) does not match shape %v capacity (%d)", actual, t.Shape, expected), ErrInvalidTensor)
		}
		if len(t.DataI64) > 0 || len(t.DataI32) > 0 {
			return NewValidationError("tensor.data", "non-FLOAT32 data slices must be empty for FLOAT32 tensor", ErrInvalidTensor)
		}

	case DataTypeInt64:
		actual := int64(len(t.DataI64))
		if actual != expected {
			return NewValidationError("tensor.data", fmt.Sprintf("INT64 data length (%d) does not match shape %v capacity (%d)", actual, t.Shape, expected), ErrInvalidTensor)
		}
		if len(t.DataF32) > 0 || len(t.DataI32) > 0 {
			return NewValidationError("tensor.data", "non-INT64 data slices must be empty for INT64 tensor", ErrInvalidTensor)
		}

	case DataTypeInt32:
		actual := int64(len(t.DataI32))
		if actual != expected {
			return NewValidationError("tensor.data", fmt.Sprintf("INT32 data length (%d) does not match shape %v capacity (%d)", actual, t.Shape, expected), ErrInvalidTensor)
		}
		if len(t.DataF32) > 0 || len(t.DataI64) > 0 {
			return NewValidationError("tensor.data", "non-INT32 data slices must be empty for INT32 tensor", ErrInvalidTensor)
		}
	}

	return nil
}

// TensorMap represents named tensors for model input/output.
type TensorMap map[string]Tensor

// Validate checks every tensor inside the map.
func (tm TensorMap) Validate() error {
	for name, tensor := range tm {
		if name == "" {
			return NewValidationError("tensor_map.name", "tensor name cannot be empty", ErrInvalidTensor)
		}
		if err := tensor.Validate(); err != nil {
			return fmt.Errorf("invalid tensor %q: %w", name, err)
		}
	}
	return nil
}
