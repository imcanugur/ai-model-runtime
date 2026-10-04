package sidecar

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

// Magic numbers and protocol limits for safety
const (
	ProtocolVersion uint8  = 1
	MaxPayloadSize  uint32 = 64 * 1024 * 1024 // 64 MB maximum tensor payload
)

var (
	ErrPayloadTooLarge = errors.New("sidecar payload exceeds maximum allowed size")
	ErrProtocolFormat  = errors.New("sidecar wire protocol decoding error")
)

// InferenceRequest defines the model-agnostic input package for external execution.
type InferenceRequest struct {
	RequestID  string            `json:"request_id"`
	ModelKey   string            `json:"model_key"`
	Inputs     domain.TensorMap  `json:"inputs"`
	Parameters map[string]string `json:"parameters,omitempty"`
}

// InferenceResponse defines the model-agnostic execution output package.
type InferenceResponse struct {
	RequestID string           `json:"request_id"`
	Outputs   domain.TensorMap `json:"outputs"`
	Error     string           `json:"error,omitempty"`
}

// EncodeRequest serializes an InferenceRequest into a length-prefixed binary frame.
func EncodeRequest(w io.Writer, req *InferenceRequest) error {
	if req == nil {
		return fmt.Errorf("%w: cannot encode nil request", ErrProtocolFormat)
	}

	var buf bytes.Buffer

	// 1. Protocol Version
	buf.WriteByte(ProtocolVersion)

	// 2. RequestID
	if err := writeString(&buf, req.RequestID); err != nil {
		return err
	}

	// 3. ModelKey
	if err := writeString(&buf, req.ModelKey); err != nil {
		return err
	}

	// 4. Parameters
	if len(req.Parameters) > math.MaxUint16 {
		return fmt.Errorf("%w: parameter count exceeds uint16", ErrProtocolFormat)
	}
	if err := binary.Write(&buf, binary.BigEndian, uint16(len(req.Parameters))); err != nil {
		return err
	}
	for k, v := range req.Parameters {
		if err := writeString(&buf, k); err != nil {
			return err
		}
		if err := writeString(&buf, v); err != nil {
			return err
		}
	}

	// 5. Tensor Inputs
	if err := encodeTensorMap(&buf, req.Inputs); err != nil {
		return err
	}

	return writeFrame(w, buf.Bytes())
}

// DecodeRequest reads and deserializes an InferenceRequest from a length-prefixed binary frame.
func DecodeRequest(r io.Reader) (*InferenceRequest, error) {
	payload, err := readFrame(r)
	if err != nil {
		return nil, err
	}

	buf := bytes.NewReader(payload)

	// 1. Version
	ver, err := buf.ReadByte()
	if err != nil {
		return nil, fmt.Errorf("%w: failed to read protocol version: %v", ErrProtocolFormat, err)
	}
	if ver != ProtocolVersion {
		return nil, fmt.Errorf("%w: unsupported protocol version %d (expected %d)", ErrProtocolFormat, ver, ProtocolVersion)
	}

	// 2. RequestID
	reqID, err := readString(buf)
	if err != nil {
		return nil, err
	}

	// 3. ModelKey
	modelKey, err := readString(buf)
	if err != nil {
		return nil, err
	}

	// 4. Parameters
	var paramCount uint16
	if err := binary.Read(buf, binary.BigEndian, &paramCount); err != nil {
		return nil, fmt.Errorf("%w: failed to read param count: %v", ErrProtocolFormat, err)
	}
	params := make(map[string]string, paramCount)
	for i := 0; i < int(paramCount); i++ {
		k, err := readString(buf)
		if err != nil {
			return nil, err
		}
		v, err := readString(buf)
		if err != nil {
			return nil, err
		}
		params[k] = v
	}

	// 5. Tensor Inputs
	inputs, err := decodeTensorMap(buf)
	if err != nil {
		return nil, err
	}

	return &InferenceRequest{
		RequestID:  reqID,
		ModelKey:   modelKey,
		Inputs:     inputs,
		Parameters: params,
	}, nil
}

// EncodeResponse serializes an InferenceResponse into a length-prefixed binary frame.
func EncodeResponse(w io.Writer, resp *InferenceResponse) error {
	if resp == nil {
		return fmt.Errorf("%w: cannot encode nil response", ErrProtocolFormat)
	}

	var buf bytes.Buffer

	// 1. Protocol Version
	buf.WriteByte(ProtocolVersion)

	// 2. RequestID
	if err := writeString(&buf, resp.RequestID); err != nil {
		return err
	}

	// 3. Error String
	if err := writeString(&buf, resp.Error); err != nil {
		return err
	}

	// 4. Tensor Outputs
	if err := encodeTensorMap(&buf, resp.Outputs); err != nil {
		return err
	}

	return writeFrame(w, buf.Bytes())
}

// DecodeResponse reads and deserializes an InferenceResponse from a length-prefixed binary frame.
func DecodeResponse(r io.Reader) (*InferenceResponse, error) {
	payload, err := readFrame(r)
	if err != nil {
		return nil, err
	}

	buf := bytes.NewReader(payload)

	// 1. Version
	ver, err := buf.ReadByte()
	if err != nil {
		return nil, fmt.Errorf("%w: failed to read protocol version: %v", ErrProtocolFormat, err)
	}
	if ver != ProtocolVersion {
		return nil, fmt.Errorf("%w: unsupported protocol version %d (expected %d)", ErrProtocolFormat, ver, ProtocolVersion)
	}

	// 2. RequestID
	reqID, err := readString(buf)
	if err != nil {
		return nil, err
	}

	// 3. Error String
	errStr, err := readString(buf)
	if err != nil {
		return nil, err
	}

	// 4. Tensor Outputs
	outputs, err := decodeTensorMap(buf)
	if err != nil {
		return nil, err
	}

	return &InferenceResponse{
		RequestID: reqID,
		Outputs:   outputs,
		Error:     errStr,
	}, nil
}

func encodeTensorMap(buf *bytes.Buffer, tm domain.TensorMap) error {
	if len(tm) > math.MaxUint16 {
		return fmt.Errorf("%w: tensor map count exceeds uint16", ErrProtocolFormat)
	}
	if err := binary.Write(buf, binary.BigEndian, uint16(len(tm))); err != nil {
		return err
	}

	for name, tensor := range tm {
		if err := writeString(buf, name); err != nil {
			return err
		}

		var typeCode byte
		switch tensor.Type {
		case domain.DataTypeFloat32:
			typeCode = 1
		case domain.DataTypeInt64:
			typeCode = 2
		case domain.DataTypeInt32:
			typeCode = 3
		default:
			return fmt.Errorf("%w: unsupported tensor type %q", ErrProtocolFormat, tensor.Type)
		}
		buf.WriteByte(typeCode)

		// Shape
		if len(tensor.Shape) > math.MaxUint16 {
			return fmt.Errorf("%w: tensor rank exceeds uint16", ErrProtocolFormat)
		}
		if err := binary.Write(buf, binary.BigEndian, uint16(len(tensor.Shape))); err != nil {
			return err
		}
		for _, dim := range tensor.Shape {
			if err := binary.Write(buf, binary.BigEndian, dim); err != nil {
				return err
			}
		}

		// Data
		switch tensor.Type {
		case domain.DataTypeFloat32:
			count := uint32(len(tensor.DataF32))
			if err := binary.Write(buf, binary.BigEndian, count); err != nil {
				return err
			}
			for _, v := range tensor.DataF32 {
				if err := binary.Write(buf, binary.BigEndian, math.Float32bits(v)); err != nil {
					return err
				}
			}
		case domain.DataTypeInt64:
			count := uint32(len(tensor.DataI64))
			if err := binary.Write(buf, binary.BigEndian, count); err != nil {
				return err
			}
			for _, v := range tensor.DataI64 {
				if err := binary.Write(buf, binary.BigEndian, v); err != nil {
					return err
				}
			}
		case domain.DataTypeInt32:
			count := uint32(len(tensor.DataI32))
			if err := binary.Write(buf, binary.BigEndian, count); err != nil {
				return err
			}
			for _, v := range tensor.DataI32 {
				if err := binary.Write(buf, binary.BigEndian, v); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

func decodeTensorMap(buf *bytes.Reader) (domain.TensorMap, error) {
	var count uint16
	if err := binary.Read(buf, binary.BigEndian, &count); err != nil {
		return nil, fmt.Errorf("%w: failed to read tensor count: %v", ErrProtocolFormat, err)
	}

	tm := make(domain.TensorMap, count)
	for i := 0; i < int(count); i++ {
		name, err := readString(buf)
		if err != nil {
			return nil, err
		}

		typeCode, err := buf.ReadByte()
		if err != nil {
			return nil, fmt.Errorf("%w: failed to read tensor type: %v", ErrProtocolFormat, err)
		}

		var rank uint16
		if err := binary.Read(buf, binary.BigEndian, &rank); err != nil {
			return nil, fmt.Errorf("%w: failed to read tensor rank: %v", ErrProtocolFormat, err)
		}

		shape := make([]int64, rank)
		for r := 0; r < int(rank); r++ {
			if err := binary.Read(buf, binary.BigEndian, &shape[r]); err != nil {
				return nil, fmt.Errorf("%w: failed to read tensor dimension: %v", ErrProtocolFormat, err)
			}
		}

		var elemCount uint32
		if err := binary.Read(buf, binary.BigEndian, &elemCount); err != nil {
			return nil, fmt.Errorf("%w: failed to read element count: %v", ErrProtocolFormat, err)
		}

		var tensor domain.Tensor
		switch typeCode {
		case 1: // FLOAT32
			data := make([]float32, elemCount)
			for e := 0; e < int(elemCount); e++ {
				var bits uint32
				if err := binary.Read(buf, binary.BigEndian, &bits); err != nil {
					return nil, fmt.Errorf("%w: failed to read float32 element: %v", ErrProtocolFormat, err)
				}
				data[e] = math.Float32frombits(bits)
			}
			t, err := domain.NewTensorFloat32(shape, data)
			if err != nil {
				return nil, fmt.Errorf("%w: invalid float32 tensor %q: %v", ErrProtocolFormat, name, err)
			}
			tensor = t

		case 2: // INT64
			data := make([]int64, elemCount)
			for e := 0; e < int(elemCount); e++ {
				if err := binary.Read(buf, binary.BigEndian, &data[e]); err != nil {
					return nil, fmt.Errorf("%w: failed to read int64 element: %v", ErrProtocolFormat, err)
				}
			}
			t, err := domain.NewTensorInt64(shape, data)
			if err != nil {
				return nil, fmt.Errorf("%w: invalid int64 tensor %q: %v", ErrProtocolFormat, name, err)
			}
			tensor = t

		case 3: // INT32
			data := make([]int32, elemCount)
			for e := 0; e < int(elemCount); e++ {
				if err := binary.Read(buf, binary.BigEndian, &data[e]); err != nil {
					return nil, fmt.Errorf("%w: failed to read int32 element: %v", ErrProtocolFormat, err)
				}
			}
			t, err := domain.NewTensorInt32(shape, data)
			if err != nil {
				return nil, fmt.Errorf("%w: invalid int32 tensor %q: %v", ErrProtocolFormat, name, err)
			}
			tensor = t

		default:
			return nil, fmt.Errorf("%w: unknown tensor type code %d", ErrProtocolFormat, typeCode)
		}

		tm[name] = tensor
	}

	return tm, nil
}

func writeFrame(w io.Writer, payload []byte) error {
	length := uint32(len(payload))
	if length > MaxPayloadSize {
		return ErrPayloadTooLarge
	}
	if err := binary.Write(w, binary.BigEndian, length); err != nil {
		return fmt.Errorf("failed to write frame length: %w", err)
	}
	if _, err := w.Write(payload); err != nil {
		return fmt.Errorf("failed to write frame payload: %w", err)
	}
	return nil
}

func readFrame(r io.Reader) ([]byte, error) {
	var length uint32
	if err := binary.Read(r, binary.BigEndian, &length); err != nil {
		return nil, fmt.Errorf("failed to read frame length: %w", err)
	}
	if length > MaxPayloadSize {
		return nil, ErrPayloadTooLarge
	}

	payload := make([]byte, length)
	if _, err := io.ReadFull(r, payload); err != nil {
		return nil, fmt.Errorf("failed to read frame payload: %w", err)
	}
	return payload, nil
}

func writeString(buf *bytes.Buffer, s string) error {
	b := []byte(s)
	if len(b) > math.MaxUint16 {
		return fmt.Errorf("%w: string length exceeds uint16", ErrProtocolFormat)
	}
	if err := binary.Write(buf, binary.BigEndian, uint16(len(b))); err != nil {
		return err
	}
	buf.Write(b)
	return nil
}

func readString(buf *bytes.Reader) (string, error) {
	var length uint16
	if err := binary.Read(buf, binary.BigEndian, &length); err != nil {
		return "", fmt.Errorf("%w: failed to read string length: %v", ErrProtocolFormat, err)
	}
	bytes := make([]byte, length)
	if _, err := buf.Read(bytes); err != nil {
		return "", fmt.Errorf("%w: failed to read string data: %v", ErrProtocolFormat, err)
	}
	return string(bytes), nil
}
