package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sync"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
)

// TensorInfo describes metadata of a stored tensor.
type TensorInfo struct {
	DType       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// File provides access to tensors stored in a HuggingFace SafeTensors file.
type File struct {
	path       string
	file       *os.File
	headerLen  uint64
	dataOffset int64
	metadata   map[string]string
	tensors    map[string]TensorInfo
	mu         sync.RWMutex
}

// OpenFile opens and indexes a SafeTensors file on disk.
func OpenFile(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open safetensors file %q: %w", path, err)
	}

	var headerLen uint64
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("failed to read safetensors header length: %w", err)
	}

	if headerLen == 0 || headerLen > 100*1024*1024 {
		_ = f.Close()
		return nil, fmt.Errorf("invalid safetensors header length: %d bytes", headerLen)
	}

	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("failed to read safetensors header JSON: %w", err)
	}

	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("failed to parse safetensors header JSON: %w", err)
	}

	meta := make(map[string]string)
	tensors := make(map[string]TensorInfo)

	for k, v := range rawHeader {
		if k == "__metadata__" {
			_ = json.Unmarshal(v, &meta)
			continue
		}
		var ti TensorInfo
		if err := json.Unmarshal(v, &ti); err != nil {
			continue
		}
		tensors[k] = ti
	}

	dataOffset := int64(8 + headerLen)

	return &File{
		path:       path,
		file:       f,
		headerLen:  headerLen,
		dataOffset: dataOffset,
		metadata:   meta,
		tensors:    tensors,
	}, nil
}

// Close closes the underlying file handle.
func (f *File) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.file != nil {
		err := f.file.Close()
		f.file = nil
		return err
	}
	return nil
}

// TensorNames returns the list of tensor names stored in the file.
func (f *File) TensorNames() []string {
	names := make([]string, 0, len(f.tensors))
	for k := range f.tensors {
		names = append(names, k)
	}
	return names
}

// HasTensor checks whether a tensor exists.
func (f *File) HasTensor(name string) bool {
	_, exists := f.tensors[name]
	return exists
}

// GetTensorInfo returns metadata for the named tensor.
func (f *File) GetTensorInfo(name string) (TensorInfo, bool) {
	ti, exists := f.tensors[name]
	return ti, exists
}

// ReadTensor loads and converts a named tensor into domain.Tensor.
func (f *File) ReadTensor(name string) (domain.Tensor, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	info, exists := f.tensors[name]
	if !exists {
		return domain.Tensor{}, fmt.Errorf("tensor %q not found in safetensors file", name)
	}

	byteStart := f.dataOffset + info.DataOffsets[0]
	byteCount := info.DataOffsets[1] - info.DataOffsets[0]
	if byteCount < 0 {
		return domain.Tensor{}, fmt.Errorf("invalid negative byte count for tensor %q", name)
	}

	buf := make([]byte, byteCount)
	if _, err := f.file.ReadAt(buf, byteStart); err != nil {
		return domain.Tensor{}, fmt.Errorf("failed to read bytes for tensor %q: %w", name, err)
	}

	switch info.DType {
	case "F32":
		elemCount := len(buf) / 4
		data := make([]float32, elemCount)
		for i := 0; i < elemCount; i++ {
			bits := binary.LittleEndian.Uint32(buf[i*4 : (i+1)*4])
			data[i] = math.Float32frombits(bits)
		}
		return domain.NewTensorFloat32(info.Shape, data)

	case "BF16":
		elemCount := len(buf) / 2
		data := make([]float32, elemCount)
		for i := 0; i < elemCount; i++ {
			b16 := binary.LittleEndian.Uint16(buf[i*2 : (i+1)*2])
			data[i] = bfloat16ToFloat32(b16)
		}
		return domain.NewTensorFloat32(info.Shape, data)

	case "F16":
		elemCount := len(buf) / 2
		data := make([]float32, elemCount)
		for i := 0; i < elemCount; i++ {
			f16 := binary.LittleEndian.Uint16(buf[i*2 : (i+1)*2])
			data[i] = float16ToFloat32(f16)
		}
		return domain.NewTensorFloat32(info.Shape, data)

	case "I64":
		elemCount := len(buf) / 8
		data := make([]int64, elemCount)
		for i := 0; i < elemCount; i++ {
			data[i] = int64(binary.LittleEndian.Uint64(buf[i*8 : (i+1)*8]))
		}
		return domain.NewTensorInt64(info.Shape, data)

	case "I32":
		elemCount := len(buf) / 4
		data := make([]int32, elemCount)
		for i := 0; i < elemCount; i++ {
			data[i] = int32(binary.LittleEndian.Uint32(buf[i*4 : (i+1)*4]))
		}
		return domain.NewTensorInt32(info.Shape, data)

	default:
		return domain.Tensor{}, fmt.Errorf("unsupported safetensors dtype %q for tensor %q", info.DType, name)
	}
}

func bfloat16ToFloat32(b uint16) float32 {
	return math.Float32frombits(uint32(b) << 16)
}

func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 0x00000001
	exp := uint32(h>>10) & 0x0000001f
	mant := uint32(h) & 0x000003ff

	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		for mant&0x00000400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= ^uint32(0x00000400)
		exp = exp + (127 - 15)
		mant = mant << 13
		return math.Float32frombits((sign << 31) | (exp << 23) | mant)
	} else if exp == 31 {
		if mant == 0 {
			return math.Float32frombits((sign << 31) | 0x7f800000)
		}
		return math.Float32frombits((sign << 31) | 0x7f800000 | (mant << 13))
	}
	exp = exp + (127 - 15)
	mant = mant << 13
	return math.Float32frombits((sign << 31) | (exp << 23) | mant)
}
