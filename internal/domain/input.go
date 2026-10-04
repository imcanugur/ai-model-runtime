package domain

// ValueType enumerates primitive types for schema-safe flexible fields.
type ValueType string

const (
	TypeString ValueType = "string"
	TypeFloat  ValueType = "float"
	TypeInt    ValueType = "int"
	TypeBool   ValueType = "bool"
	TypeBytes  ValueType = "bytes"
	TypeSlice  ValueType = "slice"
)

// TypedValue holds a strongly-typed value without losing type safety.
type TypedValue struct {
	Type      ValueType `json:"type"`
	StringVal string    `json:"string_val,omitempty"`
	FloatVal  float64   `json:"float_val,omitempty"`
	IntVal    int64     `json:"int_val,omitempty"`
	BoolVal   bool      `json:"bool_val,omitempty"`
	BytesVal  []byte    `json:"bytes_val,omitempty"`
	SliceVal  []string  `json:"slice_val,omitempty"`
}

// StringValue creates a string typed value.
func StringValue(s string) TypedValue { return TypedValue{Type: TypeString, StringVal: s} }

// FloatValue creates a float typed value.
func FloatValue(f float64) TypedValue { return TypedValue{Type: TypeFloat, FloatVal: f} }

// IntValue creates an int typed value.
func IntValue(i int64) TypedValue { return TypedValue{Type: TypeInt, IntVal: i} }

// BoolValue creates a bool typed value.
func BoolValue(b bool) TypedValue { return TypedValue{Type: TypeBool, BoolVal: b} }

// BytesValue creates a bytes typed value.
func BytesValue(b []byte) TypedValue { return TypedValue{Type: TypeBytes, BytesVal: b} }

// SliceValue creates a slice typed value.
func SliceValue(s []string) TypedValue { return TypedValue{Type: TypeSlice, SliceVal: s} }

// InputDocument represents a single structured document, passage, or sentence.
type InputDocument struct {
	ID       string            `json:"id,omitempty"`
	Text     string            `json:"text"`
	Language string            `json:"language,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// InferenceInput represents universal model input across any modality or task.
type InferenceInput struct {
	Documents  []InputDocument       `json:"documents,omitempty"`
	Fields     map[string]TypedValue `json:"fields,omitempty"`
	Parameters map[string]string     `json:"parameters,omitempty"`
}

// GetText returns the primary text document or field.
func (in *InferenceInput) GetText() string {
	if len(in.Documents) > 0 && in.Documents[0].Text != "" {
		return in.Documents[0].Text
	}
	if in.Fields != nil {
		if v, ok := in.Fields["text"]; ok && v.Type == TypeString {
			return v.StringVal
		}
	}
	return ""
}

// GetContext returns the context document or field.
func (in *InferenceInput) GetContext() string {
	if in.Fields != nil {
		if v, ok := in.Fields["context"]; ok && v.Type == TypeString {
			return v.StringVal
		}
	}
	for _, doc := range in.Documents {
		if doc.ID == "context" {
			return doc.Text
		}
	}
	if len(in.Documents) > 0 {
		return in.Documents[0].Text
	}
	return ""
}

// GetQuestion returns the question field if present.
func (in *InferenceInput) GetQuestion() string {
	if in.Fields != nil {
		if v, ok := in.Fields["question"]; ok && v.Type == TypeString {
			return v.StringVal
		}
	}
	for _, doc := range in.Documents {
		if doc.ID == "question" {
			return doc.Text
		}
	}
	return ""
}

// ToRawInput converts universal InferenceInput into a backwards-compatible RawInput.
func (in *InferenceInput) ToRawInput() RawInput {
	raw := RawInput{
		Text:       in.GetText(),
		Context:    in.GetContext(),
		Question:   in.GetQuestion(),
		Parameters: in.Parameters,
	}
	if in.Fields != nil {
		if v, ok := in.Fields["language"]; ok && v.Type == TypeString {
			raw.Language = v.StringVal
		}
	}
	if raw.Language == "" && len(in.Documents) > 0 {
		raw.Language = in.Documents[0].Language
	}
	return raw
}

// RawInput represents the raw incoming payload from an HTTP request or asynchronous queue job.
type RawInput struct {
	Text       string            `json:"text"`
	Context    string            `json:"context,omitempty"`
	Question   string            `json:"question,omitempty"`
	Language   string            `json:"language,omitempty"`
	Parameters map[string]string `json:"parameters,omitempty"`
}

// ToInferenceInput converts a backwards-compatible RawInput into universal InferenceInput.
func (r RawInput) ToInferenceInput() InferenceInput {
	fields := make(map[string]TypedValue)
	if r.Text != "" {
		fields["text"] = StringValue(r.Text)
	}
	if r.Context != "" {
		fields["context"] = StringValue(r.Context)
	}
	if r.Question != "" {
		fields["question"] = StringValue(r.Question)
	}
	if r.Language != "" {
		fields["language"] = StringValue(r.Language)
	}

	var docs []InputDocument
	if r.Context != "" {
		docs = append(docs, InputDocument{
			ID:       "context",
			Text:     r.Context,
			Language: r.Language,
		})
	}
	if r.Question != "" {
		docs = append(docs, InputDocument{
			ID:       "question",
			Text:     r.Question,
			Language: r.Language,
		})
	}
	if r.Text != "" && r.Context == "" {
		docs = append(docs, InputDocument{
			ID:       "text",
			Text:     r.Text,
			Language: r.Language,
		})
	}

	return InferenceInput{
		Documents:  docs,
		Fields:     fields,
		Parameters: r.Parameters,
	}
}

// Sentence represents a single segmented sentence with boundary offsets.
type Sentence struct {
	Index int    `json:"index"`
	Text  string `json:"text"`
	Start int    `json:"start"`
	End   int    `json:"end"`
}

// TextDocument represents intermediate text state after normalization and optional sentence splitting.
type TextDocument struct {
	RawText        string     `json:"raw_text"`
	NormalizedText string     `json:"normalized_text"`
	Sentences      []Sentence `json:"sentences,omitempty"`
}

// TokenizedBatch represents tokenized sequences ready for tensor packing.
type TokenizedBatch struct {
	InputIDs       []int64 `json:"input_ids"`
	AttentionMask  []int64 `json:"attention_mask,omitempty"`
	TokenCounts    []int   `json:"token_counts,omitempty"`
	BatchSize      int     `json:"batch_size"`
	SequenceLength int     `json:"sequence_length"`
}
