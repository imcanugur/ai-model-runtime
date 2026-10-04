package safetensors

import (
	"context"
	"fmt"
	"math"
	"sync"
)

// GPT2ModelConfig defines the architectural hyperparameters of a GPT-2 model.
type GPT2ModelConfig struct {
	VocabSize    int
	NPositions   int
	NEmbd        int
	NLayer       int
	NHead        int
	HeadDim      int
	NInner       int
	LayerNormEps float32
	EOSTokenID   int64
	BOSTokenID   int64
}

// DefaultGPT2MediumConfig returns default configuration for GPT-2 Medium 350M.
func DefaultGPT2MediumConfig() GPT2ModelConfig {
	return GPT2ModelConfig{
		VocabSize:    50258,
		NPositions:   1024,
		NEmbd:        1024,
		NLayer:       24,
		NHead:        16,
		HeadDim:      64,
		NInner:       4096,
		LayerNormEps: 1e-05,
		EOSTokenID:   50257,
		BOSTokenID:   0,
	}
}

// GPT2Layer holds weight tensors for a single transformer block.
type GPT2Layer struct {
	LN1Weight      []float32 // [NEmbd]
	LN1Bias        []float32 // [NEmbd]
	CAttnWeight    []float32 // [NEmbd, 3*NEmbd]
	CAttnBias      []float32 // [3*NEmbd]
	CProjWeight    []float32 // [NEmbd, NEmbd]
	CProjBias      []float32 // [NEmbd]
	LN2Weight      []float32 // [NEmbd]
	LN2Bias        []float32 // [NEmbd]
	CFCWeight      []float32 // [NEmbd, NInner]
	CFCBias        []float32 // [NInner]
	CMLPProjWeight []float32 // [NInner, NEmbd]
	CMLPProjBias   []float32 // [NEmbd]
}

// GPT2Model holds loaded weights and executes pure-Go inference for GPT-2 models.
type GPT2Model struct {
	config    GPT2ModelConfig
	wte       []float32 // [VocabSize, NEmbd]
	wpe       []float32 // [NPositions, NEmbd]
	layers    []GPT2Layer
	lnFWeight []float32 // [NEmbd]
	lnFBias   []float32 // [NEmbd]
	mu        sync.RWMutex
}

// LoadGPT2Model extracts and validates all GPT-2 weights from the SafeTensors file.
func LoadGPT2Model(st *File, cfg GPT2ModelConfig) (*GPT2Model, error) {
	readF32 := func(name string, expectedLen int) ([]float32, error) {
		tns, err := st.ReadTensor(name)
		if err != nil {
			return nil, fmt.Errorf("missing tensor %q: %w", name, err)
		}
		if len(tns.DataF32) != expectedLen {
			return nil, fmt.Errorf("tensor %q length mismatch: got %d, expected %d", name, len(tns.DataF32), expectedLen)
		}
		return tns.DataF32, nil
	}

	d := cfg.NEmbd
	v := cfg.VocabSize
	pos := cfg.NPositions
	inner := cfg.NInner

	wte, err := readF32("transformer.wte.weight", v*d)
	if err != nil {
		return nil, err
	}

	wpe, err := readF32("transformer.wpe.weight", pos*d)
	if err != nil {
		return nil, err
	}

	lnFW, err := readF32("transformer.ln_f.weight", d)
	if err != nil {
		return nil, err
	}

	lnFB, err := readF32("transformer.ln_f.bias", d)
	if err != nil {
		return nil, err
	}

	layers := make([]GPT2Layer, cfg.NLayer)
	for i := 0; i < cfg.NLayer; i++ {
		prefix := fmt.Sprintf("transformer.h.%d", i)

		ln1W, err := readF32(prefix+".ln_1.weight", d)
		if err != nil {
			return nil, err
		}
		ln1B, err := readF32(prefix+".ln_1.bias", d)
		if err != nil {
			return nil, err
		}

		cAttnW, err := readF32(prefix+".attn.c_attn.weight", d*3*d)
		if err != nil {
			return nil, err
		}
		cAttnB, err := readF32(prefix+".attn.c_attn.bias", 3*d)
		if err != nil {
			return nil, err
		}

		cProjW, err := readF32(prefix+".attn.c_proj.weight", d*d)
		if err != nil {
			return nil, err
		}
		cProjB, err := readF32(prefix+".attn.c_proj.bias", d)
		if err != nil {
			return nil, err
		}

		ln2W, err := readF32(prefix+".ln_2.weight", d)
		if err != nil {
			return nil, err
		}
		ln2B, err := readF32(prefix+".ln_2.bias", d)
		if err != nil {
			return nil, err
		}

		cFCW, err := readF32(prefix+".mlp.c_fc.weight", d*inner)
		if err != nil {
			return nil, err
		}
		cFCB, err := readF32(prefix+".mlp.c_fc.bias", inner)
		if err != nil {
			return nil, err
		}

		cMLPW, err := readF32(prefix+".mlp.c_proj.weight", inner*d)
		if err != nil {
			return nil, err
		}
		cMLPB, err := readF32(prefix+".mlp.c_proj.bias", d)
		if err != nil {
			return nil, err
		}

		layers[i] = GPT2Layer{
			LN1Weight:      ln1W,
			LN1Bias:        ln1B,
			CAttnWeight:    cAttnW,
			CAttnBias:      cAttnB,
			CProjWeight:    cProjW,
			CProjBias:      cProjB,
			LN2Weight:      ln2W,
			LN2Bias:        ln2B,
			CFCWeight:      cFCW,
			CFCBias:        cFCB,
			CMLPProjWeight: cMLPW,
			CMLPProjBias:   cMLPB,
		}
	}

	return &GPT2Model{
		config:    cfg,
		wte:       wte,
		wpe:       wpe,
		layers:    layers,
		lnFWeight: lnFW,
		lnFBias:   lnFB,
	}, nil
}

func layerNormVec(x, gamma, beta []float32, eps float32) []float32 {
	d := len(x)
	var sum float32
	for _, v := range x {
		sum += v
	}
	mean := sum / float32(d)

	var varSum float32
	for _, v := range x {
		diff := v - mean
		varSum += diff * diff
	}
	variance := varSum / float32(d)
	invStd := float32(1.0 / math.Sqrt(float64(variance+eps)))

	out := make([]float32, d)
	for i := 0; i < d; i++ {
		out[i] = (x[i]-mean)*invStd*gamma[i] + beta[i]
	}
	return out
}

func matmulVec(x, W, b []float32, inDim, outDim int) []float32 {
	out := make([]float32, outDim)
	chunk := 128
	if outDim < chunk {
		for j := 0; j < outDim; j++ {
			var sum float32
			if b != nil {
				sum = b[j]
			}
			for k := 0; k < inDim; k++ {
				sum += x[k] * W[k*outDim+j]
			}
			out[j] = sum
		}
		return out
	}

	var wg sync.WaitGroup
	for start := 0; start < outDim; start += chunk {
		end := start + chunk
		if end > outDim {
			end = outDim
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for j := s; j < e; j++ {
				var sum float32
				if b != nil {
					sum = b[j]
				}
				for k := 0; k < inDim; k++ {
					sum += x[k] * W[k*outDim+j]
				}
				out[j] = sum
			}
		}(start, end)
	}
	wg.Wait()
	return out
}

func geluNewFast(x float32) float32 {
	const sqrt2OverPi = 0.7978845608
	x3 := x * x * x
	inner := sqrt2OverPi * (x + 0.044715*x3)
	return 0.5 * x * float32(1.0+math.Tanh(float64(inner)))
}

// LayerKVCache maintains cached Key and Value vectors across generation steps.
type LayerKVCache struct {
	Keys   [][]float32 // [tokens, NEmbd]
	Values [][]float32 // [tokens, NEmbd]
}

// GenerateAutoregressive performs authentic step-by-step token generation.
func (m *GPT2Model) GenerateAutoregressive(ctx context.Context, promptTokens []int64, maxNewTokens int) ([]int64, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(promptTokens) == 0 {
		promptTokens = []int64{m.config.BOSTokenID}
	}
	if maxNewTokens <= 0 {
		maxNewTokens = 64
	}

	d := m.config.NEmbd
	numHeads := m.config.NHead
	headDim := m.config.HeadDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	kvCache := make([]LayerKVCache, m.config.NLayer)
	for i := range kvCache {
		kvCache[i] = LayerKVCache{
			Keys:   make([][]float32, 0, len(promptTokens)+maxNewTokens),
			Values: make([][]float32, 0, len(promptTokens)+maxNewTokens),
		}
	}

	// Helper to run a single token position through all layers, updating KV cache
	forwardToken := func(tok int64, pos int) []float32 {
		// Embedding
		x := make([]float32, d)
		wteOffset := int(tok) * d
		wpeOffset := pos * d
		for i := 0; i < d; i++ {
			x[i] = m.wte[wteOffset+i] + m.wpe[wpeOffset+i]
		}

		for l := 0; l < m.config.NLayer; l++ {
			layer := &m.layers[l]

			// Attention LayerNorm
			ln1 := layerNormVec(x, layer.LN1Weight, layer.LN1Bias, m.config.LayerNormEps)

			// QKV projection
			qkv := matmulVec(ln1, layer.CAttnWeight, layer.CAttnBias, d, 3*d)

			qVec := qkv[:d]
			kVec := qkv[d : 2*d]
			vVec := qkv[2*d : 3*d]

			// Append new K, V to layer cache
			kStored := make([]float32, d)
			copy(kStored, kVec)
			vStored := make([]float32, d)
			copy(vStored, vVec)
			kvCache[l].Keys = append(kvCache[l].Keys, kStored)
			kvCache[l].Values = append(kvCache[l].Values, vStored)

			totalTokens := len(kvCache[l].Keys)

			// Multi-head attention across all cached positions
			attnVec := make([]float32, d)
			for h := 0; h < numHeads; h++ {
				qHead := qVec[h*headDim : (h+1)*headDim]

				// Compute scores with all cached keys
				scores := make([]float32, totalTokens)
				var maxScore float32 = -1e9
				for s := 0; s < totalTokens; s++ {
					kHead := kvCache[l].Keys[s][h*headDim : (h+1)*headDim]
					var dot float32
					for k := 0; k < headDim; k++ {
						dot += qHead[k] * kHead[k]
					}
					score := dot * scale
					scores[s] = score
					if score > maxScore {
						maxScore = score
					}
				}

				// Softmax
				var sumExp float32
				for s := 0; s < totalTokens; s++ {
					scores[s] = float32(math.Exp(float64(scores[s] - maxScore)))
					sumExp += scores[s]
				}
				for s := 0; s < totalTokens; s++ {
					scores[s] /= sumExp
				}

				// Weighted sum of values
				for k := 0; k < headDim; k++ {
					var vSum float32
					for s := 0; s < totalTokens; s++ {
						vSum += scores[s] * kvCache[l].Values[s][h*headDim+k]
					}
					attnVec[h*headDim+k] = vSum
				}
			}

			// Attention projection + residual
			attnProj := matmulVec(attnVec, layer.CProjWeight, layer.CProjBias, d, d)
			for i := 0; i < d; i++ {
				x[i] += attnProj[i]
			}

			// MLP
			ln2 := layerNormVec(x, layer.LN2Weight, layer.LN2Bias, m.config.LayerNormEps)
			fc := matmulVec(ln2, layer.CFCWeight, layer.CFCBias, d, m.config.NInner)
			for i := range fc {
				fc[i] = geluNewFast(fc[i])
			}
			mlpProj := matmulVec(fc, layer.CMLPProjWeight, layer.CMLPProjBias, m.config.NInner, d)
			for i := 0; i < d; i++ {
				x[i] += mlpProj[i]
			}
		}

		// Final LayerNorm
		lastH := layerNormVec(x, m.lnFWeight, m.lnFBias, m.config.LayerNormEps)
		return lastH
	}

	// Compute LM Head logits given hidden state
	computeLogits := func(lastH []float32) (int64, []float32) {
		vocabSize := m.config.VocabSize
		logits := make([]float32, vocabSize)
		var maxLogit float32 = -1e9
		var bestToken int64 = 0

		// Matmul against wte
		var wg sync.WaitGroup
		chunk := 2048
		type partialResult struct {
			bestTok int64
			maxVal  float32
		}
		results := make([]partialResult, (vocabSize+chunk-1)/chunk)

		for idx, start := 0, 0; start < vocabSize; idx, start = idx+1, start+chunk {
			end := start + chunk
			if end > vocabSize {
				end = vocabSize
			}
			wg.Add(1)
			go func(iIdx, s, e int) {
				defer wg.Done()
				var localMax float32 = -1e9
				var localBest int64 = int64(s)
				for v := s; v < e; v++ {
					wteOffset := v * d
					var dot float32
					for k := 0; k < d; k++ {
						dot += lastH[k] * m.wte[wteOffset+k]
					}
					logits[v] = dot
					if dot > localMax {
						localMax = dot
						localBest = int64(v)
					}
				}
				results[iIdx] = partialResult{bestTok: localBest, maxVal: localMax}
			}(idx, start, end)
		}
		wg.Wait()

		for _, r := range results {
			if r.maxVal > maxLogit {
				maxLogit = r.maxVal
				bestToken = r.bestTok
			}
		}

		return bestToken, logits
	}

	// Step 1: Ingest prompt tokens into KV cache
	currentPos := 0
	var lastH []float32
	for _, tok := range promptTokens {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		lastH = forwardToken(tok, currentPos)
		currentPos++
	}

	// First next-token prediction
	bestTok, _ := computeLogits(lastH)

	generatedTokens := make([]int64, 0, maxNewTokens)

	// Step 2: Autoregressive token-by-token loop
	for step := 0; step < maxNewTokens; step++ {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		// Stop conditions: EOS tokens (<|stop|>=50257, <|endoftext|>=0)
		if bestTok == m.config.EOSTokenID || bestTok == 0 {
			break
		}

		generatedTokens = append(generatedTokens, bestTok)
		if len(generatedTokens) >= maxNewTokens {
			break
		}

		// Forward newly generated token with cached history
		lastH = forwardToken(bestTok, currentPos)
		currentPos++

		bestTok, _ = computeLogits(lastH)
	}

	return generatedTokens, nil
}
