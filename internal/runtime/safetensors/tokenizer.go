package safetensors

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
)

type addedToken struct {
	ID      int64  `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

type tokenizerModelJSON struct {
	Type  string           `json:"type"`
	Vocab map[string]int64 `json:"vocab"`
}

type tokenizerJSON struct {
	AddedTokens []addedToken       `json:"added_tokens"`
	Model       tokenizerModelJSON `json:"model"`
}

// Tokenizer provides pure-Go tokenization and decoding supporting ByteLevel BPE and WordPiece.
type Tokenizer struct {
	isBPE        bool
	bpeRanks     map[string]int
	b2r          map[byte]rune
	r2b          map[rune]byte
	re           *regexp.Regexp
	vocab        map[string]int64
	reverseVocab map[int64]string
	specialIDs   map[int64]bool
	eosID        int64
	padID        int64
	mu           sync.RWMutex
}

// buildByteToRune builds the standard GPT-2 byte-to-unicode bidirectional mapping table.
func buildByteToRune() (map[byte]rune, map[rune]byte) {
	b2r := make(map[byte]rune, 256)
	r2b := make(map[rune]byte, 256)

	var bs []int
	for b := int('!'); b <= int('~'); b++ {
		bs = append(bs, b)
	}
	for b := 161; b <= 172; b++ {
		bs = append(bs, b)
	}
	for b := 174; b <= 255; b++ {
		bs = append(bs, b)
	}

	cs := make([]int, len(bs))
	copy(cs, bs)

	n := 0
	inBs := make(map[int]bool, 256)
	for _, b := range bs {
		inBs[b] = true
	}

	for b := 0; b < 256; b++ {
		if !inBs[b] {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}

	for i := 0; i < len(bs); i++ {
		b := byte(bs[i])
		r := rune(cs[i])
		b2r[b] = r
		r2b[r] = b
	}

	return b2r, r2b
}

// LoadTokenizer parses a HuggingFace tokenizer.json file, and detects BPE merges if available.
func LoadTokenizer(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read tokenizer file %q: %w", path, err)
	}

	var raw tokenizerJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to parse tokenizer JSON %q: %w", path, err)
	}

	vocab := make(map[string]int64, len(raw.Model.Vocab)+len(raw.AddedTokens))
	reverse := make(map[int64]string, len(raw.Model.Vocab)+len(raw.AddedTokens))
	special := make(map[int64]bool)

	for k, v := range raw.Model.Vocab {
		vocab[k] = v
		reverse[v] = k
	}

	var eosID int64 = -1
	var padID int64 = -1

	for _, at := range raw.AddedTokens {
		vocab[at.Content] = at.ID
		reverse[at.ID] = at.Content
		if at.Special {
			special[at.ID] = true
		}
		if at.Content == "<|endoftext|>" || at.Content == "</s>" || at.Content == "<|stop|>" {
			if eosID == -1 || at.Content == "<|stop|>" {
				eosID = at.ID
			}
			if padID == -1 {
				padID = at.ID
			}
		}
	}

	modelDir := filepath.Dir(path)
	mergesPath := filepath.Join(modelDir, "merges.txt")

	bpeRanks := make(map[string]int)
	isBPE := false

	// Check for merges.txt
	if _, statErr := os.Stat(mergesPath); statErr == nil {
		if f, openErr := os.Open(mergesPath); openErr == nil {
			scanner := bufio.NewScanner(f)
			rank := 0
			for scanner.Scan() {
				line := strings.TrimSpace(scanner.Text())
				if strings.HasPrefix(line, "#") || line == "" {
					continue
				}
				bpeRanks[line] = rank
				rank++
			}
			_ = f.Close()
			if rank > 0 {
				isBPE = true
			}
		}
	}

	var b2r map[byte]rune
	var r2b map[rune]byte
	var re *regexp.Regexp
	if isBPE {
		b2r, r2b = buildByteToRune()
		re = regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`)
	}

	return &Tokenizer{
		isBPE:        isBPE,
		bpeRanks:     bpeRanks,
		b2r:          b2r,
		r2b:          r2b,
		re:           re,
		vocab:        vocab,
		reverseVocab: reverse,
		specialIDs:   special,
		eosID:        eosID,
		padID:        padID,
	}, nil
}

// bpe applies iterative byte-pair merge operations on a subword token.
func (t *Tokenizer) bpe(token string) []string {
	runes := []rune(token)
	if len(runes) <= 1 {
		return []string{token}
	}
	word := make([]string, len(runes))
	for i, r := range runes {
		word[i] = string(r)
	}

	for {
		if len(word) <= 1 {
			break
		}
		minRank := 1 << 30
		minIdx := -1
		for i := 0; i < len(word)-1; i++ {
			pair := word[i] + " " + word[i+1]
			if r, ok := t.bpeRanks[pair]; ok && r < minRank {
				minRank = r
				minIdx = i
			}
		}
		if minIdx == -1 {
			break
		}
		newWord := make([]string, 0, len(word)-1)
		for i := 0; i < len(word); {
			if i == minIdx {
				newWord = append(newWord, word[i]+word[i+1])
				i += 2
			} else {
				newWord = append(newWord, word[i])
				i++
			}
		}
		word = newWord
	}
	return word
}

// Encode converts input text into a sequence of token IDs.
func (t *Tokenizer) Encode(text string) []int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if len(t.vocab) == 0 {
		return nil
	}

	if t.isBPE && t.re != nil && t.b2r != nil {
		matches := t.re.FindAllString(text, -1)
		var tokens []int64
		for _, match := range matches {
			matchBytes := []byte(match)
			byteRunes := make([]rune, len(matchBytes))
			for i, b := range matchBytes {
				byteRunes[i] = t.b2r[b]
			}
			subTokens := t.bpe(string(byteRunes))
			for _, st := range subTokens {
				if id, ok := t.vocab[st]; ok {
					tokens = append(tokens, id)
				}
			}
		}
		return tokens
	}

	// Fallback character/prefix matcher
	if _, hasG := t.vocab["Ġ"]; hasG {
		text = strings.ReplaceAll(text, " ", "Ġ")
	}

	var ids []int64
	runes := []rune(text)
	n := len(runes)

	for i := 0; i < n; {
		matched := false
		maxLen := 32
		if i+maxLen > n {
			maxLen = n - i
		}

		for l := maxLen; l >= 1; l-- {
			sub := string(runes[i : i+l])
			if id, ok := t.vocab[sub]; ok {
				ids = append(ids, id)
				i += l
				matched = true
				break
			}
		}

		if !matched {
			sub := string(runes[i : i+1])
			if id, ok := t.vocab[sub]; ok {
				ids = append(ids, id)
			} else if unk, ok := t.vocab["<unk>"]; ok {
				ids = append(ids, unk)
			}
			i++
		}
	}

	return ids
}

// Decode converts token IDs back into text.
func (t *Tokenizer) Decode(tokens []int64, skipSpecial bool) string {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.isBPE && t.r2b != nil {
		var byteRunes []rune
		for _, id := range tokens {
			if skipSpecial && t.specialIDs[id] {
				continue
			}
			if s, ok := t.reverseVocab[id]; ok {
				byteRunes = append(byteRunes, []rune(s)...)
			}
		}
		bytes := make([]byte, 0, len(byteRunes))
		for _, r := range byteRunes {
			if b, ok := t.r2b[r]; ok {
				bytes = append(bytes, b)
			}
		}
		return string(bytes)
	}

	var sb strings.Builder
	for _, id := range tokens {
		if skipSpecial && t.specialIDs[id] {
			continue
		}
		if tok, ok := t.reverseVocab[id]; ok {
			tok = strings.ReplaceAll(tok, "Ġ", " ")
			tok = strings.ReplaceAll(tok, " ", " ")
			sb.WriteString(tok)
		}
	}

	return strings.TrimSpace(sb.String())
}

// EOSID returns the end-of-sequence token ID.
func (t *Tokenizer) EOSID() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.eosID
}

// PadID returns the padding token ID.
func (t *Tokenizer) PadID() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.padID
}
