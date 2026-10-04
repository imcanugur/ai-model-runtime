# AI Model Runtime

A model-agnostic AI model runtime and serving platform written in Go.

Repository: `github.com/imcanugur/ai-model-runtime`

[![Go Version](https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat&logo=go)](https://golang.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

**AI Model Runtime** is a high-performance, model-agnostic serving engine and execution runtime written in Go. It enables hosting, executing, and serving machine learning models (NLP, text generation, embeddings, classification, and question answering) through a unified, standardized HTTP API contract.

### Key Capabilities

- **Unified Universal API**: One consistent endpoint contract for all model architectures (`/v1/models/{modelID}/predict`).
- **Pure Go SafeTensors Runtime**: Native Go execution for HuggingFace SafeTensors models (including GPT-2 generative inference with KV cache, logits processing, and greedy/top-k sampling) without external Python/CGO dependencies.
- **Pluggable Multi-Runtime Architecture**: Pluggable backend providers (`safetensors`, `onnx`, `mock`).
- **Robust Pipeline System**: Configurable preprocessing and postprocessing stages (normalizers, chunkers, tokenizers, calibrations).
- **Asynchronous Execution**: Redis-backed distributed task queue and worker pools with stateful JobStore tracking and crash recovery.
- **Built-in Developer Console**: Embedded interactive Swagger/OpenAPI documentation and playground for immediate model experimentation.

---

## Architecture

```text
+-------------------------------------------------------------------------+
|                           AI Model Runtime                              |
|                                                                         |
|   HTTP Request -> Unified Router -> Pipeline Stages -> Runtime Provider |
|                                                                         |
|   [Sync Predict]                     [Tokenization]        [SafeTensors] |
|   [Async Job Queue]                  [Chunking]            [ONNX]        |
|   [Discovery & Status]               [Calibration]         [Mock]        |
+-------------------------------------------------------------------------+
             |                                    |
             v                                    v
   +--------------------+               +--------------------+
   |  Redis Job Queue   |               | Model Repository   |
   |  & JobStore DB     |               | (model.yaml)       |
   +--------------------+               +--------------------+
```

---

## Quick Start

### Prerequisites

- **Go 1.22+**
- (Optional) **Redis** for distributed asynchronous job queue

### 1. Build and Run

```bash
# Clone repository
git clone https://github.com/imcanugur/ai-model-runtime.git
cd ai-model-runtime

# Start the runtime server
go run cmd/server/main.go
```

By default, the server starts on `http://0.0.0.0:8080`.

### 2. Access Developer Console

Open your browser and navigate to:
```text
http://127.0.0.1:8080/swagger/
```

---

## API Usage Examples

### 1. Synchronous Model Prediction

Execute synchronous inference on a loaded model:

```bash
curl -X POST http://127.0.0.1:8080/v1/models/text-generation-tr/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Yapay zeka modelleri hakkında bilgi ver."
    },
    "options": {
      "max_tokens": 64,
      "temperature": 0.7
    }
  }'
```

**Standard Response:**

```json
{
  "data": {
    "value": "Yapay zeka modelleri..."
  },
  "meta": {
    "request_id": "req-1726830000-abc",
    "model": {
      "id": "text-generation-tr",
      "version": "1.0.0"
    },
    "duration_ms": 28
  }
}
```

### 2. List Registered Models

```bash
curl http://127.0.0.1:8080/v1/models
```

### 3. Asynchronous Job Submission

Submit an asynchronous background inference task:

```bash
curl -X POST http://127.0.0.1:8080/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-generation-tr",
    "version": "1.0.0",
    "input": {
      "text": "Asenkron arka plan görev testi."
    },
    "max_attempts": 3
  }'
```

**Poll Job Status:**

```bash
curl http://127.0.0.1:8080/v1/jobs/{job_id}
```

---

## Model Manifest Specification

Each model directory contains a `model.yaml` describing its runtime, resources, and pipeline:

```yaml
schema_version: "v1"
id: "text-generation-tr"
name: "Turkish GPT-2 Generation Model"
version: "1.0.0"

metadata:
  task: "text-generation"
  languages: ["tr"]
  description: "Pure-Go HuggingFace SafeTensors generative model"
  author: "AI Model Runtime Team"

resources:
  max_concurrency: 4
  timeout_ms: 30000
  min_memory_mb: 1024

runtime:
  provider: "safetensors"
  profile: "native_go"
  entrypoint: "model.safetensors"
```

---

## Configuration & Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AI_MODEL_RUNTIME_HTTP_PORT` | `8080` | Port for HTTP server |
| `AI_MODEL_RUNTIME_MODELS_DIR` | `./models` | Directory holding model packages |
| `AI_MODEL_RUNTIME_REDIS_ADDR` | `127.0.0.1:6379` | Redis host:port for async queue |
| `AI_MODEL_RUNTIME_WORKERS` | `4` | Number of worker goroutines |

---

## Testing & Quality Assurance

Run the comprehensive test suite with data race detection:

```bash
# Format check
go fmt ./...

# Static analysis
go vet ./...

# Unit and integration tests with race detection
go test -v -race ./...
```

---

## 👨‍💻 Author & Vision

**Can Ugur**  
*Full-stack Architect & Open Source Craftsman*  

[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=flat&logo=github)](https://github.com/imcanugur) 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/can-ugur)

> *"Stop configuring. Start building."*

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="center">
  <i>Made with ☕ and passion in the dev cave.</i>
</p>