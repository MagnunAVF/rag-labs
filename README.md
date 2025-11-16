# RAG Labs

A production-ready Retrieval-Augmented Generation (RAG) system built with Go, featuring vector search, embeddings, and LLM integration.

## Overview

RAG Labs is a complete RAG pipeline that combines:
- **Vector Database**: Weaviate for efficient similarity search
- **Embeddings**: Text Embeddings Inference (TEI) with mixedbread-ai/mxbai-embed-large-v1
- **LLM**: vLLM serving microsoft/Phi-3-mini-128k-instruct
- **API**: Go-based REST API using Fiber framework

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│   Client    │─────▶│   RAG API    │─────▶│   Weaviate   │
│             │      │   (Go/Fiber) │      │  (Vectors)   │
└─────────────┘      └──────────────┘      └──────────────┘
                            │                      ▲
                            │                      │
                            ▼                      │
                     ┌──────────────┐      ┌──────────────┐
                     │     vLLM     │      │     TEI      │
                     │  (Phi-3 LLM) │      │ (Embeddings) │
                     └──────────────┘      └──────────────┘
```

## Features

- **GPU-Accelerated**: All ML services leverage NVIDIA GPUs
- **Docker Compose**: Single-command deployment of all services
- **Document Ingestion**: Python scripts for data preparation and indexing
- **REST API**: Clean Go API with health checks and query endpoints
- **Production Ready**: Includes health checks, retries, and proper error handling

## Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with Docker GPU support
- Hugging Face account (for model access)
- Python 3.8+ (for data ingestion)
- Go 1.25+ (for API development)

## Quick Start

### 1. Set Environment Variables

```bash
export HF_TOKEN='your_huggingface_token'
```

### 2. Start All Services

```bash
docker compose up -d
```

This will start:
- **Weaviate** on `localhost:8080`
- **TEI** on `localhost:8082`
- **vLLM** on `localhost:8000`
- **RAG API** on `localhost:8081`

### 3. Prepare Data (Optional)

Download and transform the movies dataset:

```bash
cd data
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python get_and_transform.py
```

### 4. Ingest Documents

```bash
python ingest.py
```

This will:
- Connect to Weaviate
- Load documents from `data/texts/`
- Generate embeddings via TEI
- Store vectors in Weaviate

### 5. Query the API

```bash
curl -X POST http://localhost:8081/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about science fiction movies"}'
```

## Project Structure

```
rag-labs/
├── api/                    # Go API service
│   ├── config/            # Configuration management
│   ├── handlers/          # HTTP request handlers
│   ├── models/            # Data models
│   ├── services/          # Business logic (RAG, TEI, vLLM, Weaviate)
│   ├── server/            # Server setup
│   ├── Dockerfile         # API container
│   └── main.go            # Entry point
├── data/                   # Data ingestion scripts
│   ├── get_and_transform.py  # Download & prepare dataset
│   ├── ingest.py          # Index documents to Weaviate
│   ├── requirements.txt   # Python dependencies
│   └── texts/             # Document storage
├── scripts/                # Utility scripts
│   ├── create-py-env.sh   # Python environment setup
│   ├── run_tei.sh         # Standalone TEI launcher
│   └── run_vllm.sh        # Standalone vLLM launcher
└── docker-compose.yml      # Multi-service orchestration
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Query
```bash
POST /query
Content-Type: application/json

{
  "query": "your question here"
}
```

Response:
```json
{
  "response": "AI-generated answer based on retrieved context"
}
```

## Configuration

### Environment Variables

The RAG API supports the following environment variables:

- `WEAVIATE_HOST`: Weaviate host (default: `weaviate:8080`)
- `WEAVIATE_SCHEME`: HTTP scheme (default: `http`)
- `TEI_BASE_URL`: Text Embeddings Inference URL (default: `http://tei:80`)
- `VLLM_BASE_URL`: vLLM API URL (default: `http://vllm:8000/v1`)
- `VLLM_MODEL_NAME`: LLM model name (default: `microsoft/Phi-3-mini-128k-instruct`)
- `COLLECTION_NAME`: Weaviate collection name (default: `LlamaIndex`)
- `SERVER_PORT`: API server port (default: `:8081`)
- `HF_TOKEN`: Hugging Face API token (required for model downloads)

### Ingestion Configuration

Edit `data/ingest.py` to customize:
- `CHUNK_SIZE`: Token chunk size (default: 512)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 20)
- `COLLECTION_NAME`: Target Weaviate collection

## Development

### Running the API Locally

```bash
cd api
go mod download
go run main.go
```

### Building the API

```bash
cd api
go build -o rag-api
./rag-api
```

### Running Individual Services

If you prefer to run services separately:

```bash
# TEI
./scripts/run_tei.sh

# vLLM
./scripts/run_vllm.sh

# Weaviate
docker compose up weaviate -d
```

## Tech Stack

### Backend
- **Go 1.25**: High-performance API
- **Fiber v2**: Fast HTTP framework
- **Weaviate Go Client**: Vector database integration

### ML/AI
- **Weaviate**: Vector database
- **Text Embeddings Inference**: HuggingFace embedding service
- **vLLM**: High-throughput LLM inference
- **LlamaIndex**: Document processing and indexing

### Infrastructure
- **Docker & Docker Compose**: Containerization
- **NVIDIA Container Toolkit**: GPU support

## Troubleshooting

### Services Not Starting

Check service logs:
```bash
docker compose logs -f [service_name]
```

### Weaviate Connection Issues

Restart Weaviate:
```bash
docker compose restart weaviate
```

### GPU Not Detected

Ensure NVIDIA Container Toolkit is installed:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Out of Memory

Adjust vLLM GPU memory utilization in `docker-compose.yml`:
```yaml
--gpu-memory-utilization 0.6  # Reduce from 0.8
```

## License

This project is for educational and research purposes.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.