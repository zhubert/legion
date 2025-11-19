# Legion: Distributed LLM Training

> _A SETI@home for training language models - distributed pre-training across the internet_

## Overview

Legion is an experimental distributed training system that aims to enable LLM pre-training across consumer-grade machines. Inspired by SETI@home, it explores whether modern distributed training techniques (ZeRO, gradient compression, fault tolerance) can work over high-latency, low-bandwidth consumer networks.

See [PROJECT.md](PROJECT.md) for the complete project plan and technical details.

## Current Status: Phase 1.3 Complete - Real Distributed Training

Legion has completed Phase 0 (simulation) and Phase 1.3 (distributed infrastructure) with working multi-worker training:

**Phase 0 Complete:**
- ✅ Parameter partitioning (ZeRO-3 style)
- ✅ Collective communication (all-gather, reduce-scatter)
- ✅ Gradient compression (INT8 quantization)
- ✅ Network latency simulation
- ✅ End-to-end training test

**Phase 1.3 Complete:**
- ✅ Coordinator server (REST + WebSocket)
- ✅ Worker client with heartbeat and telemetry
- ✅ gRPC worker-to-worker communication
- ✅ **Real distributed training with ZeRO-3 across multiple machines**
- ✅ Gradient accumulation and synchronization
- ✅ Parameter exchange via gRPC all-gather and reduce-scatter
- ✅ Multi-worker integration tests (2+ workers verified)
- ✅ HuggingFace dataset integration (FineWeb, The Pile, Shakespeare, etc.)
- ✅ Proper data parallelism with dataset sharding
- ✅ Async collective operations for improved overlap
- ✅ Version manager for model checkpoint coordination
- ✅ Work stealing infrastructure for fault tolerance

**Next Steps (Phase 2):**
- Add compression to gRPC transfers (INT8, TopK)
- Latency measurement and regional clustering
- Enhanced fault tolerance testing (worker dropout/rejoin)
- Async gradient accumulation with variable worker participation
- Scale to 4-8 workers for performance validation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/zhubert/legion.git
cd legion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Simulation

```bash
# Run single-machine simulation with 4 workers
python sim/train.py --workers 4 --model tiny

# With latency simulation (50ms)
python sim/train.py --workers 4 --model tiny --latency 50

# With compression enabled
python sim/train.py --workers 4 --model tiny --compress int8
```

### Running Distributed Training

**Important:** The coordinator controls all training configuration (dataset, model, hyperparameters). Workers automatically fetch and execute the coordinator's decisions.

**Option 1: One-Command Orchestrator (Recommended)**
```bash
# Start all services (coordinator + 2 workers + assembler) in one terminal
python scripts/start_services.py

# With log files
python scripts/start_services.py --logs-dir logs

# Custom number of workers
python scripts/start_services.py --workers 3

# Skip assembler service
python scripts/start_services.py --no-assembler
```

This orchestrator:
- Starts coordinator, workers, and checkpoint assembler
- Color-codes output per service for easy reading
- Handles graceful shutdown with Ctrl+C
- Optionally writes logs to separate files
- Shows unified, timestamped output from all services

**Option 2: Automated 2-Worker Test**
```bash
# Terminal 1: Start coordinator (uses default config)
python -m coordinator.server

# Terminal 2: Run automated test
python scripts/test_two_workers.py
```

This script will:
- Start 2 workers automatically
- Workers fetch training config from coordinator
- Run 50 training steps with real distributed training
- Verify gradient synchronization and parameter exchange
- Report performance metrics and loss convergence

**Option 3: Manual Multi-Worker Setup with Custom Configuration**
```bash
# Terminal 1: Start coordinator
python -m coordinator.server
# Server runs on http://localhost:8000

# Terminal 2: Configure training (optional - coordinator has sensible defaults)
curl -X PUT http://localhost:8000/training/config \
  -H "Content-Type: application/json" \
  -d '{"dataset_type": "distributed_dummy", "batch_size": 4, "num_steps": 100}'

# Terminal 3: Start worker 1 (automatically fetches config from coordinator)
python -m worker.client

# Terminal 4: Start worker 2 (automatically fetches config from coordinator)
python -m worker.client
```

Workers will automatically:
- Register with the coordinator
- **Fetch training configuration from coordinator**
- Send periodic heartbeats
- Wait for peers to be ready
- Form a training cluster
- Execute training with coordinator's configuration
- Exchange parameters via gRPC
- Synchronize gradients across workers

### Training Configuration (Coordinator-Driven)

**Key Design Principle:** The coordinator makes all training decisions (dataset, model, hyperparameters). Workers simply execute the coordinator's configuration.

**Setting Training Configuration:**

```bash
# Option 1: Start coordinator with default config (distributed_dummy dataset)
python -m coordinator.server

# Option 2: Configure via API after startup
curl -X PUT http://localhost:8000/training/config \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_type": "huggingface",
    "dataset_name": "tiny_shakespeare",
    "model_size": "tiny",
    "batch_size": 8,
    "seq_len": 256,
    "num_steps": 100
  }'

# Option 3: Use Python to configure
python -c "
import requests
requests.put('http://localhost:8000/training/config', json={
    'dataset_type': 'huggingface',
    'dataset_name': 'fineweb-edu',
    'batch_size': 8,
    'seq_len': 1024,
    'num_steps': 1000
})
"
```

**Available datasets:**
- `fineweb` - 15T tokens from CommonCrawl
- `fineweb-edu` - 1.3T high-quality educational tokens
- `pile` - 825GB diverse dataset
- `tiny_shakespeare` - 1MB for testing
- `shakespeare` - Complete works of Shakespeare
- `distributed_dummy` - Synthetic data for testing (default)

**Workers automatically receive configuration from coordinator:**
```bash
# Workers no longer need dataset/model flags - they fetch config from coordinator
python -m worker.client
```

## Project Structure

```
legion/
├── PROJECT.md              # Detailed project plan
├── README.md               # This file
├── CLAUDE.md               # Development guide
├── requirements.txt        # Python dependencies
├── core/                   # Shared core functionality
│   ├── model.py            # Model definitions (TinyGPT)
│   ├── partitioner.py      # ZeRO-3 parameter partitioning
│   ├── compression.py      # Gradient compression (INT8, TopK)
│   └── dataset.py          # Dataset utilities (HuggingFace integration)
├── sim/                    # Phase 0: Single-machine simulation
│   ├── collectives.py      # Shared-memory collectives
│   ├── worker.py           # Simulated worker coordinator
│   └── train.py            # Simulation entry point
├── coordinator/            # Phase 1: Central coordinator
│   ├── server.py           # FastAPI REST + WebSocket server
│   ├── registry.py         # Worker registration and health
│   ├── clustering.py       # Latency-based regional clustering
│   ├── database.py         # SQLite persistence
│   └── version_manager.py  # Model checkpoint version tracking
├── worker/                 # Phase 1: Distributed worker nodes
│   ├── client.py           # Main worker orchestration
│   ├── coordinator_client.py  # HTTP client for coordinator
│   ├── heartbeat.py        # Periodic heartbeat manager
│   ├── trainer.py          # Distributed training loop
│   ├── shard_manager.py    # Parameter shard management
│   └── telemetry.py        # Metrics reporting
├── communication/          # Phase 1: Worker-to-worker gRPC
│   ├── grpc_server.py      # gRPC server for parameters and gradients
│   ├── grpc_client.py      # gRPC client for parameter exchange
│   ├── collectives.py      # Shared-memory collective operations
│   ├── async_collectives.py # Async all-gather/reduce-scatter
│   ├── serialization.py    # Tensor serialization/chunking
│   └── proto/              # Protocol buffer definitions
└── tests/                  # Comprehensive test suite
    ├── integration/        # Multi-worker integration tests
    ├── test_*.py           # Unit tests
    └── ...                 # 168 tests total
```

## Key Concepts

### Parameter Partitioning (ZeRO-3)

Each worker owns a subset of model parameters. During training:

- **All-gather**: Workers collect parameters from others for forward/backward pass
- **Reduce-scatter**: Gradients are aggregated and sent back to parameter owners
- **Update**: Only owners update their parameters

This reduces memory usage from `O(model_size)` to `O(model_size / num_workers)` per worker.

### Gradient Compression

Gradients are compressed before transmission:

- **INT8 quantization**: 4x compression (FP32 → INT8)
- **TopK sparsification**: Send only largest gradients
- **1-bit Adam** (planned): 32x compression after warmup
- **Target**: 64-100x total compression

### Async Collective Operations

Legion uses asynchronous collective operations for improved overlap and efficiency:

- **Async all-gather**: Non-blocking parameter collection from peers
- **Async reduce-scatter**: Overlapped gradient aggregation and distribution
- **Background I/O**: Communication overlaps with computation
- **Future-based API**: Enables pipeline parallelism across training steps

This design allows workers to hide communication latency behind computation.

### Network Architecture

**Coordinator** (FastAPI server):
- Worker registration and health monitoring
- Regional clustering based on latency
- Metrics aggregation and version tracking
- Model checkpoint coordination
- NOT in the training loop (peer-to-peer communication)

**Workers** (async Python clients):
- gRPC server for serving parameter shards
- gRPC client for fetching from peers
- Training loop with ZeRO-3 partitioning
- Automatic fault detection and recovery
- Work stealing for load balancing

## Testing

Run the comprehensive test suite:

```bash
# All tests
pytest

# With coverage
pytest --cov=sim --cov=coordinator --cov=worker --cov=communication

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/test_async_collectives.py -v
```

**Test Coverage:**
- 168 total tests (164 passing, 4 skipped)
- Unit tests for all components
- Integration tests for multi-worker scenarios
- End-to-end distributed training tests
- gRPC communication tests
- Async collectives tests
- Version manager and work stealing tests

## Contributing

This is an early-stage research project. Contributions are welcome!

See [CLAUDE.md](CLAUDE.md) for development guidance.

## License

MIT License - See [LICENSE](LICENSE)

## Performance Characteristics

Current implementation benchmarks:

**Communication:**
- gRPC parameter transfer: Supports 100MB+ messages
- Tensor serialization: NumPy-based with chunking
- Async collectives: Non-blocking all-gather and reduce-scatter

**Memory:**
- ZeRO-3 partitioning: `O(model_size / num_workers)` per worker
- Parameter shards saved to disk for checkpointing
- Gradient accumulation for large batches

**Scalability:**
- Tested: 2 workers with real distributed training (verified working)
- Ready for: 4-8 workers multi-machine
- Designed for: 8-32 workers globally
- Target: 100+ workers with regional clustering

## Resources

**Papers:**
- [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) - Parameter partitioning
- [1-bit Adam](https://arxiv.org/abs/2102.02888) - Gradient compression
- [PyTorch FSDP](https://arxiv.org/abs/2304.11277) - Async collective operations

**Documentation:**
- [DeepSpeed](https://www.deepspeed.ai/)
- [gRPC Python](https://grpc.io/docs/languages/python/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

_Let's democratize AI training, one GPU at a time._ 🚀
