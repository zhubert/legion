# Legion: Distributed LLM Training

> _A SETI@home for training language models - distributed pre-training across the internet_

## ⚠️ PROJECT STATUS: EARLY EXPERIMENTAL RESEARCH - NOT READY FOR USE

**This project is in active early-stage development and is NOT functional for real-world use.**

Legion is a research prototype exploring whether modern distributed training techniques (ZeRO, gradient compression) can work over high-latency consumer internet connections. Most components are incomplete, untested at scale, or purely theoretical.

**Do not attempt to use this for actual model training.** This is research code to validate concepts, not production software.

See [PROJECT.md](PROJECT.md) for the complete project plan and technical details.

## What's Built vs What's Not

**Works (Single Machine Only):**
- ✅ Single-machine simulation with fake "workers" (threads/processes)
- ✅ Parameter partitioning (ZeRO-3 style) - simulated in-memory only
- ✅ Collective communication (all-gather, reduce-scatter) - in-memory only
- ✅ Tiny transformer model for testing

**Partially Exists (Untested/Incomplete):**
- ⚠️ Coordinator server (REST + WebSocket) - basic skeleton, minimal testing
- ⚠️ Worker client with heartbeat - exists but brittle
- ⚠️ gRPC worker-to-worker communication - prototype only, not production-ready
- ⚠️ Gradient compression (INT8 quantization) - implemented but not integrated
- ⚠️ HuggingFace dataset integration - basic loader exists, sharding untested
- ⚠️ Network latency simulation - artificial delays only, not real networks

**Doesn't Work / Not Built:**
- ❌ Real distributed training across machines - **NOT TESTED AT SCALE**
- ❌ Gradient synchronization across internet - **THEORETICAL ONLY**
- ❌ Parameter exchange over real networks - **UNTESTED WITH REAL LATENCY**
- ❌ Multi-worker training on separate machines - **MAY NOT WORK**
- ❌ Proper data parallelism - **NOT VALIDATED**
- ❌ Async collective operations - **EXPERIMENTAL, LIKELY BROKEN**
- ❌ Fault tolerance (worker dropout/rejoin) - **NOT IMPLEMENTED**
- ❌ Work stealing - **SKELETON CODE ONLY**
- ❌ Regional clustering based on latency - **NOT IMPLEMENTED**
- ❌ Security (authentication, encryption) - **NOT IMPLEMENTED**
- ❌ Performance optimization for internet - **NOT IMPLEMENTED**

**Bottom Line:**
The simulation works on a single machine. Everything else is incomplete, untested, or purely theoretical. Distributed training across the internet has not been validated.

## Quick Start (Local Experimentation Only)

**Warning:** These instructions are for local development and testing only. Do not expect production-quality distributed training.

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

### Running Distributed Training (Experimental - Likely Broken)

**CRITICAL WARNING:** Distributed training is experimental prototype code. It may work on localhost but has NOT been validated on real distributed networks. Expect bugs, crashes, and data loss.

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

This is an **extremely early-stage research prototype**. The codebase is unstable, incomplete, and poorly tested.

If you're interested in contributing:
1. Understand this is experimental research code, not production software
2. Most features are incomplete or broken
3. Breaking changes happen frequently
4. Documentation may be outdated or aspirational

See [CLAUDE.md](CLAUDE.md) for development guidance and [PROJECT.md](PROJECT.md) for the (ambitious) roadmap.

## License

MIT License - See [LICENSE](LICENSE)

## Performance Characteristics (Theoretical/Untested)

**Warning:** These are design goals and minimal local testing, NOT production benchmarks.

**Communication:**
- gRPC parameter transfer: Theoretically supports 100MB+ messages (untested at scale)
- Tensor serialization: Basic NumPy-based with chunking (not optimized)
- Async collectives: Experimental non-blocking all-gather and reduce-scatter (likely broken)

**Memory:**
- ZeRO-3 partitioning: `O(model_size / num_workers)` per worker
- Parameter shards saved to disk for checkpointing
- Gradient accumulation for large batches

**Scalability:**
- Actually Tested: Single machine simulation only
- Maybe Works: 2 workers on localhost (not verified on separate machines)
- Aspirational Goal: 4-8 workers multi-machine (completely untested)
- Theoretical Design: 8-32 workers globally (no validation)
- Pie-in-the-Sky Target: 100+ workers with regional clustering (very far away)

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

_Attempting to democratize AI training, one broken prototype at a time._ 🧪

**Remember: This is research code. It probably doesn't work. Use at your own risk.**
