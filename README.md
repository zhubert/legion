# Legion: Distributed LLM Training

> _A SETI@home for training language models - distributed pre-training across the internet_

## Overview

Legion is an experimental distributed training system that aims to enable LLM pre-training across consumer-grade machines. Inspired by SETI@home, it explores whether modern distributed training techniques (ZeRO, gradient compression, fault tolerance) can work over high-latency, low-bandwidth consumer networks.

See [PROJECT.md](PROJECT.md) for the complete project plan and technical details.

## Current Status: Phase 1 - Core Infrastructure

Legion has completed the proof-of-concept simulation (Phase 0) and is now in Phase 1 with functional distributed infrastructure:

**Phase 0 Complete:**
- ✅ Parameter partitioning (ZeRO-3 style)
- ✅ Collective communication (all-gather, reduce-scatter)
- ✅ Gradient compression (INT8 quantization)
- ✅ Network latency simulation
- ✅ End-to-end training test

**Phase 1 Complete:**
- ✅ Coordinator server (REST + WebSocket)
- ✅ Worker client with heartbeat and telemetry
- ✅ gRPC worker-to-worker communication
- ✅ Ring-based collectives (8x-512x bandwidth savings)
- ✅ Multi-worker integration tests

**Next Steps (Phase 1 Remaining):**
- Real multi-machine distributed training (2-4 workers)
- Latency measurement and regional clustering
- Fault tolerance testing

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

**Terminal 1: Start the coordinator server**
```bash
python -m coordinator.server
# Server runs on http://localhost:8000
```

**Terminal 2+: Start worker nodes**
```bash
# Worker 1
python -m worker.client

# Worker 2 (in another terminal)
python -m worker.client
```

Workers will automatically:
- Register with the coordinator
- Send periodic heartbeats
- Form a training cluster
- Exchange parameters via gRPC

## Project Structure

```
legion/
├── PROJECT.md              # Detailed project plan
├── README.md               # This file
├── CLAUDE.md               # Development guide
├── requirements.txt        # Python dependencies
├── sim/                    # Phase 0: Single-machine simulation
│   ├── model.py            # Tiny transformer for testing
│   ├── partitioner.py      # Parameter partitioning (ZeRO-3)
│   ├── collectives.py      # All-gather, reduce-scatter
│   ├── compression.py      # Gradient compression
│   ├── worker.py           # Simulated worker coordinator
│   └── train.py            # Simulation entry point
├── coordinator/            # Phase 1: Central coordinator
│   ├── server.py           # FastAPI REST + WebSocket server
│   ├── registry.py         # Worker registration and health
│   ├── clustering.py       # Latency-based regional clustering
│   └── database.py         # SQLite persistence
├── worker/                 # Phase 1: Distributed worker nodes
│   ├── client.py           # Main worker orchestration
│   ├── coordinator_client.py  # HTTP client for coordinator
│   ├── heartbeat.py        # Periodic heartbeat manager
│   ├── trainer.py          # Distributed training loop
│   ├── shard_manager.py    # Parameter shard management
│   └── telemetry.py        # Metrics reporting
├── communication/          # Phase 1: Worker-to-worker gRPC
│   ├── grpc_server.py      # gRPC server for parameters
│   ├── grpc_client.py      # gRPC client for requests
│   ├── grpc_collectives.py # gRPC-based all-gather/reduce-scatter
│   ├── ring_collectives.py # Ring-based bandwidth optimization
│   ├── serialization.py    # Tensor serialization/chunking
│   └── proto/              # Protocol buffer definitions
└── tests/                  # Comprehensive test suite
    ├── integration/        # Multi-worker integration tests
    ├── test_*.py           # Unit tests
    └── ...                 # 147 tests total
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

### Ring-Based Collectives

Bandwidth-efficient communication pattern where each worker only talks to 2 neighbors:

```
Worker topology: 0 <-> 1 <-> 2 <-> 3 <-> 0
```

**Bandwidth savings vs naive all-to-all:**
- 4 workers: 8x reduction
- 8 workers: 32x reduction
- 16 workers: 128x reduction
- 32 workers: 512x reduction

Example: For a 1B parameter model (4GB) on 16 workers:
- Ring all-reduce: ~7.75 GB total communication
- Naive all-reduce: ~960 GB total communication
- **128x improvement!**

### Network Architecture

**Coordinator** (FastAPI server):
- Worker registration and health monitoring
- Regional clustering based on latency
- Metrics aggregation
- NOT in the training loop (peer-to-peer communication)

**Workers** (async Python clients):
- gRPC server for serving parameter shards
- gRPC client for fetching from peers
- Training loop with ZeRO-3 partitioning
- Automatic fault detection and recovery

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
pytest tests/test_ring_collectives.py -v
```

**Test Coverage:**
- 147 total tests (146 passing, 1 requires running coordinator)
- Unit tests for all components
- Integration tests for multi-worker scenarios
- gRPC communication tests
- Ring collectives performance tests

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
- Ring all-reduce: O(N) steps, O(1) bandwidth per worker

**Memory:**
- ZeRO-3 partitioning: `O(model_size / num_workers)` per worker
- Parameter shards saved to disk for checkpointing
- Gradient accumulation for large batches

**Scalability:**
- Tested: 2-4 workers locally
- Designed for: 8-32 workers globally
- Target: 100+ workers with regional clustering

## Resources

**Papers:**
- [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) - Parameter partitioning
- [1-bit Adam](https://arxiv.org/abs/2102.02888) - Gradient compression
- [Ring All-Reduce](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/) - Bandwidth optimization

**Documentation:**
- [DeepSpeed](https://www.deepspeed.ai/)
- [gRPC Python](https://grpc.io/docs/languages/python/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

_Let's democratize AI training, one GPU at a time._ 🚀
