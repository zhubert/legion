# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Legion is a distributed LLM training system that enables training across consumer-grade machines connected via the internet. The project is currently in **Phase 1** (Core Infrastructure), having completed Phase 0 (Proof of Concept).

**Key Innovation**: Adapts datacenter techniques (ZeRO-3, gradient compression) to work over high-latency consumer networks (10-100ms) rather than datacenter networks (<1ms).

## Architecture

Legion consists of three main components:

### 1. Simulation Layer (`sim/`)
Single-machine simulation that validates distributed training concepts. Workers are simulated as processes/threads.

**Key files:**
- `sim/train.py` - Main training script with both distributed and baseline modes
- `sim/model.py` - Tiny transformer for testing
- `sim/partitioner.py` - ZeRO-3 style parameter partitioning
- `sim/collectives.py` - All-gather and reduce-scatter implementations
- `sim/compression.py` - INT8 quantization and TopK compression
- `sim/worker.py` - Worker coordinator for simulation

### 2. Coordinator Server (`coordinator/`)
Central coordinator managing worker registry, health monitoring, regional clustering, and **training configuration**.

**Key files:**
- `coordinator/server.py` - FastAPI server with REST + WebSocket endpoints
- `coordinator/registry.py` - Worker registration and lifecycle management
- `coordinator/clustering.py` - Latency-based regional clustering
- `coordinator/database.py` - SQLite persistence
- `coordinator/training_config.py` - **Training configuration management (NEW)**

**Training Configuration (Coordinator-Driven):**
- **The coordinator makes ALL training decisions**: dataset, model, hyperparameters, compression, etc.
- Workers fetch their configuration from the coordinator on startup
- Supports worker-specific overrides (e.g., custom batch size based on GPU memory)
- Ensures all workers train consistently on the same model/dataset

**Communication:**
- Workers use REST for registration/heartbeat and fetching training config
- WebSocket for real-time event broadcasting
- Coordinator is NOT in the training loop (peer-to-peer worker communication)

### 3. Worker Client (`worker/`)
Distributed training worker that coordinates with the coordinator and performs local training.

**Key files:**
- `worker/client.py` - Main worker orchestration
- `worker/coordinator_client.py` - HTTP client for coordinator communication
- `worker/heartbeat.py` - Periodic heartbeat management
- `worker/shard_manager.py` - Parameter shard loading/saving
- `worker/trainer.py` - Local training loop with distributed collectives
- `worker/telemetry.py` - Metrics collection and reporting

### 4. Communication Layer (`communication/`)
Handles worker-to-worker communication via gRPC.

**Key files:**
- `communication/grpc_server.py` - gRPC server for serving parameters and accumulating gradients
- `communication/grpc_client.py` - gRPC client for fetching parameters and sending gradients
- `communication/grpc_collectives.py` - All-gather and reduce-scatter implementations
- `communication/serialization.py` - Tensor serialization with chunking for large transfers
- `communication/proto/` - Protocol buffer definitions

## Training Configuration (Coordinator-Driven Design)

**Philosophy:** The coordinator is the single source of truth for all training decisions. Workers are execution engines that fetch and follow the coordinator's configuration.

### Why Coordinator-Driven Configuration?

1. **Consistency**: All workers must train the same model on the same dataset with the same hyperparameters
2. **Centralized Control**: Operator configures training once at the coordinator, not per-worker
3. **Dynamic Adjustment**: Can adjust batch sizes per worker based on hardware capabilities
4. **Fault Tolerance**: New workers joining mid-training automatically get the correct configuration

### Configuration Workflow

```bash
# 1. Start coordinator (uses sensible defaults)
python -m coordinator.server

# 2. Configure training (optional - can do this before or after workers join)
curl -X PUT http://localhost:8000/training/config \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_type": "huggingface",
    "dataset_name": "tiny_shakespeare",
    "model_size": "tiny",
    "batch_size": 8,
    "seq_len": 256,
    "learning_rate": 0.001,
    "num_steps": 1000,
    "compression": "int8"
  }'

# 3. Start workers (they automatically fetch config from coordinator)
python -m worker.client  # Worker 1
python -m worker.client  # Worker 2

# 4. View current config
curl http://localhost:8000/training/config

# 5. Get worker-specific config (includes rank, world_size, any overrides)
curl http://localhost:8000/training/config/worker/{worker_id}

# 6. Set custom batch size for specific worker (e.g., GPU worker can handle more)
curl -X PUT http://localhost:8000/training/config/worker/{worker_id}/batch_size \
  -H "Content-Type: application/json" \
  -d '8'
```

### What the Coordinator Controls

- **Dataset**: Type (dummy/huggingface), name (fineweb, pile, shakespeare), tokenizer
- **Model**: Architecture size (tiny, small, medium)
- **Hyperparameters**: Batch size, sequence length, learning rate, training steps
- **Compression**: Strategy (none, int8, topk)
- **Worker-Specific Overrides**: Custom batch sizes based on hardware

### What Workers Do

Workers **no longer** make training decisions via CLI flags. They:
1. Register with coordinator
2. **Fetch training configuration from coordinator**
3. Load the assigned dataset shard (based on rank/world_size from coordinator)
4. Execute training with coordinator's hyperparameters

## Development Commands

### Running All Services (Recommended)

```bash
# Start coordinator + workers + assembler in one terminal
python scripts/start_services.py

# With log files
python scripts/start_services.py --logs-dir logs

# Custom number of workers
python scripts/start_services.py --workers 3

# Skip assembler
python scripts/start_services.py --no-assembler
```

The `start_services.py` orchestrator:
- Starts all required services with color-coded, unified output
- Handles graceful shutdown with Ctrl+C
- Optionally writes separate log files per service
- Shows timestamped output from all processes
- Configurable worker count

### Running Tests
```bash
# All tests
pytest

# Specific test file
pytest tests/test_model.py

# With coverage
pytest --cov=sim --cov=coordinator --cov=worker

# Integration tests only
pytest tests/integration/

# Verbose output
pytest -v
```

### Running the Simulation (Phase 0)
```bash
# Basic simulation with 4 workers
python sim/train.py --workers 4 --model tiny

# With network latency simulation
python sim/train.py --workers 4 --model tiny --latency 50

# With compression
python sim/train.py --workers 4 --model tiny --compress int8

# Compare distributed vs baseline
python sim/train.py --mode both --workers 4 --steps 100
```

### Running Coordinator Server
```bash
# Start coordinator (uses default training config)
python -m coordinator.server

# Or with uvicorn directly
uvicorn coordinator.server:app --reload --host 0.0.0.0 --port 8000

# Configure training via API (can be done before or after workers join)
curl -X PUT http://localhost:8000/training/config \
  -H "Content-Type: application/json" \
  -d '{"dataset_type": "distributed_dummy", "batch_size": 4, "num_steps": 100}'
```

### Running Worker Client
```bash
# Start worker (automatically fetches config from coordinator)
python -m worker.client

# Workers no longer need --dataset-type, --batch-size, etc.
# All training configuration comes from the coordinator

# Or via asyncio
python -c "import asyncio; from worker.client import main; asyncio.run(main())"
```

### Testing Distributed Training

**2-Worker Test (Recommended):**
```bash
# Terminal 1: Start coordinator
python -m coordinator.server

# Terminal 2: Run 2-worker test script
python scripts/test_two_workers.py
```

**Manual Multi-Worker:**
```bash
# Terminal 1: Start coordinator
python -m coordinator.server

# Terminal 2: Start worker 1
python -m worker.client

# Terminal 3: Start worker 2
python -m worker.client

# Terminal 4: Check status
curl http://localhost:8000/health
curl http://localhost:8000/workers
```

## Code Organization Principles

### Parameter Partitioning (ZeRO-3)
Each worker owns a subset of model parameters:
1. **All-gather**: Workers collect parameters from others for forward/backward pass
2. **Reduce-scatter**: Gradients aggregated and sent to parameter owners
3. **Update**: Only owners update their parameters

Memory reduction: `O(model_size / num_workers)` per worker.

### Training Flow
```python
# 1. All-gather parameters (async, overlapped)
parameters = all_gather_parameters(regional_workers)

# 2. Forward pass
activations = model.forward(batch, parameters)

# 3. Backward pass
gradients = model.backward(activations)

# 4. Reduce-scatter gradients to owners (async)
reduce_scatter_gradients(gradients, regional_workers, compression)

# 5. Parameter update (owners only)
if is_parameter_owner:
    optimizer.step(gradients)
```

### Compression Stack
- **INT8 quantization**: 4x compression (FP32 → INT8)
- **TopK sparsification**: Send only largest gradients
- **1-bit Adam** (planned Phase 3): 32x compression after warmup
- **Target**: 64-100x total compression

### Coordinator Heartbeat Protocol
- Workers send heartbeat every 30s
- Coordinator marks workers offline after 90s timeout
- WebSocket broadcasts worker status changes

## Testing Strategy

The project has comprehensive tests across all components:

- **Unit tests**: `tests/test_*.py` - Test individual components
- **Integration tests**: `tests/integration/` - Test component interactions
- **End-to-end tests**: `scripts/test_two_workers.py` - Full distributed training
- **Current coverage**: 164 passing tests, 4 skipped

When adding new features:
1. Write unit tests for new components
2. Add integration tests for multi-component interactions
3. Ensure simulation tests still pass (baseline for correctness)

## Style Guidelines

Following the author's style guide (STYLEGUIDE.md):

- **Voice**: Conversational, technically fluent, teaching through storytelling
- **Code comments**: Explain "why" not "what"
- **Documentation**: Context over definition, assume reader intelligence
- **Technical writing**: Specific when it matters, accessible without dumbing down

## Current Development Status

**Completed (Phase 0)**:
- ✅ Single-machine simulation
- ✅ ZeRO-3 parameter partitioning
- ✅ All-gather/reduce-scatter collectives
- ✅ INT8 and TopK compression
- ✅ Network latency simulation
- ✅ End-to-end training test

**Completed (Phase 1.3)**:
- ✅ Coordinator server with REST + WebSocket
- ✅ Worker client with heartbeat and telemetry
- ✅ gRPC worker-to-worker communication
- ✅ Real distributed training with ZeRO-3
- ✅ Multi-worker integration tests (2+ workers)
- ✅ Gradient accumulation across workers
- ✅ Parameter synchronization via gRPC
- ✅ Training barriers and synchronization

**Next Steps (Phase 2)**:
1. Add gradient compression to gRPC transfers (INT8)
2. Latency measurement and regional clustering
3. Fault tolerance testing (worker dropout/rejoin)
4. Async gradient accumulation with variable worker participation
5. Scale to 4-8 workers for production testing

## Common Pitfalls

1. **Coordinator-driven configuration**: Workers do NOT choose their own dataset, model, or hyperparameters. The coordinator makes ALL training decisions. Workers fetch configuration from the coordinator on startup via `/training/config/worker/{worker_id}`. If you're setting `--dataset-type` or similar flags on workers, you're doing it wrong.

2. **Coordinator is NOT in the training loop**: Workers communicate peer-to-peer for training. Coordinator only manages discovery, health, clustering, and **configuration**.

3. **Simulation vs Real Distribution**: The `sim/` code simulates multiple workers in one process. The `worker/` code runs actual distributed workers. Don't confuse them.

4. **Parameter ownership**: In ZeRO-3, each worker only updates parameters it owns. All-gather temporarily brings all parameters for computation.

5. **Async/await**: Worker client uses asyncio extensively. Remember to await async functions and use `asyncio.run()` for entry points.

6. **Device management**: The simulation defaults to CPU. Real workers should detect GPU availability via `torch.cuda.is_available()`.

7. **NO RING TOPOLOGIES**: Legion explicitly does NOT use ring-based collectives (ring allreduce, etc.). Ring topologies assume stable, synchronous workers completing at the same time - the exact opposite of Legion's design. Legion has hundreds of heterogeneous consumer machines (CPU/GPU mix) joining/leaving dynamically and finishing training units at vastly different times. The peer-to-peer async all-gather/reduce-scatter architecture is the correct approach for this use case.

## Dependencies

Core stack:
- **PyTorch 2.9+**: Model and training
- **FastAPI + Uvicorn**: Coordinator server
- **httpx**: Async HTTP client for workers
- **pytest**: Testing framework

Communication:
- **gRPC**: Worker-to-worker communication
- **protobuf**: Tensor serialization

## Monitoring and Debugging

### Logs
All components use Python's `logging` module:
- Coordinator: INFO level by default
- Worker: Configurable via `WorkerConfig.log_level`
- Simulation: Print statements for user-facing output

### Metrics
Workers report metrics to coordinator:
- Loss, throughput, memory usage
- Stored in SQLite (`coordinator.db`)
- Accessible via `/metrics` REST endpoint
- Real-time via WebSocket `/ws/events`

### Debugging Distributed Issues
1. Start with simulation (`sim/train.py`) to verify correctness
2. Run single worker to test coordinator communication
3. Add second worker to test distributed collectives
4. Check coordinator logs for registration/heartbeat issues
5. Use WebSocket client to monitor real-time events

## References

The project draws from several key papers:
- **ZeRO** ([arXiv:1910.02054](https://arxiv.org/abs/1910.02054)): Parameter partitioning
- **1-bit Adam** ([arXiv:2102.02888](https://arxiv.org/abs/2102.02888)): Gradient compression
- **DeepSpeed**: General distributed training techniques

See PROJECT.md for complete technical background and roadmap.
