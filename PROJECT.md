# Legion: Distributed LLM Training Across Consumer Hardware

> _A SETI@home for training language models - distributed pre-training across the internet_

## Project Vision

Train large language models by distributing the computational workload across consumer-grade machines connected via the internet. This project explores whether cutting-edge distributed training techniques (inspired by DeepSpeed) can be adapted to work over high-latency, low-bandwidth consumer networks.

### Educational Goals

1. **Understand distributed systems** at scale with real-world constraints
2. **Explore state-of-the-art ML systems** techniques (ZeRO, gradient compression, pipeline parallelism)
3. **Discover novel algorithms** for high-latency, fault-tolerant training
4. **Demonstrate feasibility** of democratic, decentralized AI training

### Related Projects

- `../transformer` - Explores pre-training fundamentals
- `../attention-to-detail` - Complete training pass by hand
- `../autotune` - Post-training optimization
- **Legion** - Distributed pre-training at scale

## Technical Foundation

### Core Challenges

1. **Communication Bottleneck**
   - Consumer internet: 10-1000 Mbps bandwidth, 10-100ms latency
   - Datacenter: 100-400 GB/s bandwidth, <1ms latency
   - **Gap: ~1000x slower bandwidth, ~10,000x higher latency**

2. **Fault Tolerance**
   - Consumer machines are unreliable (users turn off, networks drop)
   - Need elastic training that handles 10-20% worker churn

3. **Heterogeneous Hardware**
   - Mix of GPUs (RTX 4090 to integrated graphics)
   - Different CPU/RAM configurations (8GB to 128GB)
   - Various network speeds (DSL to fiber)

### Key Insights from DeepSpeed

#### 1. ZeRO (Zero Redundancy Optimizer)

**Memory Partitioning Strategy:**

- **Stage 1**: Partition optimizer states → 4x memory reduction
- **Stage 2**: Partition gradients → 8x memory reduction
- **Stage 3**: Partition model parameters → Linear scaling with workers

**Impact:**

- 175B model across 1000 workers = ~175M params/worker = **700MB instead of 700GB**
- Each worker only stores/updates parameters it "owns"
- Temporarily all-gather parameters needed for computation

**Communication Pattern:**

```
Forward:  All-gather parameters from owners
Compute:  Process batch with full (temporary) model
Backward: Compute gradients
Reduce:   Reduce-scatter gradients to owners
Update:   Owners update their parameters
```

#### 2. Gradient Compression

**1-bit Adam:**

- Warmup phase (15-20%): Standard Adam
- Compression phase: Variance stabilizes
  - Keep variance fixed (no communication)
  - Compress momentum to 1-bit
  - Error compensation tracks quantization errors
- **Result: 32x compression with same convergence**

**ZeRO++ Enhancements:**

- Block-based INT8 quantization: 2x reduction
- Hierarchical partitioning: Trade memory for communication
- Quantized gradient all-to-all: 75% reduction
- **Combined: 100x+ compression potential**

#### 3. ZeRO-Offload (Heterogeneous Memory)

**Memory Hierarchy:**

```
GPU VRAM (fastest, smallest):    Active parameters during compute
CPU RAM (fast, larger):          Optimizer states, gradients
NVMe SSD (slow, huge):           Checkpoint storage
```

**Benefits:**

- Workers with weak GPUs can participate
- Consumer machines: 8-16GB GPU, 32-64GB RAM, TB+ SSD
- Democratizes participation

#### 4. Fault Tolerance & Elastic Training

**Mechanisms:**

- Continuous checkpointing to distributed storage
- Elastic recovery when workers drop/rejoin
- Universal checkpointing: Reshape parallelism on recovery
- **Essential for consumer hardware with high churn**

#### 5. Communication/Computation Overlap

**Pipeline the Pipeline:**

- While computing layer N gradients, send layer N-1 gradients
- While computing forward pass, all-gather next layer's parameters
- **If computation_time > communication_time: zero overhead**

### Bandwidth Reality Check

#### 7B Parameter Model (28GB in bf16)

```
Base size:              28 GB
÷ 100 workers (ZeRO-3): 280 MB per worker
÷ 64x compression:      4.4 MB per sync
÷ 10 Mbps connection:   ~3.5 seconds

Computation time per micro-batch: ~10 seconds
Conclusion: FEASIBLE ✓
```

#### 175B Parameter Model (700GB in bf16)

```
Base size:               700 GB
÷ 1000 workers (ZeRO-3): 700 MB per worker
÷ 64x compression:       11 MB per sync
÷ 50 Mbps connection:    ~2 seconds

With 30-second local training intervals: FEASIBLE ✓
```

## Architecture Design

### System Overview

```
┌─────────────────────────────────────────────────────┐
│           Coordinator / Tracker Server              │
│                                                     │
│  - Worker registry & health monitoring             │
│  - Checkpoint coordination                          │
│  - Training metrics aggregation                     │
│  - Regional cluster assignment                      │
└──────────────┬──────────────────────────────────────┘
               │
    ┌──────────┴──────────┬─────────────┬────────────┐
    │                     │             │            │
┌───▼─────┐          ┌───▼──────┐  ┌──▼──────┐  ┌──▼──────┐
│ Region  │          │ Region   │  │ Region  │  │ Region  │
│ US-East │          │ US-West  │  │ Europe  │  │  Asia   │
│         │          │          │  │         │  │         │
│ Hub Node│          │ Hub Node │  │Hub Node │  │Hub Node │
└────┬────┘          └────┬─────┘  └────┬────┘  └────┬────┘
     │                    │             │            │
  ┌──┴──┬──┬──┐      ┌───┴──┬──┐   ┌───┴──┬──┐  ┌──┴──┬──┐
  │  │  │  │  │      │   │  │  │   │   │  │  │  │  │  │  │
 Workers....        Workers...     Workers...    Workers...
(Consumer PCs)    (Consumer PCs) (Consumer PCs) (Consumer PCs)
```

### Component Architecture

#### 1. Coordinator Server

**Responsibilities:**

- Maintain global worker registry
- Assign workers to regional clusters based on latency
- Coordinate checkpoint distribution
- Track training progress and metrics
- Handle worker join/leave events

**Technology:**

- Lightweight server (Python + FastAPI)
- SQLite/PostgreSQL for state
- WebSocket for real-time updates
- Can run on modest cloud instance

#### 2. Regional Hub Nodes

**Responsibilities:**

- Coordinate synchronization within region
- Act as high-bandwidth relay
- Buffer gradients for cross-region sync
- Local checkpoint caching

**Technology:**

- Optional (system works without hubs)
- Can be high-bandwidth volunteers
- Reduces cross-region traffic

#### 3. Worker Nodes (Consumer Machines)

**Responsibilities:**

- Store partition of model parameters
- Compute forward/backward passes
- Participate in gradient synchronization
- Checkpoint owned parameters

**Requirements:**

- Minimum: 8GB RAM, 10 Mbps internet
- Recommended: 16GB+ RAM, GPU, 50+ Mbps
- Storage: 10-100GB for checkpoints

### Communication Protocol

#### Hierarchical Synchronization

**Within Region (Fast - every micro-batch):**

```
1. All-gather: Collect parameters from regional workers
2. Forward pass: Compute activations
3. Backward pass: Compute gradients
4. Reduce-scatter: Send gradients to parameter owners
5. Update: Owners apply optimizer step
```

**Cross-Region (Slow - every K micro-batches):**

```
1. Regional hubs aggregate local updates
2. Hubs exchange compressed gradient statistics
3. Global parameter adjustment
4. Propagate back to regional workers
```

#### Compression Stack

**Layer 1: Quantization**

- FP32 → INT8 for gradients (4x compression)
- Block-based quantization for better accuracy
- Dynamic range per tensor block

**Layer 2: 1-bit Adam**

- Momentum compression to 1-bit after warmup
- Error compensation for convergence
- Variance kept locally (no sync needed)

**Layer 3: Sparsification**

- Top-K gradient selection (send only largest)
- Adaptive K based on bandwidth
- Residual accumulation for dropped gradients

**Target: 64-100x total compression**

#### Transport Layer

**Options to Evaluate:**

1. **gRPC** (Initial implementation)
   - Efficient binary protocol
   - Streaming support
   - Good for point-to-point

2. **BitTorrent Protocol** (Future enhancement)
   - Decentralized parameter sharing
   - Natural load balancing
   - Seed/leech parameter shards

3. **IPFS** (Checkpoint distribution)
   - Content-addressed storage
   - Decentralized hosting
   - Automatic deduplication

### Training Algorithm

#### Phase 1: Initialization

```python
1. Worker joins network:
   a. Connect to coordinator
   b. Receive parameter shard assignment
   c. Download checkpoint from distributed storage
   d. Join regional cluster

2. Model partitioning (ZeRO-3):
   - Each worker assigned parameter range [start:end]
   - Initialize optimizer state in CPU RAM
   - Allocate GPU memory for active computation
```

#### Phase 2: Training Loop

```python
for global_step in range(max_steps):
    # LOCAL TRAINING (multiple micro-batches)
    for local_step in range(gradient_accumulation_steps):

        # 1. All-gather parameters (async, overlapped)
        parameters = all_gather_parameters(
            regional_workers,
            needed_layers=current_batch_layers,
            compression=quantize_int8
        )

        # 2. Forward pass
        activations = model.forward(
            batch=get_next_batch(),
            parameters=parameters
        )

        # 3. Backward pass
        gradients = model.backward(activations)

        # 4. Reduce-scatter gradients to owners (async)
        reduce_scatter_gradients(
            gradients,
            regional_workers,
            compression=onebit_compress if warmed_up else quantize_int8
        )

        # 5. Accumulate gradients
        accumulated_gradients += gradients

    # PARAMETER UPDATE
    if is_parameter_owner(param_range):
        optimizer.step(accumulated_gradients)
        accumulated_gradients.zero_()

    # CROSS-REGION SYNC (periodic)
    if global_step % cross_region_interval == 0:
        sync_across_regions(compression=aggressive)

    # CHECKPOINTING (periodic)
    if global_step % checkpoint_interval == 0:
        checkpoint_shard_to_ipfs(owned_parameters)

    # HEARTBEAT
    send_heartbeat_to_coordinator(metrics)
```

#### Phase 3: Fault Tolerance

```python
# Worker dropout detection
if worker_heartbeat_timeout():
    # 1. Mark worker as offline
    mark_worker_offline(worker_id)

    # 2. Redistribute parameter shard
    reassign_shard(
        shard=worker.parameter_range,
        to=find_underutilized_worker()
    )

    # 3. New owner downloads checkpoint
    download_checkpoint_from_ipfs(worker.parameter_range)

# Worker rejoin
if new_worker_joins():
    # 1. Assign parameter shard (may be new or reassigned)
    shard = assign_parameter_shard(worker_capacity)

    # 2. Download checkpoint
    download_checkpoint_from_ipfs(shard)

    # 3. Join regional cluster
    assign_to_region(worker.latency_map)
```

## Development Roadmap

### Phase 0: Proof of Concept (Week 1-2) ✅ COMPLETED

**Goal:** Validate core concepts with minimal implementation

**Tasks:**

- [x] Single-machine simulation of distributed training
  - Simulate N workers as threads/processes
  - Implement basic ZeRO-3 parameter partitioning
  - Test all-gather / reduce-scatter collectives
- [x] Benchmark compression techniques
  - INT8 quantization
  - Top-K sparsification
  - Measure accuracy impact
- [x] Network latency simulation
  - Add artificial delays to simulate internet latency
  - Test convergence with delayed gradients

**Deliverables:**

- ✅ `sim/train.py` - Local simulation with distributed training
- ✅ `sim/compression.py` - Compression primitives (INT8, TopK)
- ✅ `sim/partitioner.py` - ZeRO-3 parameter partitioning
- ✅ `sim/collectives.py` - All-gather and reduce-scatter
- ✅ `sim/worker.py` - Worker coordination
- ✅ `sim/model.py` - Tiny GPT model
- ✅ 37 passing unit tests

**Success Criteria:**

- ✅ Model trains successfully with simulated ZeRO-3
- ✅ Compression maintains <5% accuracy degradation
- ✅ Algorithm converges with 50-100ms simulated latency
- ✅ End-to-end distributed training works

---

### Phase 1: Core Infrastructure (Week 3-4) 🚧 IN PROGRESS

**Goal:** Build foundational distributed components to move from single-machine simulation to actual multi-machine distributed training

**Implementation Approach:**

1. **Incremental builds** - Each component testable independently
2. **Docker-first** - Use containers to simulate multiple machines
3. **Test early, test often** - Write tests alongside code
4. **Document as you go** - Add docstrings and README updates immediately

**Development Priority:**

1. Coordinator Server (Days 1-3) - Core infrastructure
2. Worker Client (Days 4-5) - Workers can register and communicate
3. gRPC Communication (Days 6-8) - Enable actual distributed training
4. Integration Tests (Days 9-10) - Validate entire system

**Tasks:**

**1. Coordinator Server (Days 1-3)**

- [ ] Worker registry & discovery
  - REST API endpoints for worker registration/deregistration
  - Worker metadata storage (ID, IP, GPU info, network bandwidth)
  - SQLite database for persistent worker state
- [ ] Health monitoring
  - Heartbeat protocol (30s intervals)
  - Worker failure detection (60-90s timeout)
  - Graceful worker removal
- [ ] Regional cluster assignment
  - Latency measurement between workers (ping-based)
  - Automatic region detection (group workers with <50ms latency)
  - Dynamic rebalancing when new workers join
- [ ] Checkpoint coordination
  - Track which workers own which parameter shards
  - Checkpoint metadata registry
  - Version tracking for recovery
- [ ] Telemetry & monitoring
  - Aggregate training metrics from all workers
  - WebSocket endpoint for real-time status updates
  - Basic web dashboard for visualization

**2. Worker Client (Days 4-5)**

- [ ] Coordinator connection
  - Register with coordinator on startup
  - Send periodic heartbeats (every 30s)
  - Handle graceful shutdown and deregistration
- [ ] Parameter shard management
  - Download assigned parameter shard
  - Store parameters in memory (GPU/CPU)
  - Serve parameters to other workers on request
- [ ] Telemetry reporting
  - Collect local metrics (loss, throughput, memory usage)
  - Report to coordinator periodically
- [ ] Basic training loop integration
  - Load data locally
  - Execute forward/backward passes
  - Participate in collective communications

**3. Communication Layer - gRPC (Days 6-8)**

- [ ] gRPC setup
  - Define protobuf messages for tensors/parameters
  - Service definitions for parameter transfer
  - Async streaming for large tensors
- [ ] All-gather implementation
  - Each worker broadcasts its owned parameters
  - Ring-based all-gather for efficiency
  - Support for layer-by-layer gathering (reduce memory)
- [ ] Reduce-scatter implementation
  - Gradient aggregation across workers
  - Ring-based reduce-scatter
  - Route gradients to parameter owners
- [ ] Point-to-point transfer
  - Direct parameter request/response
  - Chunked transfer for large tensors
  - Retry logic for network failures

**4. Integration Testing (Days 9-10)**

- [ ] Multi-machine setup
  - Test with 2-4 machines (or Docker containers)
  - Workers discover each other via coordinator
  - Verify heartbeat and failure detection
- [ ] Communication tests
  - All-gather correctness across workers
  - Reduce-scatter correctness
  - Latency and bandwidth measurements
- [ ] Fault tolerance tests
  - Worker graceful shutdown
  - Worker crash detection (heartbeat timeout)
  - Network partition handling

**Deliverables:**

- `coordinator/server.py` - FastAPI server with REST + WebSocket
- `coordinator/registry.py` - Worker registry management
- `coordinator/clustering.py` - Regional clustering logic
- `coordinator/database.py` - SQLite operations
- `worker/client.py` - Worker daemon
- `worker/heartbeat.py` - Heartbeat logic
- `worker/telemetry.py` - Metrics collection
- `communication/collectives.py` - All-gather, reduce-scatter abstractions
- `communication/grpc_transport.py` - gRPC implementation
- `communication/proto/tensor.proto` - Protobuf definitions
- `tests/integration/` - Integration test suite

**Success Criteria:**

- ✅ 10 workers can connect to coordinator
- ✅ Workers successfully discover each other via coordinator
- ✅ Basic all-gather works across real network
- ✅ Heartbeat correctly detects worker failures (within 60s)
- ✅ System works across multiple machines (not just processes)

**Technical Decisions:**

- **FastAPI** for coordinator (fast, modern, WebSocket support)
- **SQLite** for coordinator state (simple, upgradeable to Postgres)
- **gRPC** for worker-to-worker (efficient binary protocol, streaming)
- **Heartbeat every 30s** (balance responsiveness vs overhead)

**Key Challenges & Mitigations:**

- NAT/Firewall issues → Start with same network, add NAT traversal later
- gRPC complexity → Extensive docs and examples, keep sim/ as reference
- Coordinator bottleneck → Minimal involvement in training loop, peer-to-peer communication

---

### Phase 2: ZeRO-3 Implementation (Week 5-6)

**Goal:** Implement memory-efficient parameter partitioning

**Tasks:**

- [ ] Parameter partitioning logic
  - Shard model weights across workers
  - Dynamic load balancing
  - Handle heterogeneous worker capacity
- [ ] All-gather with partitioning
  - Collect parameters only when needed
  - Layer-by-layer gathering
  - Overlap with computation
- [ ] Reduce-scatter with partitioning
  - Route gradients to parameter owners
  - Gradient accumulation
  - Overlap with backward pass
- [ ] CPU offloading (ZeRO-Offload)
  - Optimizer states in CPU RAM
  - GPU ↔ CPU transfers
  - NVMe offload for extreme scale

**Deliverables:**

- `zero/partitioner.py` - Parameter sharding logic
- `zero/all_gather.py` - ZeRO all-gather
- `zero/reduce_scatter.py` - ZeRO reduce-scatter
- `zero/offload.py` - CPU/NVMe offloading

**Success Criteria:**

- Train 7B model split across 10 workers
- Each worker uses <1GB GPU memory
- Optimizer states successfully offloaded to CPU
- Training throughput within 2x of single-GPU

---

### Phase 3: Compression & Communication (Week 7-8)

**Goal:** Minimize bandwidth requirements

**Tasks:**

- [ ] INT8 quantization
  - Block-based quantization
  - Dynamic range computation
  - Dequantization
- [ ] 1-bit Adam implementation
  - Warmup phase detection
  - Momentum compression
  - Error compensation
  - Convergence validation
- [ ] Gradient sparsification
  - Top-K selection
  - Residual accumulation
  - Adaptive K based on bandwidth
- [ ] Communication overlap
  - Pipeline all-gather with forward pass
  - Pipeline reduce-scatter with backward pass
  - Async communication primitives

**Deliverables:**

- `compression/int8.py` - INT8 quantization
- `compression/onebit_adam.py` - 1-bit Adam optimizer
- `compression/sparse.py` - Sparsification
- `communication/async_ops.py` - Async communication

**Success Criteria:**

- Achieve 50x+ compression on gradients
- 1-bit Adam converges equivalently to regular Adam
- Communication overlapped with 80%+ of computation
- Bandwidth usage <10 MB/s per worker

---

### Phase 4: Hierarchical Training (Week 9-10)

**Goal:** Optimize for regional clusters and cross-region sync

**Tasks:**

- [ ] Regional cluster formation
  - Latency measurement between workers
  - Automatic region assignment
  - Dynamic rebalancing
- [ ] Hub node implementation (optional)
  - Gradient aggregation within region
  - Cross-region relay
  - Bandwidth optimization
- [ ] Hierarchical synchronization
  - Fast intra-region sync (every micro-batch)
  - Slow inter-region sync (every N steps)
  - Elastic averaging for cross-region
- [ ] Adaptive sync intervals
  - Bandwidth-aware scheduling
  - Computation/communication balance
  - Worker-specific intervals

**Deliverables:**

- `regions/clustering.py` - Region formation
- `regions/hub.py` - Hub node (optional)
- `training/hierarchical_sync.py` - Multi-level sync
- `training/adaptive_schedule.py` - Dynamic intervals

**Success Criteria:**

- Workers auto-organize into <5 regional clusters
- Intra-region sync <100ms
- Inter-region sync doesn't block training
- Convergence maintained with hierarchical sync

---

### Phase 5: Fault Tolerance (Week 11-12)

**Goal:** Handle worker churn gracefully

**Tasks:**

- [ ] Checkpointing system
  - Periodic checkpoint creation
  - Shard-based checkpoints (each worker saves owned params)
  - Checkpoint versioning
  - Incremental checkpoints
- [ ] Distributed checkpoint storage
  - IPFS integration for decentralized storage
  - BitTorrent protocol for checkpoint distribution
  - Local caching
  - Automatic pruning of old checkpoints
- [ ] Elastic training
  - Detect worker dropout
  - Redistribute parameter shards
  - Worker rejoin protocol
  - Training continuation without restart
- [ ] Fault recovery
  - Checkpoint restoration
  - Gradient state recovery
  - Optimizer state reconstruction
  - Progress tracking and resume

**Deliverables:**

- `checkpoint/manager.py` - Checkpoint orchestration
- `checkpoint/ipfs_backend.py` - IPFS storage
- `elastic/shard_manager.py` - Dynamic sharding
- `elastic/recovery.py` - Fault recovery

**Success Criteria:**

- Checkpoint created and restored successfully
- Worker can drop out and rejoin without data loss
- Training continues with 20% worker churn
- Checkpoints distributed via IPFS

---

### Phase 6: Training & Validation (Week 13-14)

**Goal:** End-to-end training on real model

**Tasks:**

- [ ] Dataset preparation
  - Download and preprocess training data (e.g., OpenWebText)
  - Tokenization
  - Data distribution across workers (no duplication)
  - Streaming from distributed storage
- [ ] Model implementation
  - GPT-2 or LLaMA architecture
  - Start with smaller model (1B params)
  - Scale to 7B if resources allow
- [ ] Training run
  - Multi-day training across distributed workers
  - Real-time monitoring dashboard
  - Convergence tracking
  - Loss curves and validation
- [ ] Evaluation
  - Perplexity on validation set
  - Compare to baseline (single-GPU training)
  - Measure communication overhead
  - Analyze fault tolerance in practice

**Deliverables:**

- `data/dataset.py` - Dataset handling
- `models/gpt.py` - Model architecture
- `training/trainer.py` - Main training loop
- `monitoring/dashboard.py` - Web dashboard
- Training logs and metrics

**Success Criteria:**

- Successfully train 1B+ parameter model
- Convergence matches single-machine baseline (±5%)
- System survives worker dropout/rejoin
- Communication overhead <30% of training time

---

### Phase 7: Optimization & Scaling (Week 15-16)

**Goal:** Optimize for performance and scale

**Tasks:**

- [ ] Performance profiling
  - Identify bottlenecks
  - CPU/GPU utilization analysis
  - Network profiling
  - Memory usage optimization
- [ ] Advanced optimizations
  - Mixed precision training (FP16/BF16)
  - Gradient accumulation tuning
  - Better communication scheduling
  - Kernel fusion where possible
- [ ] Scalability testing
  - Scale to 50+ workers
  - Test with heterogeneous hardware
  - Measure scaling efficiency
  - Identify scaling limits
- [ ] Algorithm improvements
  - Experiment with asynchronous SGD variants
  - Test elastic averaging SGD (EASGD)
  - Implement staleness-aware learning rates
  - Explore local SGD (multiple local steps)

**Deliverables:**

- `profiling/` - Profiling tools and reports
- `optimization/` - Performance optimizations
- Scalability analysis document
- Algorithm comparison results

**Success Criteria:**

- Training throughput improved by 2x+
- System scales to 50+ workers
- Communication overhead reduced to <20%
- Algorithm variations explored and documented

---

### Phase 8: Documentation & Release (Week 17-18)

**Goal:** Prepare for public release and community

**Tasks:**

- [ ] Documentation
  - Architecture documentation
  - API reference
  - Tutorial: Running your first distributed training
  - Tutorial: Adding a new worker
  - Design decisions and tradeoffs
- [ ] Code cleanup
  - Refactoring
  - Type hints
  - Unit tests
  - Integration tests
  - CI/CD setup
- [ ] Deployment tooling
  - Docker containers for coordinator and worker
  - Docker Compose for local testing
  - Kubernetes manifests (optional)
  - Easy installer scripts
- [ ] Community preparation
  - README with clear value proposition
  - Contributing guidelines
  - Code of conduct
  - Example training runs with metrics
  - Blog post explaining the project

**Deliverables:**

- Complete documentation site
- Test coverage >70%
- Docker Hub images
- Public GitHub repository
- Blog post / paper

**Success Criteria:**

- New contributor can set up worker in <30 minutes
- Documentation covers all major features
- CI/CD runs tests automatically
- Ready for community contributions

---

## Technical Specifications

### Model Support (Initial)

**Target Models:**

- GPT-2 (124M, 355M, 774M, 1.5B)
- GPT-Neo (125M, 1.3B, 2.7B)
- LLaMA (7B, 13B if resources allow)

**Stretch Goal:**

- Support for 30B+ models with 100+ workers

### Data Support

**Datasets:**

- OpenWebText (open-source GPT-2 dataset)
- The Pile (825GB, diverse text)
- Custom datasets (user-provided)

**Data Distribution:**

- Each worker assigned non-overlapping data shards
- Streaming from cloud storage or IPFS
- Local caching for frequently accessed data

### Hardware Requirements

**Minimum Worker:**

```
CPU:      4+ cores
RAM:      8GB
GPU:      Optional (CPU-only mode supported)
Storage:  50GB
Network:  10 Mbps upload/download
```

**Recommended Worker:**

```
CPU:      8+ cores
RAM:      16-32GB
GPU:      8GB+ VRAM (RTX 3060 or better)
Storage:  100GB+ SSD
Network:  50+ Mbps upload/download
```

**Coordinator Server:**

```
CPU:      2-4 cores
RAM:      4GB
Storage:  20GB
Network:  100+ Mbps (hosted on cloud/VPS)
```

### Software Stack

**Core:**

- Python 3.10+
- PyTorch 2.0+
- NumPy

**Communication:**

- gRPC (worker-to-worker)
- WebSocket (coordinator-to-worker)
- FastAPI (coordinator API)

**Storage:**

- IPFS (go-ipfs or kubo)
- SQLite/PostgreSQL (coordinator state)

**Monitoring:**

- Prometheus (metrics)
- Grafana (dashboards)
- Weights & Biases (experiment tracking)

**Deployment:**

- Docker
- Docker Compose
- Kubernetes (optional)

## Success Metrics

### Performance Metrics

**Training Efficiency:**

- Target: 50-70% of single-machine throughput per worker
- Communication overhead: <30% of total time
- GPU utilization: >70% on workers with GPUs

**Scalability:**

- Linear scaling up to 20 workers
- Sub-linear but functional up to 100+ workers
- Graceful degradation with worker churn

**Convergence:**

- Final loss within 5% of baseline
- Validation perplexity within 5% of baseline
- Training time: 2-3x slower than datacenter (acceptable given cost savings)

### Reliability Metrics

**Fault Tolerance:**

- Survive 20% simultaneous worker dropout
- Resume training within 5 minutes of failure
- Zero data loss with proper checkpointing

**Availability:**

- Coordinator uptime: >99.9%
- Worker discovery: <30 seconds
- Checkpoint restoration: <5 minutes

### Educational Metrics

**Learning Outcomes:**

- Understand distributed training algorithms
- Experience with real distributed systems
- Knowledge of ZeRO, gradient compression, fault tolerance
- Practical ML systems engineering

**Community Engagement:**

- GitHub stars: 100+ (12 months)
- Contributors: 5+ (12 months)
- Successful training runs: 10+ (by community)

## Novel Contributions

### Beyond Existing Work

1. **Internet-Scale Training**
   - First system designed for consumer internet latency/bandwidth
   - Most research assumes datacenter networks (<1ms latency)
   - We target 10-100ms latency, 10-1000 Mbps bandwidth

2. **Extreme Fault Tolerance**
   - Handle high worker churn (20%+ dropout rate)
   - Elastic training with dynamic resharding
   - Decentralized checkpoint distribution

3. **Heterogeneous Hardware Support**
   - Mix of GPUs, CPUs, various memory sizes
   - Adaptive workload assignment
   - CPU-only workers can participate

4. **Hierarchical Communication**
   - Regional clustering based on latency
   - Multi-tier synchronization strategy
   - Bandwidth-aware scheduling

5. **Democratic AI Training**
   - Anyone can contribute compute
   - No expensive datacenter required
   - Open, transparent training process

### Research Questions

1. **Does asynchronous training work at internet scale?**
   - How much staleness can models tolerate?
   - Optimal sync intervals for different network conditions?

2. **What compression ratios are achievable?**
   - Can we exceed 100x with novel techniques?
   - Accuracy vs. compression tradeoffs?

3. **How does hierarchical training affect convergence?**
   - Fast intra-region + slow inter-region sync
   - Does it improve generalization?

4. **What are the scaling limits?**
   - At what point does communication dominate?
   - Optimal cluster size for different models?

## Risks & Mitigations

### Technical Risks

**Risk 1: Communication overhead dominates computation**

- _Mitigation_: Aggressive compression (100x target)
- _Mitigation_: Computation/communication overlap
- _Mitigation_: Increase local training steps
- _Fallback_: Target smaller models (1-7B instead of 175B)

**Risk 2: Convergence fails with high latency**

- _Mitigation_: Elastic averaging SGD
- _Mitigation_: Staleness-aware learning rates
- _Mitigation_: Extensive simulation before deployment
- _Fallback_: Synchronous training with longer sync intervals

**Risk 3: Fault tolerance insufficient**

- _Mitigation_: Frequent checkpointing (every 100 steps)
- _Mitigation_: Redundant parameter storage
- _Mitigation_: Robust coordinator with failover
- _Fallback_: Accept some training loss, restart from checkpoint

**Risk 4: Bandwidth costs too high**

- _Mitigation_: Compression reduces bandwidth to <10 MB/s per worker
- _Mitigation_: Workers can limit upload/download rates
- _Mitigation_: Off-peak training scheduling
- _Fallback_: Train smaller models or fewer workers

### Operational Risks

**Risk 1: Insufficient workers to test**

- _Mitigation_: Start with simulated workers
- _Mitigation_: Recruit early testers from ML community
- _Mitigation_: Provide easy Docker setup
- _Fallback_: Use cloud VMs to simulate distributed workers

**Risk 2: Security/privacy concerns**

- _Mitigation_: Worker authentication via tokens
- _Mitigation_: Encrypted communication (TLS)
- _Mitigation_: Optional differential privacy
- _Fallback_: Closed-network testing first

**Risk 3: Coordinator becomes bottleneck**

- _Mitigation_: Minimal coordinator involvement in training loop
- _Mitigation_: Workers communicate peer-to-peer
- _Mitigation_: Stateless coordinator (can scale horizontally)
- _Fallback_: Multiple coordinator replicas

## Future Enhancements

### Post-MVP Features

1. **BitTorrent-style Parameter Sharing**
   - Decentralized parameter distribution
   - Seed/leech model updates
   - No central server bottleneck

2. **Incentive Mechanism**
   - Compute credits for contributing workers
   - Blockchain-based verification (optional)
   - Tiered access based on contribution

3. **Privacy-Preserving Training**
   - Differential privacy for gradients
   - Secure aggregation
   - Federated learning integration

4. **Advanced Parallelism**
   - Pipeline parallelism across workers
   - Tensor parallelism for extremely large layers
   - Expert parallelism for MoE models

5. **Auto-Tuning**
   - Automatic hyperparameter optimization
   - Adaptive compression ratios
   - Self-tuning communication schedules

6. **Multi-Tenancy**
   - Multiple training jobs on shared worker pool
   - Resource allocation and scheduling
   - Priority queues

### Research Directions

1. **Novel Optimization Algorithms**
   - Internet-optimized variants of Adam/AdamW
   - Delay-tolerant optimizers
   - Asynchronous momentum methods

2. **Communication Theory**
   - Optimal compression under quality constraints
   - Information-theoretic limits
   - Adaptive coding schemes

3. **Distributed Systems**
   - Byzantine fault tolerance
   - Consensus algorithms for training
   - Decentralized coordination

4. **Emergent Behaviors**
   - Does distributed training find different minima?
   - Generalization properties
   - Implicit regularization from asynchrony

## Timeline Summary

| Phase                      | Duration | Key Deliverable                     |
| -------------------------- | -------- | ----------------------------------- |
| 0: Proof of Concept        | 2 weeks  | Simulated distributed training      |
| 1: Core Infrastructure     | 2 weeks  | Coordinator + Worker + gRPC         |
| 2: ZeRO-3                  | 2 weeks  | Parameter partitioning working      |
| 3: Compression             | 2 weeks  | 50x+ compression achieved           |
| 4: Hierarchical Training   | 2 weeks  | Regional clusters functional        |
| 5: Fault Tolerance         | 2 weeks  | Elastic training + IPFS checkpoints |
| 6: Training & Validation   | 2 weeks  | 1B+ model trained end-to-end        |
| 7: Optimization            | 2 weeks  | 2x performance improvement          |
| 8: Documentation & Release | 2 weeks  | Public release ready                |

**Total: 16-18 weeks (4-5 months)**

## Getting Started

### For Developers

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/legion.git
   cd legion
   ```

2. **Set up development environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements-dev.txt
   ```

3. **Run local simulation**
   ```bash
   python sim/single_node_sim.py --workers 4 --model gpt2-small
   ```

### For Contributors

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### For Workers (Running a Node)

**Coming in Phase 6!**

Will be as simple as:

```bash
docker run -d legion/worker --coordinator https://legion.example.com
```

## Resources & References

### Foundational Papers

- **ZeRO**: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- **1-bit Adam**: [1-bit Adam: Communication Efficient Large-Scale Training](https://arxiv.org/abs/2102.02888)
- **DeepSpeed**: [DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters](https://arxiv.org/abs/2004.09985)
- **ZeRO-Offload**: [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- **EASGD**: [Deep Learning with Elastic Averaging SGD](https://arxiv.org/abs/1412.6651)

### Related Projects

- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - Foundation for many techniques
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Large-scale training framework
- [Hivemind](https://github.com/learning-at-home/hivemind) - Decentralized deep learning
- [Petals](https://github.com/bigscience-workshop/petals) - Distributed inference (related problem)

### Educational Resources

- [Distributed Training Guide](https://pytorch.org/tutorials/beginner/dist_overview.html) - PyTorch official docs
- [Understanding ZeRO](https://www.deepspeed.ai/tutorials/zero/) - DeepSpeed tutorials
- [Gradient Compression Survey](https://arxiv.org/abs/2003.01942) - Comprehensive overview

## License

MIT License - See [LICENSE](LICENSE)

## Contact

- Project Lead: Zack Hubert
- Issues: [GitHub Issues](https://github.com/zhubert/legion/issues)

---

_Let's democratize AI training, one GPU at a time._ 🚀
