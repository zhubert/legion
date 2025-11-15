# Legion: Distributed LLM Training

> _A SETI@home for training language models - distributed pre-training across the internet_

## Overview

Legion is an experimental distributed training system that aims to enable LLM pre-training across consumer-grade machines. Inspired by SETI@home, it explores whether modern distributed training techniques (ZeRO, gradient compression, fault tolerance) can work over high-latency, low-bandwidth consumer networks.

See [PROJECT.md](PROJECT.md) for the complete project plan and technical details.

## Current Status: Proof of Concept (Phase 0)

We're building a single-machine simulation to validate the core concepts:

- ✅ Parameter partitioning (ZeRO-3 style)
- ✅ Collective communication (all-gather, reduce-scatter)
- ✅ Gradient compression (INT8 quantization)
- ✅ Network latency simulation
- 🚧 End-to-end training test

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

### Running the PoC Simulation

```bash
# Run single-machine simulation with 4 workers
python sim/train.py --workers 4 --model tiny

# With latency simulation (50ms)
python sim/train.py --workers 4 --model tiny --latency 50

# With compression enabled
python sim/train.py --workers 4 --model tiny --compress int8
```

## Project Structure

```
legion/
├── PROJECT.md              # Detailed project plan
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── sim/                    # Proof of Concept simulation
    ├── __init__.py
    ├── model.py            # Simple transformer model
    ├── partitioner.py      # Parameter partitioning (ZeRO-3)
    ├── collectives.py      # All-gather, reduce-scatter
    ├── compression.py      # Gradient compression
    ├── network_sim.py      # Network latency simulation
    ├── worker.py           # Worker process
    └── train.py            # Main training script
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
- **1-bit Adam** (future): 32x compression after warmup
- **Target**: 64-100x total compression

### Network Simulation

Artificial delays simulate internet latency:

- Typical consumer internet: 10-100ms latency
- Datacenter networks: <1ms latency
- We need to prove training works with high latency

## Contributing

This is an early-stage research project. Contributions are welcome!

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - See [LICENSE](LICENSE)

## Resources

- [DeepSpeed ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [1-bit Adam Paper](https://arxiv.org/abs/2102.02888)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)

---

_Let's democratize AI training, one GPU at a time._ 🚀
