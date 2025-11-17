# Dataset Guide

This guide explains how to use different datasets with Legion for distributed training.

## Overview

Legion supports three types of datasets:

1. **Dummy datasets** - Random data for testing (default)
2. **Distributed dummy datasets** - Sharded random data for multi-worker testing
3. **HuggingFace datasets** - Real datasets (FineWeb, The Pile, Shakespeare, etc.)

## Quick Start

### Using HuggingFace Datasets

To train on a real dataset, configure your worker with:

```python
from worker.config import WorkerConfig

config = WorkerConfig(
    dataset_type="huggingface",
    dataset_name="tiny_shakespeare",  # or "fineweb-edu", "pile", etc.
    batch_size=8,
    seq_len=512
)
```

Or pass configuration when starting the worker:

```bash
python -m worker.client \
    --dataset-type huggingface \
    --dataset-name fineweb-edu \
    --batch-size 8 \
    --seq-len 1024
```

## Available Datasets

### Preconfigured Datasets

Legion includes optimized configurations for popular LLM training datasets:

| Dataset | Name | Size | Streaming | Description |
|---------|------|------|-----------|-------------|
| **FineWeb** | `fineweb` | 15T tokens | Yes | CommonCrawl web data, high quality |
| **FineWeb-Edu** | `fineweb-edu` | 1.3T tokens | Yes | Educational content, very high quality |
| **The Pile** | `pile` | 825 GB | Yes | Diverse dataset (22 sources) |
| **Tiny Shakespeare** | `tiny_shakespeare` | 1 MB | No | For testing/debugging |
| **Shakespeare** | `shakespeare` | ~5 MB | No | Complete works of Shakespeare |

### Using Preconfigured Datasets

```python
from worker.config import WorkerConfig

# FineWeb-Edu (recommended for high-quality pretraining)
config = WorkerConfig(
    dataset_type="huggingface",
    dataset_name="fineweb-edu",
    seq_len=1024,
    batch_size=8
)

# The Pile (diverse dataset)
config = WorkerConfig(
    dataset_type="huggingface",
    dataset_name="pile",
    seq_len=2048,
    batch_size=4
)

# Tiny Shakespeare (fast, for testing)
config = WorkerConfig(
    dataset_type="huggingface",
    dataset_name="tiny_shakespeare",
    seq_len=256,
    batch_size=16
)
```

### Using Custom HuggingFace Datasets

You can use any HuggingFace dataset by providing the full name:

```python
config = WorkerConfig(
    dataset_type="huggingface",
    dataset_name="HuggingFaceFW/fineweb",  # Full HuggingFace path
    tokenizer_name="gpt2",  # Optional: override tokenizer
    seq_len=512,
    batch_size=8
)
```

## How Dataset Sharding Works

In distributed training, each worker loads a **non-overlapping shard** of the dataset:

### Example: 4 Workers Training on FineWeb-Edu

```
Global Dataset: [sample0, sample1, sample2, sample3, sample4, sample5, ...]

Worker 0 gets: [sample0, sample4, sample8, ...]  (every 4th starting at 0)
Worker 1 gets: [sample1, sample5, sample9, ...]  (every 4th starting at 1)
Worker 2 gets: [sample2, sample6, sample10, ...] (every 4th starting at 2)
Worker 3 gets: [sample3, sample7, sample11, ...] (every 4th starting at 3)
```

### Effective Batch Size

The **effective global batch size** is:

```
effective_batch_size = local_batch_size × world_size
```

Example:
- 4 workers, each with `batch_size=8`
- Effective global batch size = 32

This means each training step processes 32 samples across all workers.

## Dataset Loading Process

### Automatic Sharding

Workers automatically determine their rank and load the appropriate shard:

```python
# Worker automatically gets rank from coordinator
rank = 0  # Determined by coordinator
world_size = 4  # Total number of workers

# Dataset is automatically sharded based on rank
dataset = create_huggingface_dataset(
    dataset_name="fineweb-edu",
    rank=rank,  # This worker gets every 4th sample starting at 0
    world_size=world_size,
    num_batches=100,
    batch_size=8
)
```

### Streaming for Large Datasets

Large datasets (FineWeb, Pile) use **streaming mode** to avoid downloading the entire dataset:

```python
# Streaming enabled automatically for large datasets
dataset = create_huggingface_dataset(
    dataset_name="fineweb",  # 15TB dataset
    streaming=True,  # Downloads samples on-demand
    ...
)
```

Benefits:
- **Instant start**: No waiting for full download
- **Low memory**: Only buffers samples currently in use
- **Scalable**: Works with datasets larger than disk

### Tokenization

Datasets are automatically tokenized using the appropriate tokenizer:

```python
# Default tokenizer from dataset config
dataset_name="fineweb-edu"  # Uses GPT-2 tokenizer by default

# Override tokenizer
config = WorkerConfig(
    dataset_name="fineweb-edu",
    tokenizer_name="EleutherAI/gpt-neox-20b"  # Use different tokenizer
)
```

## Configuration Options

### Worker Config Dataset Fields

```python
@dataclass
class WorkerConfig:
    # Dataset configuration
    dataset_type: str = "dummy"
    # Options: "dummy", "distributed_dummy", "huggingface"

    dataset_name: Optional[str] = None
    # HuggingFace dataset name (e.g., "fineweb-edu")

    tokenizer_name: Optional[str] = None
    # Override default tokenizer

    batch_size: int = 4
    # Local batch size per worker

    seq_len: int = 32
    # Sequence length for training

    num_steps: int = 100
    # Number of training steps
```

### Programmatic Configuration

```python
from worker.config import WorkerConfig
from worker.client import WorkerClient

# Create configuration
config = WorkerConfig(
    dataset_type="huggingface",
    dataset_name="tiny_shakespeare",
    batch_size=16,
    seq_len=256,
    num_steps=1000
)

# Create and start worker
worker = WorkerClient(config)
await worker.start()
await worker.run_training(use_distributed=True)
```

## Testing Dataset Integration

### List Available Datasets

```python
from core.dataset import list_available_datasets

datasets = list_available_datasets()
for ds in datasets:
    print(f"{ds['name']}: {ds['description']}")
```

Output:
```
fineweb: FineWeb: 15T tokens from CommonCrawl
fineweb-edu: FineWeb-Edu: 1.3T high-quality educational tokens
pile: The Pile: 825GB diverse dataset (22 sources)
tiny_shakespeare: Tiny Shakespeare: 40K lines for testing
shakespeare: Complete Works of Shakespeare, line-by-line
```

### Test Dataset Loading

```python
from core.dataset import create_huggingface_dataset

# Load a small batch for testing
dataset = create_huggingface_dataset(
    dataset_name="tiny_shakespeare",
    rank=0,
    world_size=1,
    num_batches=5,
    batch_size=4,
    seq_len=128
)

print(f"Loaded {len(dataset)} batches")
for inputs, labels in dataset:
    print(f"Batch shape: {inputs.shape}")
    break
```

## Dependencies

HuggingFace dataset support requires additional packages:

```bash
# Install HuggingFace libraries
pip install datasets transformers

# Or with uv
uv pip install datasets transformers
```

These are **optional dependencies** - Legion works without them using dummy datasets.

## Performance Tips

### 1. Use Streaming for Large Datasets

Always use streaming for datasets > 10GB:

```python
config = WorkerConfig(
    dataset_name="fineweb",  # Automatically uses streaming
    ...
)
```

### 2. Adjust Batch Size for Memory

Larger models require smaller batches:

```python
# Small model
config = WorkerConfig(model_size="tiny", batch_size=32)

# Large model
config = WorkerConfig(model_size="medium", batch_size=4)
```

### 3. Start Small, Scale Up

Test with `tiny_shakespeare` before using large datasets:

```python
# Development/testing
config = WorkerConfig(
    dataset_name="tiny_shakespeare",
    num_steps=10
)

# Production
config = WorkerConfig(
    dataset_name="fineweb-edu",
    num_steps=100_000
)
```

### 4. Monitor Effective Batch Size

```python
effective_batch_size = config.batch_size * world_size
print(f"Effective batch size: {effective_batch_size}")

# Common configurations:
# - 4 workers × 8 batch = 32 effective
# - 8 workers × 4 batch = 32 effective
# - 16 workers × 2 batch = 32 effective
```

## Troubleshooting

### ImportError: datasets not installed

```
ImportError: HuggingFace datasets and transformers are required.
Install with: pip install datasets transformers
```

**Solution**: Install dependencies:
```bash
pip install datasets transformers
```

### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size or sequence length:
```python
config = WorkerConfig(
    batch_size=4,  # Reduced from 8
    seq_len=512    # Reduced from 1024
)
```

### Download Errors

```
ConnectionError: Failed to download dataset
```

**Solution**: Check internet connection or try a different dataset. For testing, use `tiny_shakespeare` which is small and fast to download.

### Dataset Not Found

```
FileNotFoundError: Dataset 'my-dataset' not found
```

**Solution**: Verify dataset name on [HuggingFace](https://huggingface.co/datasets):
```python
# Correct format
dataset_name="HuggingFaceFW/fineweb-edu"  # Full path

# Or use short name for preconfigured datasets
dataset_name="fineweb-edu"
```

## Examples

### Example 1: Single Worker with Shakespeare

```python
from worker.config import WorkerConfig
from worker.client import WorkerClient
import asyncio

async def main():
    config = WorkerConfig(
        dataset_type="huggingface",
        dataset_name="tiny_shakespeare",
        batch_size=16,
        seq_len=256,
        num_steps=100
    )

    worker = WorkerClient(config)
    await worker.start()
    await worker.run_training()
    await worker.stop()

asyncio.run(main())
```

### Example 2: Multi-Worker with FineWeb-Edu

```python
# Start multiple workers, each will automatically get a shard

# Worker 1
config1 = WorkerConfig(
    worker_id="worker-1",
    dataset_type="huggingface",
    dataset_name="fineweb-edu",
    batch_size=8,
    seq_len=1024
)

# Worker 2
config2 = WorkerConfig(
    worker_id="worker-2",
    dataset_type="huggingface",
    dataset_name="fineweb-edu",
    batch_size=8,
    seq_len=1024
)

# Each worker automatically gets non-overlapping data based on rank
```

### Example 3: Custom Dataset

```python
config = WorkerConfig(
    dataset_type="huggingface",
    dataset_name="bigcode/the-stack",  # Custom HuggingFace dataset
    tokenizer_name="codegen-350M",     # Custom tokenizer
    seq_len=2048,
    batch_size=4
)
```

## Next Steps

- See [README.md](README.md) for general usage
- See [PROJECT.md](PROJECT.md) for architecture details
- See [tests/test_huggingface_dataset.py](tests/test_huggingface_dataset.py) for more examples
