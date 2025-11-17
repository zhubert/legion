"""
Dataset utilities for testing and training.

Supports both dummy datasets (for testing) and real HuggingFace datasets
(fineweb, fineweb-edu, shakespeare, the pile, etc.) for production training.
"""

from typing import List, Tuple, Iterator, Optional, Dict, Any
import torch
import logging

logger = logging.getLogger(__name__)


def create_dummy_dataset(
    vocab_size: int,
    seq_len: int,
    num_batches: int,
    batch_size: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create dummy dataset for testing.

    In a real system, this would load actual text data.

    Args:
        vocab_size: Size of vocabulary
        seq_len: Sequence length
        num_batches: Number of batches to create
        batch_size: Batch size

    Returns:
        List of (input, target) batches
    """
    dataset = []
    for _ in range(num_batches):
        # Random token IDs
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Targets are shifted inputs (next token prediction)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        dataset.append((inputs, targets))

    return dataset


def create_distributed_dataset(
    vocab_size: int,
    seq_len: int,
    num_batches: int,
    batch_size: int,
    rank: int,
    world_size: int,
    seed: int = 42
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create dataset shard for distributed training.

    Each worker gets non-overlapping data using interleaved sharding.
    The effective global batch size is batch_size * world_size.

    Args:
        vocab_size: Size of vocabulary
        seq_len: Sequence length
        num_batches: Number of batches per worker
        batch_size: Batch size per worker (local batch size)
        rank: Worker rank (0 to world_size-1)
        world_size: Total number of workers
        seed: Random seed for reproducibility across workers

    Returns:
        List of (input, target) batches for this worker

    Example:
        With 4 workers, batch_size=8, num_batches=10:
        - Each worker gets 10 batches of 8 samples
        - Total samples per worker: 80
        - Total samples across all workers: 320
        - Effective global batch size per step: 32 (8 * 4)

        Worker 0 gets global samples: [0, 4, 8, 12, 16, 20, ...]
        Worker 1 gets global samples: [1, 5, 9, 13, 17, 21, ...]
        Worker 2 gets global samples: [2, 6, 10, 14, 18, 22, ...]
        Worker 3 gets global samples: [3, 7, 11, 15, 19, 23, ...]
    """
    # Set seed for reproducible data generation
    # All workers use the same seed to generate the same "global dataset"
    torch.manual_seed(seed)

    # Calculate total samples needed across all workers
    total_samples = num_batches * batch_size * world_size

    # Generate all data indices (workers will take interleaved subsets)
    all_indices = torch.arange(total_samples)

    # Optional: Shuffle with fixed seed (all workers do this identically)
    # This ensures workers agree on the global data ordering
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_samples, generator=generator)
    shuffled_indices = all_indices[perm]

    # Each worker takes every world_size-th sample, starting at rank
    # Worker 0: indices [0, 4, 8, 12, ...]
    # Worker 1: indices [1, 5, 9, 13, ...]
    worker_indices = shuffled_indices[rank::world_size]

    # Verify we got the right number of samples
    expected_samples = num_batches * batch_size
    assert len(worker_indices) == expected_samples, \
        f"Worker {rank}: expected {expected_samples} samples, got {len(worker_indices)}"

    # Create batches from this worker's indices
    dataset = []
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_indices = worker_indices[start:end]

        # Generate data for these specific indices
        # Use indices as seeds to ensure deterministic generation
        inputs = torch.empty((batch_size, seq_len), dtype=torch.long)
        targets = torch.empty((batch_size, seq_len), dtype=torch.long)

        for j, idx in enumerate(batch_indices):
            # Use index as seed for this specific sample
            sample_generator = torch.Generator().manual_seed(seed + idx.item())
            inputs[j] = torch.randint(
                0, vocab_size, (seq_len,),
                generator=sample_generator
            )
            targets[j] = torch.randint(
                0, vocab_size, (seq_len,),
                generator=sample_generator
            )

        dataset.append((inputs, targets))

    return dataset


# ============================================================================
# HuggingFace Dataset Support
# ============================================================================

# Dataset configurations for popular LLM training datasets
DATASET_CONFIGS = {
    'fineweb': {
        'full_name': 'HuggingFaceFW/fineweb',
        'streaming': True,
        'buffer_size': 100_000,
        'tokenizer': 'gpt2',
        'seq_len': 1024,
        'description': 'FineWeb: 15T tokens from CommonCrawl'
    },
    'fineweb-edu': {
        'full_name': 'HuggingFaceFW/fineweb-edu',
        'streaming': True,
        'buffer_size': 50_000,
        'tokenizer': 'gpt2',
        'seq_len': 1024,
        'description': 'FineWeb-Edu: 1.3T high-quality educational tokens'
    },
    'pile': {
        'full_name': 'EleutherAI/pile',
        'streaming': True,
        'buffer_size': 50_000,
        'tokenizer': 'EleutherAI/gpt-neox-20b',
        'seq_len': 2048,
        'description': 'The Pile: 825GB diverse dataset (22 sources)'
    },
    'tiny_shakespeare': {
        'full_name': 'karpathy/tiny_shakespeare',
        'streaming': False,  # Small dataset, can load fully
        'buffer_size': 1_000,
        'tokenizer': 'gpt2',
        'seq_len': 256,
        'description': 'Tiny Shakespeare: 40K lines for testing'
    },
    'shakespeare': {
        'full_name': 'benchaffe/shakespeare-lines',
        'streaming': False,
        'buffer_size': 5_000,
        'tokenizer': 'gpt2',
        'seq_len': 512,
        'description': 'Complete Works of Shakespeare, line-by-line'
    }
}


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get configuration for a dataset.

    Args:
        dataset_name: Short name or full HuggingFace name

    Returns:
        Configuration dictionary with settings

    Example:
        >>> config = get_dataset_config('fineweb-edu')
        >>> print(config['full_name'])
        'HuggingFaceFW/fineweb-edu'
    """
    # Check if it's a known short name
    if dataset_name in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset_name].copy()

    # Otherwise assume it's a full HuggingFace name
    return {
        'full_name': dataset_name,
        'streaming': True,
        'buffer_size': 10_000,
        'tokenizer': 'gpt2',
        'seq_len': 512,
        'description': f'Custom dataset: {dataset_name}'
    }


def create_huggingface_dataset(
    dataset_name: str,
    rank: int,
    world_size: int,
    num_batches: int,
    batch_size: int,
    tokenizer_name: Optional[str] = None,
    seq_len: Optional[int] = None,
    streaming: Optional[bool] = None,
    seed: int = 42,
    buffer_size: Optional[int] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create HuggingFace dataset shard for distributed training.

    Each worker loads a non-overlapping shard of the dataset based on rank.
    Supports streaming for large datasets and proper tokenization.

    Args:
        dataset_name: Dataset name (short name or full HuggingFace path)
                     Examples: 'fineweb-edu', 'pile', 'HuggingFaceFW/fineweb'
        rank: Worker rank (0 to world_size-1)
        world_size: Total number of workers
        num_batches: Number of batches to load per worker
        batch_size: Batch size per worker (local batch size)
        tokenizer_name: Optional tokenizer override (uses dataset config default if None)
        seq_len: Optional sequence length override
        streaming: Optional streaming mode override
        seed: Random seed for reproducibility
        buffer_size: Shuffle buffer size override

    Returns:
        List of (input_ids, labels) batches for this worker

    Example:
        >>> dataset = create_huggingface_dataset(
        ...     dataset_name='fineweb-edu',
        ...     rank=0,
        ...     world_size=4,
        ...     num_batches=100,
        ...     batch_size=8
        ... )
        >>> print(f"Loaded {len(dataset)} batches")

    Note:
        Requires: pip install datasets transformers
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "HuggingFace datasets and transformers are required. "
            "Install with: pip install datasets transformers"
        )

    # Get dataset configuration
    config = get_dataset_config(dataset_name)

    # Override config with provided values
    if tokenizer_name is not None:
        config['tokenizer'] = tokenizer_name
    if seq_len is not None:
        config['seq_len'] = seq_len
    if streaming is not None:
        config['streaming'] = streaming
    if buffer_size is not None:
        config['buffer_size'] = buffer_size

    logger.info(
        f"Loading dataset '{config['full_name']}' for rank {rank}/{world_size} "
        f"(streaming={config['streaming']}, seq_len={config['seq_len']})"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset(
        config['full_name'],
        split='train',
        streaming=config['streaming']
    )

    # Shuffle with consistent seed across all workers
    dataset = dataset.shuffle(seed=seed, buffer_size=config['buffer_size'])

    # Shard by worker rank (each worker gets every world_size-th sample)
    dataset = dataset.shard(num_shards=world_size, index=rank)

    logger.info(f"Dataset sharded for rank {rank}/{world_size}")

    # Tokenization function
    def tokenize_function(examples):
        """Tokenize text for causal language modeling."""
        result = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=config['seq_len'],
            return_tensors='pt'
        )
        # For causal LM, labels = input_ids (shifted internally by model)
        result['labels'] = result['input_ids'].clone()
        return result

    # Apply tokenization with batching (much faster than per-example)
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=['text']  # Remove text column to save memory
    )

    # Convert to PyTorch format
    dataset = dataset.with_format("torch")

    # Collect specified number of batches
    batches = []
    sample_count = 0
    target_samples = num_batches * batch_size

    logger.info(f"Collecting {num_batches} batches of size {batch_size}...")

    current_batch_inputs = []
    current_batch_labels = []

    for sample in dataset:
        current_batch_inputs.append(sample['input_ids'])
        current_batch_labels.append(sample['labels'])

        # When we have batch_size samples, create a batch
        if len(current_batch_inputs) == batch_size:
            batch_inputs = torch.stack(current_batch_inputs)
            batch_labels = torch.stack(current_batch_labels)
            batches.append((batch_inputs, batch_labels))

            current_batch_inputs = []
            current_batch_labels = []

            sample_count += batch_size

            # Stop when we have enough batches
            if len(batches) >= num_batches:
                break

            # Progress logging
            if len(batches) % 10 == 0:
                logger.info(f"  Collected {len(batches)}/{num_batches} batches...")

    # Handle any remaining samples (incomplete last batch)
    if current_batch_inputs and len(batches) < num_batches:
        # Pad to batch_size if needed
        while len(current_batch_inputs) < batch_size:
            current_batch_inputs.append(current_batch_inputs[0])  # Duplicate first sample
            current_batch_labels.append(current_batch_labels[0])

        batch_inputs = torch.stack(current_batch_inputs)
        batch_labels = torch.stack(current_batch_labels)
        batches.append((batch_inputs, batch_labels))

    logger.info(
        f"Dataset loading complete: {len(batches)} batches "
        f"(effective global batch size: {batch_size * world_size})"
    )

    return batches


def list_available_datasets() -> List[Dict[str, str]]:
    """
    List all available preconfigured datasets.

    Returns:
        List of dataset information dictionaries

    Example:
        >>> datasets = list_available_datasets()
        >>> for ds in datasets:
        ...     print(f"{ds['name']}: {ds['description']}")
    """
    return [
        {
            'name': name,
            'full_name': config['full_name'],
            'description': config['description'],
            'seq_len': config['seq_len'],
            'streaming': config['streaming']
        }
        for name, config in DATASET_CONFIGS.items()
    ]
