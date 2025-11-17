"""
Dataset utilities for testing and training.
"""

from typing import List, Tuple
import torch


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
