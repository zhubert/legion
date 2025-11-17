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
