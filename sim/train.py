"""
Main Training Script for Legion PoC

Simulates distributed training across multiple workers on a single machine.
"""

import argparse
import time
from typing import Tuple
import torch
import torch.nn as nn

from core.model import create_model
from core.partitioner import Partitioner
from core.dataset import create_dummy_dataset
from sim.collectives import CollectiveCoordinator
from sim.worker import WorkerCoordinator


def train_distributed(
    model_size: str = "tiny",
    world_size: int = 4,
    num_steps: int = 100,
    batch_size: int = 4,
    seq_len: int = 32,
    learning_rate: float = 0.001,
    compression: str = "none",
    latency_ms: float = 0.0,
    device: str = "cpu"
) -> dict:
    """
    Run distributed training simulation.

    Args:
        model_size: Model size ('tiny', 'small', 'medium')
        world_size: Number of workers
        num_steps: Training steps
        batch_size: Batch size per worker
        seq_len: Sequence length
        learning_rate: Learning rate
        compression: Compression method
        latency_ms: Simulated network latency in milliseconds
        device: Device to use

    Returns:
        Dictionary of training results
    """
    print(f"\n{'='*60}")
    print(f"Legion Distributed Training Simulation")
    print(f"{'='*60}")
    print(f"Model size: {model_size}")
    print(f"Workers: {world_size}")
    print(f"Steps: {num_steps}")
    print(f"Batch size per worker: {batch_size}")
    print(f"Compression: {compression}")
    print(f"Simulated latency: {latency_ms}ms")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # 1. Create model
    print("Creating model...")
    model = create_model(model_size)
    vocab_size = model.vocab_size
    print(f"  Parameters: {model.count_parameters():,}")

    # 2. Partition model across workers
    print(f"\nPartitioning model across {world_size} workers...")
    partitioner = Partitioner(model, world_size=world_size)
    partitioner.print_partition_info()

    # 3. Create collective communication coordinator
    print("\nSetting up collective communication...")
    max_tensor_size = max(p.num_params for p in partitioner.partitions)
    collective_coordinator = CollectiveCoordinator(world_size, max_tensor_size)

    # 4. Create worker coordinator
    print("Initializing workers...")

    def collective_ops_factory(rank):
        return collective_coordinator.get_collective_ops(rank, latency_ms=latency_ms)

    worker_coordinator = WorkerCoordinator(
        world_size=world_size,
        model=model,
        partitions=partitioner.partitions,
        collective_ops_factory=collective_ops_factory,
        learning_rate=learning_rate,
        compression=compression,
        device=device
    )

    print(f"  Created {world_size} workers")

    # 5. Create dataset
    print(f"\nGenerating dataset ({num_steps} batches)...")
    dataset = create_dummy_dataset(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_batches=num_steps,
        batch_size=batch_size
    )

    # 6. Training loop
    print(f"\nStarting training...\n")
    start_time = time.time()

    losses = []
    for step, batch in enumerate(dataset):
        # Each worker gets the same batch (data parallelism)
        # In real system, workers would have different batches
        batches = [batch for _ in range(world_size)]

        # Execute training step
        metrics = worker_coordinator.train_step(batches)

        # Collect loss from first worker (all should be similar)
        loss = metrics[0]['loss']
        losses.append(loss)

        # Print progress
        if (step + 1) % 10 == 0 or step == 0:
            avg_loss = sum(losses[-10:]) / len(losses[-10:])
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            print(f"Step {step+1:3d}/{num_steps} | "
                  f"Loss: {loss:.4f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Speed: {steps_per_sec:.2f} steps/s")

    total_time = time.time() - start_time

    # 7. Print statistics
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Steps/sec: {num_steps / total_time:.2f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Loss reduction: {losses[0] - losses[-1]:.4f}")

    # Worker statistics
    print(f"\n{'='*60}")
    print("Worker Statistics (Averages)")
    print(f"{'='*60}")

    all_stats = worker_coordinator.get_all_stats()
    if all_stats and all_stats[0]:
        # Print stats from first worker (all should be similar)
        stats = all_stats[0]
        print(f"Forward pass:     {stats.get('avg_forward_time', 0)*1000:.2f}ms")
        print(f"Backward pass:    {stats.get('avg_backward_time', 0)*1000:.2f}ms")
        print(f"All-gather:       {stats.get('avg_all_gather_time', 0)*1000:.2f}ms")
        print(f"Reduce-scatter:   {stats.get('avg_reduce_scatter_time', 0)*1000:.2f}ms")
        print(f"Compression:      {stats.get('avg_compression_time', 0)*1000:.2f}ms")
        print(f"Parameter update: {stats.get('avg_update_time', 0)*1000:.2f}ms")

    print(f"\n{'='*60}\n")

    return {
        'losses': losses,
        'total_time': total_time,
        'final_loss': losses[-1],
        'initial_loss': losses[0],
        'worker_stats': all_stats
    }


def train_baseline(
    model_size: str = "tiny",
    num_steps: int = 100,
    batch_size: int = 4,
    seq_len: int = 32,
    learning_rate: float = 0.001,
    device: str = "cpu"
) -> dict:
    """
    Run baseline single-GPU training for comparison.

    Args:
        model_size: Model size
        num_steps: Training steps
        batch_size: Batch size
        seq_len: Sequence length
        learning_rate: Learning rate
        device: Device to use

    Returns:
        Dictionary of training results
    """
    print(f"\n{'='*60}")
    print(f"Baseline Single-GPU Training")
    print(f"{'='*60}")
    print(f"Model size: {model_size}")
    print(f"Steps: {num_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Create model
    print("Creating model...")
    model = create_model(model_size).to(device)
    vocab_size = model.vocab_size
    print(f"  Parameters: {model.count_parameters():,}")

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Create dataset
    print(f"\nGenerating dataset ({num_steps} batches)...")
    dataset = create_dummy_dataset(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_batches=num_steps,
        batch_size=batch_size
    )

    # Training loop
    print(f"\nStarting training...\n")
    start_time = time.time()

    losses = []
    for step, (inputs, targets) in enumerate(dataset):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward
        model.train()
        logits, loss = model(inputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Print progress
        if (step + 1) % 10 == 0 or step == 0:
            avg_loss = sum(losses[-10:]) / len(losses[-10:])
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            print(f"Step {step+1:3d}/{num_steps} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Speed: {steps_per_sec:.2f} steps/s")

    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Steps/sec: {num_steps / total_time:.2f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Loss reduction: {losses[0] - losses[-1]:.4f}")
    print(f"\n{'='*60}\n")

    return {
        'losses': losses,
        'total_time': total_time,
        'final_loss': losses[-1],
        'initial_loss': losses[0]
    }


def main():
    parser = argparse.ArgumentParser(description="Legion Distributed Training PoC")

    parser.add_argument(
        '--mode',
        type=str,
        default='distributed',
        choices=['distributed', 'baseline', 'both'],
        help='Training mode'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='tiny',
        choices=['tiny', 'small', 'medium'],
        help='Model size'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of workers for distributed training'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=100,
        help='Number of training steps'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size per worker'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='none',
        choices=['none', 'int8', 'topk'],
        help='Gradient compression method'
    )
    parser.add_argument(
        '--latency',
        type=float,
        default=0.0,
        help='Simulated network latency in milliseconds'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use'
    )

    args = parser.parse_args()

    # Run training
    if args.mode == 'distributed' or args.mode == 'both':
        distributed_results = train_distributed(
            model_size=args.model,
            world_size=args.workers,
            num_steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            compression=args.compression,
            latency_ms=args.latency,
            device=args.device
        )

    if args.mode == 'baseline' or args.mode == 'both':
        baseline_results = train_baseline(
            model_size=args.model,
            num_steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device
        )

    # Compare results if both were run
    if args.mode == 'both':
        print(f"\n{'='*60}")
        print("Comparison: Distributed vs Baseline")
        print(f"{'='*60}")
        print(f"Final loss:")
        print(f"  Distributed: {distributed_results['final_loss']:.4f}")
        print(f"  Baseline:    {baseline_results['final_loss']:.4f}")
        print(f"  Difference:  {abs(distributed_results['final_loss'] - baseline_results['final_loss']):.4f}")
        print(f"\nTraining time:")
        print(f"  Distributed: {distributed_results['total_time']:.2f}s")
        print(f"  Baseline:    {baseline_results['total_time']:.2f}s")
        print(f"  Overhead:    {(distributed_results['total_time'] / baseline_results['total_time'] - 1) * 100:.1f}%")
        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
