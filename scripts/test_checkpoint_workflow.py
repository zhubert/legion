"""
Test script for end-to-end checkpoint workflow.

Demonstrates:
1. Workers training and saving distributed checkpoint shards
2. Coordinator triggering checkpoint assembly
3. Assembler reconstructing complete model
"""

import asyncio
import logging
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model import create_model
from core.partitioner import Partitioner
from worker.shard_manager import ShardManager
from worker.assembler import CheckpointAssembler, create_checkpoint_metadata


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def simulate_worker_training(
    worker_id: str,
    rank: int,
    world_size: int,
    checkpoint_dir: str,
    global_step: int,
    reference_model: torch.nn.Module
):
    """
    Simulate a worker training and saving a checkpoint shard.

    Args:
        worker_id: Worker identifier
        rank: Worker rank
        world_size: Total number of workers
        checkpoint_dir: Checkpoint directory
        global_step: Training step to checkpoint
        reference_model: Reference model to use for consistent parameters
    """
    logger.info(f"Worker {worker_id} (rank {rank}): Starting training simulation...")

    # Use the reference model (same instance for all workers)
    model = reference_model

    # Partition model
    partitioner = Partitioner(model, world_size)
    partition = partitioner.get_partition(rank)

    # Create shard manager
    shard_manager = ShardManager(
        worker_id=worker_id,
        model=model,
        shard_start=partition.start_idx,
        shard_end=partition.end_idx,
        checkpoint_dir=checkpoint_dir,
        rank=rank,
        world_size=world_size,
        partition=partition
    )

    # Load shard
    shard_manager.load_shard()

    # Simulate training (in reality, parameters would be updated)
    logger.info(f"Worker {worker_id}: Training for {global_step} steps...")

    # Save checkpoint shard
    logger.info(f"Worker {worker_id}: Saving checkpoint shard at step {global_step}...")
    shard_path = shard_manager.save_shard_to_checkpoint(
        checkpoint_dir=checkpoint_dir,
        global_step=global_step,
        rank=rank,
        model_config={
            'model_size': 'tiny',
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 4
        }
    )

    logger.info(f"Worker {worker_id}: Checkpoint shard saved to {shard_path}")
    return shard_path


async def coordinator_trigger_assembly(
    checkpoint_dir: str,
    global_step: int,
    num_workers: int
):
    """
    Simulate coordinator triggering checkpoint assembly.

    Args:
        checkpoint_dir: Checkpoint directory
        global_step: Training step
        num_workers: Number of workers
    """
    logger.info(f"Coordinator: Creating checkpoint metadata for step {global_step}...")

    # Create metadata
    worker_info = [
        {
            'rank': rank,
            'worker_id': f'worker_{rank}',
            'shard_file': f'shard_rank_{rank}.pt'
        }
        for rank in range(num_workers)
    ]

    metadata_path = create_checkpoint_metadata(
        checkpoint_dir=checkpoint_dir,
        global_step=global_step,
        workers=worker_info,
        model_config={'model_size': 'tiny'}
    )

    logger.info(f"Coordinator: Metadata created at {metadata_path}")

    # Trigger assembler
    logger.info(f"Coordinator: Triggering checkpoint assembly...")
    assembler = CheckpointAssembler(checkpoint_dir)

    try:
        assembled_path = assembler.assemble_checkpoint(global_step)
        logger.info(f"Coordinator: Checkpoint assembled successfully at {assembled_path}")

        # Verify assembled checkpoint
        checkpoint = torch.load(assembled_path)
        logger.info(f"Assembled checkpoint contains:")
        logger.info(f"  - Global step: {checkpoint['global_step']}")
        logger.info(f"  - Num workers: {checkpoint['num_workers']}")
        logger.info(f"  - Parameters: {len(checkpoint['model_state_dict'])}")

        total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
        logger.info(f"  - Total parameters: {total_params:,}")

        return assembled_path

    except Exception as e:
        logger.error(f"Coordinator: Assembly failed: {e}")
        raise


async def main():
    """Main test workflow."""
    logger.info("=" * 70)
    logger.info("Legion Checkpoint Workflow Test")
    logger.info("=" * 70)

    # Configuration
    checkpoint_dir = "./test_checkpoints"
    global_step = 100
    world_size = 2

    # Clean up old test checkpoints
    import shutil
    if Path(checkpoint_dir).exists():
        shutil.rmtree(checkpoint_dir)

    logger.info(f"\nConfiguration:")
    logger.info(f"  Checkpoint dir: {checkpoint_dir}")
    logger.info(f"  Global step: {global_step}")
    logger.info(f"  World size: {world_size}")
    logger.info("")

    # Create a reference model ONCE (before workers)
    # This will be used for comparison
    logger.info("Creating reference model...")
    reference_model = create_model("tiny")
    reference_params = torch.cat([p.data.clone().flatten() for p in reference_model.parameters()])
    logger.info(f"Reference model has {reference_params.numel():,} parameters")

    # Phase 1: Workers save checkpoint shards
    logger.info("Phase 1: Workers Training & Saving Shards")
    logger.info("-" * 70)

    worker_tasks = []
    for rank in range(world_size):
        worker_id = f"worker_{rank}"
        task = simulate_worker_training(
            worker_id=worker_id,
            rank=rank,
            world_size=world_size,
            checkpoint_dir=checkpoint_dir,
            global_step=global_step,
            reference_model=reference_model
        )
        worker_tasks.append(task)

    # Run workers in parallel
    shard_paths = await asyncio.gather(*worker_tasks)
    logger.info(f"\nAll {world_size} workers saved their shards")

    # Phase 2: Coordinator triggers assembly
    logger.info("\n" + "=" * 70)
    logger.info("Phase 2: Coordinator Assembling Checkpoint")
    logger.info("-" * 70)

    assembled_path = await coordinator_trigger_assembly(
        checkpoint_dir=checkpoint_dir,
        global_step=global_step,
        num_workers=world_size
    )

    # Phase 3: Verify assembly correctness
    logger.info("\n" + "=" * 70)
    logger.info("Phase 3: Verifying Assembly Correctness")
    logger.info("-" * 70)

    # Load assembled model
    checkpoint = torch.load(assembled_path)
    assembled_params = torch.cat([
        p.flatten() for p in checkpoint['model_state_dict'].values()
    ])

    # Compare with reference model
    logger.info(f"Reference params shape: {reference_params.shape}")
    logger.info(f"Assembled params shape: {assembled_params.shape}")

    if torch.allclose(reference_params, assembled_params, atol=1e-6):
        logger.info("✓ Assembly verification PASSED")
        logger.info(f"✓ Reconstructed parameters match original")
    else:
        logger.error("✗ Assembly verification FAILED")
        logger.error(f"✗ Parameter mismatch detected")

        # Debug info
        diff = torch.abs(reference_params - assembled_params)
        logger.error(f"  Max difference: {diff.max().item()}")
        logger.error(f"  Mean difference: {diff.mean().item()}")
        return 1

    logger.info("\n" + "=" * 70)
    logger.info("Checkpoint Workflow Test PASSED")
    logger.info("=" * 70)

    # Clean up
    logger.info(f"\nCleaning up test checkpoints...")
    shutil.rmtree(checkpoint_dir)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
