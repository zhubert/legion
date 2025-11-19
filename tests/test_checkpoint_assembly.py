"""
Integration tests for checkpoint assembly.

Tests the full checkpoint workflow:
1. Workers save shards with partition metadata
2. Coordinator creates metadata.json
3. Assembler reconstructs complete model
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import json
import tempfile
import shutil

from sim.model import TinyGPT
from core.partitioner import Partitioner
from worker.shard_manager import ShardManager
from worker.assembler import CheckpointAssembler, create_checkpoint_metadata


@pytest.fixture
def model():
    """Create a tiny model for testing."""
    return TinyGPT(
        vocab_size=100,
        d_model=32,
        n_heads=2,
        n_layers=2,
        max_seq_len=64
    )


@pytest.fixture
def checkpoint_dir():
    """Create temporary checkpoint directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_shard_manager_save_to_checkpoint(model, checkpoint_dir):
    """Test ShardManager.save_shard_to_checkpoint()."""
    # Create partitioner
    world_size = 2
    partitioner = Partitioner(model, world_size)

    # Create shard manager for worker 0
    partition = partitioner.get_partition(0)
    shard_manager = ShardManager(
        worker_id="worker_0",
        model=model,
        shard_start=partition.start_idx,
        shard_end=partition.end_idx,
        checkpoint_dir=checkpoint_dir,
        rank=0,
        world_size=world_size,
        partition=partition
    )

    # Load shard
    shard_manager.load_shard()

    # Save to checkpoint
    global_step = 100
    rank = 0
    model_config = {
        "vocab_size": 100,
        "d_model": 32,
        "n_heads": 2
    }

    shard_path = shard_manager.save_shard_to_checkpoint(
        checkpoint_dir=checkpoint_dir,
        global_step=global_step,
        rank=rank,
        model_config=model_config
    )

    # Verify file exists
    assert Path(shard_path).exists()

    # Load and verify contents
    shard_data = torch.load(shard_path)
    assert shard_data['worker_id'] == "worker_0"
    assert shard_data['rank'] == rank
    assert shard_data['global_step'] == global_step
    assert 'parameters' in shard_data
    assert 'partition' in shard_data
    assert shard_data['model_config'] == model_config

    # Verify partition metadata
    partition_meta = shard_data['partition']
    assert partition_meta['shard_start'] == partition.start_idx
    assert partition_meta['shard_end'] == partition.end_idx
    assert partition_meta['world_size'] == world_size


def test_create_checkpoint_metadata(checkpoint_dir):
    """Test create_checkpoint_metadata()."""
    global_step = 100
    workers = [
        {
            'rank': 0,
            'worker_id': 'worker_0',
            'shard_file': 'shard_rank_0.pt'
        },
        {
            'rank': 1,
            'worker_id': 'worker_1',
            'shard_file': 'shard_rank_1.pt'
        }
    ]
    model_config = {'d_model': 32}

    metadata_path = create_checkpoint_metadata(
        checkpoint_dir=checkpoint_dir,
        global_step=global_step,
        workers=workers,
        model_config=model_config
    )

    # Verify file exists
    assert Path(metadata_path).exists()

    # Load and verify contents
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    assert metadata['global_step'] == global_step
    assert metadata['num_workers'] == 2
    assert metadata['workers'] == workers
    assert metadata['model_config'] == model_config
    assert metadata['partition_scheme'] == 'zero3'
    assert metadata['status'] == 'shards_saved'


def test_checkpoint_assembly_two_workers(model, checkpoint_dir):
    """Test full checkpoint assembly with 2 workers."""
    world_size = 2
    global_step = 100
    partitioner = Partitioner(model, world_size)

    # Save original model state for comparison
    original_state = {
        name: param.data.clone()
        for name, param in model.named_parameters()
    }

    # Simulate workers saving shards
    worker_info = []
    for rank in range(world_size):
        partition = partitioner.get_partition(rank)

        # Create shard manager
        shard_manager = ShardManager(
            worker_id=f"worker_{rank}",
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

        # Save to checkpoint
        shard_manager.save_shard_to_checkpoint(
            checkpoint_dir=checkpoint_dir,
            global_step=global_step,
            rank=rank,
            model_config={'d_model': 32}
        )

        worker_info.append({
            'rank': rank,
            'worker_id': f"worker_{rank}",
            'shard_file': f"shard_rank_{rank}.pt"
        })

    # Create metadata
    create_checkpoint_metadata(
        checkpoint_dir=checkpoint_dir,
        global_step=global_step,
        workers=worker_info,
        model_config={'d_model': 32}
    )

    # Assemble checkpoint
    assembler = CheckpointAssembler(checkpoint_dir)
    assembled_path = assembler.assemble_checkpoint(global_step)

    # Verify assembled checkpoint exists
    assert Path(assembled_path).exists()

    # Load assembled checkpoint
    checkpoint = torch.load(assembled_path)
    assert 'model_state_dict' in checkpoint
    assert checkpoint['global_step'] == global_step
    assert checkpoint['num_workers'] == world_size

    # Verify reconstructed parameters match original
    state_dict = checkpoint['model_state_dict']

    # Note: We can't directly compare parameter names because the assembler
    # reconstructs flat parameters. We'll verify the total parameter count.
    total_params_original = sum(p.numel() for p in model.parameters())
    total_params_reconstructed = sum(p.numel() for p in state_dict.values())

    assert total_params_original == total_params_reconstructed

    # Verify each flat parameter piece matches original when concatenated
    flat_original = torch.cat([p.data.flatten() for p in model.parameters()])
    flat_reconstructed = torch.cat([p.flatten() for p in state_dict.values()])

    assert flat_original.shape == flat_reconstructed.shape
    assert torch.allclose(flat_original, flat_reconstructed, atol=1e-6)


def test_checkpoint_assembly_validation(model, checkpoint_dir):
    """Test assembler validation catches errors."""
    world_size = 2
    global_step = 100
    partitioner = Partitioner(model, world_size)

    # Save only worker 0 shard (missing worker 1)
    partition = partitioner.get_partition(0)
    shard_manager = ShardManager(
        worker_id="worker_0",
        model=model,
        shard_start=partition.start_idx,
        shard_end=partition.end_idx,
        checkpoint_dir=checkpoint_dir,
        rank=0,
        world_size=world_size,
        partition=partition
    )
    shard_manager.load_shard()
    shard_manager.save_shard_to_checkpoint(
        checkpoint_dir=checkpoint_dir,
        global_step=global_step,
        rank=0
    )

    # Create metadata claiming 2 workers
    worker_info = [
        {'rank': 0, 'worker_id': 'worker_0', 'shard_file': 'shard_rank_0.pt'},
        {'rank': 1, 'worker_id': 'worker_1', 'shard_file': 'shard_rank_1.pt'}
    ]
    create_checkpoint_metadata(
        checkpoint_dir=checkpoint_dir,
        global_step=global_step,
        workers=worker_info
    )

    # Assembler should fail validation (missing shard)
    assembler = CheckpointAssembler(checkpoint_dir)
    with pytest.raises(FileNotFoundError, match="Shard not found.*rank 1"):
        assembler.assemble_checkpoint(global_step)


def test_checkpoint_list_and_status(model, checkpoint_dir):
    """Test checkpoint listing and status queries."""
    world_size = 2
    partitioner = Partitioner(model, world_size)
    assembler = CheckpointAssembler(checkpoint_dir)

    # Create two checkpoints at different steps
    for global_step in [100, 200]:
        # Save shards
        for rank in range(world_size):
            partition = partitioner.get_partition(rank)
            shard_manager = ShardManager(
                worker_id=f"worker_{rank}",
                model=model,
                shard_start=partition.start_idx,
                shard_end=partition.end_idx,
                checkpoint_dir=checkpoint_dir,
                rank=rank,
                world_size=world_size,
                partition=partition
            )
            shard_manager.load_shard()
            shard_manager.save_shard_to_checkpoint(
                checkpoint_dir=checkpoint_dir,
                global_step=global_step,
                rank=rank
            )

        # Create metadata
        worker_info = [
            {'rank': r, 'worker_id': f'worker_{r}', 'shard_file': f'shard_rank_{r}.pt'}
            for r in range(world_size)
        ]
        create_checkpoint_metadata(
            checkpoint_dir=checkpoint_dir,
            global_step=global_step,
            workers=worker_info
        )

        # Assemble
        assembler.assemble_checkpoint(global_step)

    # List checkpoints
    checkpoints = assembler.list_checkpoints()
    assert len(checkpoints) == 2
    assert checkpoints[0]['global_step'] in [100, 200]
    assert checkpoints[1]['global_step'] in [100, 200]

    # Check individual status
    status_100 = assembler.get_checkpoint_status(100)
    assert status_100 is not None
    assert status_100['global_step'] == 100
    assert status_100['status'] == 'assembled'
    assert status_100['assembled_exists'] is True

    status_200 = assembler.get_checkpoint_status(200)
    assert status_200 is not None
    assert status_200['global_step'] == 200


def test_checkpoint_assembly_four_workers(model, checkpoint_dir):
    """Test checkpoint assembly with 4 workers (more complex partitioning)."""
    world_size = 4
    global_step = 500
    partitioner = Partitioner(model, world_size)

    # Save shards from all workers
    for rank in range(world_size):
        partition = partitioner.get_partition(rank)
        shard_manager = ShardManager(
            worker_id=f"worker_{rank}",
            model=model,
            shard_start=partition.start_idx,
            shard_end=partition.end_idx,
            checkpoint_dir=checkpoint_dir,
            rank=rank,
            world_size=world_size,
            partition=partition
        )
        shard_manager.load_shard()
        shard_manager.save_shard_to_checkpoint(
            checkpoint_dir=checkpoint_dir,
            global_step=global_step,
            rank=rank
        )

    # Create metadata
    worker_info = [
        {'rank': r, 'worker_id': f'worker_{r}', 'shard_file': f'shard_rank_{r}.pt'}
        for r in range(world_size)
    ]
    create_checkpoint_metadata(
        checkpoint_dir=checkpoint_dir,
        global_step=global_step,
        workers=worker_info
    )

    # Assemble
    assembler = CheckpointAssembler(checkpoint_dir)
    assembled_path = assembler.assemble_checkpoint(global_step)

    # Verify
    assert Path(assembled_path).exists()
    checkpoint = torch.load(assembled_path)
    assert checkpoint['num_workers'] == world_size

    # Verify parameter reconstruction
    flat_original = torch.cat([p.data.flatten() for p in model.parameters()])
    flat_reconstructed = torch.cat([
        p.flatten() for p in checkpoint['model_state_dict'].values()
    ])
    assert torch.allclose(flat_original, flat_reconstructed, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
