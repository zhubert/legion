"""
Checkpoint assembler for reconstructing complete models from distributed shards.

The assembler reads parameter shards saved by workers and reconstructs the
complete model state_dict. This enables exporting trained models for inference
or fine-tuning.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import torch

logger = logging.getLogger(__name__)


class CheckpointAssembler:
    """
    Assembles complete model checkpoints from distributed parameter shards.

    Workers save their parameter shards to a shared checkpoint directory.
    The assembler reads all shards, validates consistency, and reconstructs
    the complete model state_dict.
    """

    def __init__(self, checkpoint_base_dir: str = "./checkpoints"):
        """
        Initialize checkpoint assembler.

        Args:
            checkpoint_base_dir: Base directory containing checkpoints
        """
        self.checkpoint_base_dir = Path(checkpoint_base_dir)
        self.checkpoint_base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint assembler initialized: {self.checkpoint_base_dir}")

    def assemble_checkpoint(
        self,
        global_step: int,
        output_name: str = "assembled_model.pt",
        validate: bool = True
    ) -> str:
        """
        Assemble complete model checkpoint from shards.

        Args:
            global_step: Training step to assemble
            output_name: Name of output file (default: 'assembled_model.pt')
            validate: Whether to validate shard consistency

        Returns:
            Path to assembled checkpoint file

        Raises:
            FileNotFoundError: If checkpoint directory or shards not found
            ValueError: If shard validation fails
        """
        step_dir = self.checkpoint_base_dir / f"step_{global_step}"
        if not step_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {step_dir}")

        logger.info(f"Assembling checkpoint from: {step_dir}")

        # Load metadata
        metadata_path = step_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load all shards
        shards = self._load_shards(step_dir, metadata)
        logger.info(f"Loaded {len(shards)} shards")

        # Validate shards
        if validate:
            self._validate_shards(shards, metadata)
            logger.info("Shard validation passed")

        # Reconstruct full model
        assembled_state = self._reconstruct_model(shards, metadata)
        logger.info(f"Reconstructed model with {len(assembled_state)} parameters")

        # Save assembled checkpoint
        output_path = step_dir / output_name
        self._save_assembled_checkpoint(
            output_path,
            assembled_state,
            metadata,
            shards
        )

        # Update metadata status
        self._update_metadata_status(metadata_path, "assembled", str(output_path))

        logger.info(f"Checkpoint assembled successfully: {output_path}")
        return str(output_path)

    def _load_shards(
        self,
        step_dir: Path,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Load all parameter shards from checkpoint directory.

        Args:
            step_dir: Checkpoint step directory
            metadata: Checkpoint metadata

        Returns:
            List of loaded shard data dictionaries

        Raises:
            FileNotFoundError: If expected shard not found
        """
        shards = []
        num_workers = metadata['num_workers']

        for worker_info in metadata['workers']:
            rank = worker_info['rank']
            shard_file = worker_info['shard_file']
            shard_path = step_dir / shard_file

            if not shard_path.exists():
                raise FileNotFoundError(
                    f"Shard not found: {shard_path} (rank {rank})"
                )

            logger.debug(f"Loading shard: {shard_path}")
            shard_data = torch.load(shard_path, map_location='cpu')
            shards.append(shard_data)

        # Sort by rank to ensure correct order
        shards.sort(key=lambda s: s['rank'])

        return shards

    def _validate_shards(
        self,
        shards: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ):
        """
        Validate shard consistency.

        Args:
            shards: Loaded shard data
            metadata: Checkpoint metadata

        Raises:
            ValueError: If validation fails
        """
        num_workers = metadata['num_workers']

        # Check we have all shards
        if len(shards) != num_workers:
            raise ValueError(
                f"Expected {num_workers} shards, found {len(shards)}"
            )

        # Check ranks are sequential
        expected_ranks = set(range(num_workers))
        actual_ranks = {s['rank'] for s in shards}
        if expected_ranks != actual_ranks:
            raise ValueError(
                f"Rank mismatch: expected {expected_ranks}, got {actual_ranks}"
            )

        # Check global_step consistency
        global_step = metadata['global_step']
        for shard in shards:
            if shard['global_step'] != global_step:
                raise ValueError(
                    f"Global step mismatch: expected {global_step}, "
                    f"got {shard['global_step']} for rank {shard['rank']}"
                )

        # Check world_size consistency
        for shard in shards:
            shard_world_size = shard['partition']['world_size']
            if shard_world_size != num_workers:
                raise ValueError(
                    f"World size mismatch: expected {num_workers}, "
                    f"got {shard_world_size} for rank {shard['rank']}"
                )

        # Check partition coverage (no gaps or overlaps)
        total_params = sum(s['partition']['num_params'] for s in shards)
        expected_start = 0
        for shard in shards:
            partition = shard['partition']
            if partition['shard_start'] != expected_start:
                raise ValueError(
                    f"Partition gap detected: expected start {expected_start}, "
                    f"got {partition['shard_start']} for rank {shard['rank']}"
                )
            expected_start = partition['shard_end']

        logger.debug(f"Validation passed: {len(shards)} shards, {total_params} total params")

    def _reconstruct_model(
        self,
        shards: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct complete model state_dict from shards.

        Args:
            shards: Loaded shard data (sorted by rank)
            metadata: Checkpoint metadata

        Returns:
            Complete model state_dict (flat parameters by name)
        """
        # Concatenate all parameter shards into flat tensor
        # Shards are already sorted by rank, so we just concatenate
        flat_params_pieces = [shard['parameters'] for shard in shards]
        full_flat_params = torch.cat(flat_params_pieces)

        logger.debug(f"Full flat parameters: {full_flat_params.shape}")

        # Now reconstruct individual parameters from the flat space
        # We need to figure out which parameters exist and their sizes
        # Collect all unique parameter names and their total sizes
        param_info = {}  # param_name -> total_size

        for shard in shards:
            partition = shard['partition']
            param_slices = partition['param_slices']

            for param_name, (slice_start, slice_end, total_size) in param_slices.items():
                if param_name not in param_info:
                    param_info[param_name] = total_size

        # Reconstruct each parameter by extracting from full_flat_params
        # The parameters are stored in sorted order
        state_dict = {}
        global_offset = 0

        for param_name in sorted(param_info.keys()):
            param_size = param_info[param_name]
            state_dict[param_name] = full_flat_params[global_offset:global_offset + param_size]
            global_offset += param_size

        logger.debug(f"Reconstructed {len(state_dict)} parameters")
        return state_dict

    def _save_assembled_checkpoint(
        self,
        output_path: Path,
        state_dict: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
        shards: List[Dict[str, Any]]
    ):
        """
        Save assembled checkpoint with metadata.

        Args:
            output_path: Output file path
            state_dict: Complete model state_dict
            metadata: Checkpoint metadata
            shards: Original shard data (for optimizer state)
        """
        # Build checkpoint data
        checkpoint = {
            'model_state_dict': state_dict,
            'global_step': metadata['global_step'],
            'timestamp': datetime.now().isoformat(),
            'num_workers': metadata['num_workers'],
            'model_config': metadata.get('model_config'),
        }

        # Optionally include optimizer states (if present)
        optimizer_states = []
        for shard in shards:
            if shard.get('optimizer_state'):
                optimizer_states.append({
                    'rank': shard['rank'],
                    'state': shard['optimizer_state']
                })

        if optimizer_states:
            checkpoint['optimizer_states'] = optimizer_states

        # Save checkpoint
        torch.save(checkpoint, output_path)

        # Calculate file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved assembled checkpoint: {file_size_mb:.2f} MB")

    def _update_metadata_status(
        self,
        metadata_path: Path,
        status: str,
        assembled_path: Optional[str] = None
    ):
        """
        Update metadata with assembly status.

        Args:
            metadata_path: Path to metadata.json
            status: New status ('assembled', 'failed', etc.)
            assembled_path: Optional path to assembled file
        """
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        metadata['status'] = status
        metadata['assembled_at'] = datetime.now().isoformat()

        if assembled_path:
            metadata['assembled_file'] = assembled_path

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Updated metadata status: {status}")

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint info dictionaries
        """
        checkpoints = []

        for step_dir in sorted(self.checkpoint_base_dir.glob("step_*")):
            metadata_path = step_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            checkpoints.append({
                'global_step': metadata['global_step'],
                'timestamp': metadata['timestamp'],
                'status': metadata.get('status', 'unknown'),
                'num_workers': metadata['num_workers'],
                'path': str(step_dir)
            })

        return checkpoints

    def get_checkpoint_status(self, global_step: int) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific checkpoint.

        Args:
            global_step: Training step

        Returns:
            Checkpoint status dictionary, or None if not found
        """
        step_dir = self.checkpoint_base_dir / f"step_{global_step}"
        metadata_path = step_dir / "metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Check for assembled file
        assembled_exists = False
        if metadata.get('assembled_file'):
            assembled_path = Path(metadata['assembled_file'])
            assembled_exists = assembled_path.exists()

        return {
            'global_step': metadata['global_step'],
            'timestamp': metadata['timestamp'],
            'status': metadata.get('status', 'shards_saved'),
            'num_workers': metadata['num_workers'],
            'assembled_exists': assembled_exists,
            'assembled_file': metadata.get('assembled_file'),
            'path': str(step_dir)
        }


def create_checkpoint_metadata(
    checkpoint_dir: str,
    global_step: int,
    workers: List[Dict[str, Any]],
    model_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create metadata.json for a checkpoint.

    This is typically called by the coordinator after all workers have saved shards.

    Args:
        checkpoint_dir: Base checkpoint directory
        global_step: Training step
        workers: List of worker info dicts with 'rank', 'worker_id', 'shard_file'
        model_config: Optional model configuration

    Returns:
        Path to created metadata file
    """
    step_dir = Path(checkpoint_dir) / f"step_{global_step}"
    step_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        'global_step': global_step,
        'timestamp': datetime.now().isoformat(),
        'num_workers': len(workers),
        'workers': workers,
        'partition_scheme': 'zero3',
        'model_config': model_config,
        'status': 'shards_saved'
    }

    metadata_path = step_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Created checkpoint metadata: {metadata_path}")
    return str(metadata_path)


if __name__ == "__main__":
    # Example usage
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python assembler.py <global_step>")
        print("Example: python assembler.py 1000")
        sys.exit(1)

    global_step = int(sys.argv[1])

    assembler = CheckpointAssembler()

    print(f"\nAssembling checkpoint for step {global_step}...")
    try:
        output_path = assembler.assemble_checkpoint(global_step)
        print(f"✓ Success! Assembled checkpoint: {output_path}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        sys.exit(1)
