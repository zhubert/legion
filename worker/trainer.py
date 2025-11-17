"""
Distributed training integration for worker nodes.

Wraps Phase 0 training loop with worker client integration for distributed training.
"""

import asyncio
import logging
import time
from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn

from core.model import create_model
from core.partitioner import Partitioner
from sim.worker import WorkerCoordinator
from sim.collectives import CollectiveCoordinator

from worker.shard_manager import ShardManager
from worker.telemetry import TelemetryReporter
from worker.coordinator_client import CoordinatorClient

from communication.grpc_server import WorkerGRPCServer
from communication.grpc_client import WorkerGRPCClient
from communication.grpc_collectives import GRPCCollectiveOps


logger = logging.getLogger(__name__)


class DistributedTrainer:
    """
    Distributed trainer integrating worker client with Phase 0 training loop.

    Combines existing simulation components with real coordinator integration.
    """

    def __init__(
        self,
        worker_id: str,
        rank: int,
        world_size: int,
        model_size: str = "tiny",
        device: str = "cpu",
        learning_rate: float = 0.001,
        compression: str = "none",
        latency_ms: float = 0.0,
        shard_manager: Optional[ShardManager] = None,
        telemetry_reporter: Optional[TelemetryReporter] = None,
        coordinator_client: Optional[CoordinatorClient] = None,
        grpc_client: Optional[WorkerGRPCClient] = None,
        grpc_server: Optional[WorkerGRPCServer] = None,
        worker_addresses: Optional[List[str]] = None
    ):
        """
        Initialize distributed trainer.

        Args:
            worker_id: Unique worker identifier
            rank: Worker rank in cluster
            world_size: Total number of workers
            model_size: Model size ('tiny', 'small', 'medium')
            device: Device to use ('cpu' or 'cuda')
            learning_rate: Learning rate
            compression: Compression method ('none', 'int8', 'topk')
            latency_ms: Simulated network latency
            shard_manager: Optional shard manager
            telemetry_reporter: Optional telemetry reporter
            grpc_client: Optional gRPC client for distributed training
            grpc_server: Optional gRPC server for serving parameters
            worker_addresses: Optional list of worker addresses for distributed training
        """
        self.worker_id = worker_id
        self.rank = rank
        self.world_size = world_size
        self.model_size = model_size
        self.device = device
        self.learning_rate = learning_rate
        self.compression = compression
        self.latency_ms = latency_ms

        # External components
        self.shard_manager = shard_manager
        self.telemetry_reporter = telemetry_reporter
        self.coordinator_client = coordinator_client

        # gRPC components for distributed training
        self.grpc_client = grpc_client
        self.grpc_server = grpc_server
        self.worker_addresses = worker_addresses
        self.use_grpc = grpc_client is not None and grpc_server is not None

        # Training components (will be initialized in setup)
        self.model: Optional[nn.Module] = None
        self.partitioner: Optional[Partitioner] = None
        self.collective_coordinator: Optional[CollectiveCoordinator] = None
        self.grpc_collective_ops: Optional[GRPCCollectiveOps] = None
        self.worker_coordinator: Optional[WorkerCoordinator] = None

        # Training state
        self._setup_complete = False
        self._current_step = 0

        logger.info(
            f"Distributed trainer initialized: worker_id={worker_id}, "
            f"rank={rank}/{world_size}, model={model_size}, "
            f"use_grpc={self.use_grpc}"
        )

    def setup(self):
        """
        Set up training components.

        Initializes model, partitioner, and coordinators.
        """
        if self._setup_complete:
            logger.warning("Trainer already set up")
            return

        # Create model
        logger.info(f"Creating {self.model_size} model...")
        self.model = create_model(self.model_size)
        logger.info(f"Model parameters: {self.model.count_parameters():,}")

        # Partition model across workers
        logger.info(f"Partitioning model across {self.world_size} workers...")
        self.partitioner = Partitioner(self.model, world_size=self.world_size)
        self.partitioner.print_partition_info()

        # Create collective communication coordinator
        logger.info("Setting up collective communication...")
        max_tensor_size = max(p.num_params for p in self.partitioner.partitions)

        if self.use_grpc:
            # Use gRPC-based collectives for real distributed training
            logger.info("Using gRPC-based collective operations")
            self.grpc_collective_ops = GRPCCollectiveOps(
                rank=self.rank,
                world_size=self.world_size,
                worker_id=self.worker_id,
                worker_addresses=self.worker_addresses,
                grpc_client=self.grpc_client,
                timeout=30.0
            )

            # Update gRPC server's parameter store with owned parameters
            if self.grpc_server:
                # Set expected gradient count (world_size - 1, excluding self)
                self.grpc_server.servicer.set_expected_gradient_count(self.world_size - 1)
                logger.info(f"gRPC server expects {self.world_size - 1} gradient contributions per step")

                # We'll populate parameter store during training as parameters are updated
                logger.info("gRPC server parameter store ready for updates")

            # For gRPC mode, we still need a collective coordinator for the worker coordinator
            # but we'll override the collective ops with gRPC-based ones
            self.collective_coordinator = CollectiveCoordinator(
                self.world_size,
                max_tensor_size
            )
        else:
            # Use simulation-based collectives (shared memory)
            logger.info("Using simulation-based collective operations")
            self.collective_coordinator = CollectiveCoordinator(
                self.world_size,
                max_tensor_size
            )

        # Create worker coordinator
        logger.info("Initializing worker coordinator...")

        def collective_ops_factory(rank):
            return self.collective_coordinator.get_collective_ops(
                rank,
                latency_ms=self.latency_ms
            )

        self.worker_coordinator = WorkerCoordinator(
            world_size=self.world_size,
            model=self.model,
            partitions=self.partitioner.partitions,
            collective_ops_factory=collective_ops_factory,
            learning_rate=self.learning_rate,
            compression=self.compression,
            device=self.device
        )

        self._setup_complete = True
        logger.info("Trainer setup complete")

    async def train(
        self,
        dataset: List[Tuple[torch.Tensor, torch.Tensor]],
        num_steps: Optional[int] = None,
        checkpoint_interval: int = 100
    ) -> Dict[str, Any]:
        """
        Run training loop.

        Args:
            dataset: List of (input, target) batches
            num_steps: Number of training steps (defaults to dataset length)
            checkpoint_interval: Save checkpoint every N steps

        Returns:
            Training results dictionary
        """
        if not self._setup_complete:
            raise RuntimeError("Trainer not set up. Call setup() first.")

        num_steps = num_steps or len(dataset)
        logger.info(f"Starting training for {num_steps} steps...")

        # Training metrics
        losses = []
        start_time = time.time()

        # Choose training path based on whether we're using gRPC
        if self.use_grpc:
            # Real distributed training with gRPC
            results = await self._train_distributed_grpc(dataset, num_steps, checkpoint_interval)
        else:
            # Simulation-based training (single-process multi-worker)
            results = await self._train_simulation(dataset, num_steps, checkpoint_interval)

        return results

    async def _train_simulation(
        self,
        dataset: List[Tuple[torch.Tensor, torch.Tensor]],
        num_steps: int,
        checkpoint_interval: int
    ) -> Dict[str, Any]:
        """
        Training loop using simulation-based collectives (shared memory).

        Used for single-process testing.
        """
        losses = []
        start_time = time.time()

        for step, batch in enumerate(dataset[:num_steps]):
            step_start = time.time()

            # Prepare batches for simulation (WorkerCoordinator manages all workers)
            batches = [batch for _ in range(self.world_size)]

            # Execute training step
            metrics = self.worker_coordinator.train_step(batches)

            # Collect loss from this worker
            worker_metrics = metrics[self.rank]
            loss = worker_metrics['loss']
            losses.append(loss)

            # Calculate throughput
            step_time = time.time() - step_start
            throughput = len(batches) / step_time if step_time > 0 else 0.0

            # Report telemetry
            if self.telemetry_reporter:
                memory_usage = 0.0
                if self.shard_manager:
                    mem_info = self.shard_manager.get_memory_usage()
                    memory_usage = mem_info.get('total_gb', 0.0)

                await self.telemetry_reporter.report_step_async(
                    step=step,
                    loss=loss.item() if isinstance(loss, torch.Tensor) else loss,
                    throughput=throughput,
                    memory_usage_gb=memory_usage
                )

            # Print progress
            if (step + 1) % 10 == 0 or step == 0:
                avg_loss = sum(losses[-10:]) / len(losses[-10:])
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0.0

                logger.info(
                    f"Step {step+1:3d}/{num_steps} | "
                    f"Loss: {loss:.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Speed: {steps_per_sec:.2f} steps/s"
                )

            # Save checkpoint
            if self.shard_manager and (step + 1) % checkpoint_interval == 0:
                try:
                    checkpoint_path = self.shard_manager.save_checkpoint(
                        global_step=step + 1
                    )
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {e}")

            self._current_step = step + 1

        total_time = time.time() - start_time

        # Training results
        results = {
            'num_steps': num_steps,
            'total_time': total_time,
            'steps_per_sec': num_steps / total_time if total_time > 0 else 0.0,
            'final_loss': losses[-1] if losses else None,
            'initial_loss': losses[0] if losses else None,
            'avg_loss': sum(losses) / len(losses) if losses else None,
            'losses': losses
        }

        logger.info(
            f"Training complete! "
            f"{num_steps} steps in {total_time:.2f}s "
            f"({results['steps_per_sec']:.2f} steps/s)"
        )

        return results

    async def _train_distributed_grpc(
        self,
        dataset: List[Tuple[torch.Tensor, torch.Tensor]],
        num_steps: int,
        checkpoint_interval: int
    ) -> Dict[str, Any]:
        """
        Training loop using gRPC-based distributed collectives.

        Each worker runs independently and coordinates via gRPC.
        """
        logger.info(f"Starting gRPC distributed training: rank {self.rank}/{self.world_size}")

        losses = []
        start_time = time.time()

        # Get this worker's owned parameters
        owned_params = {}
        for name, param in self.model.named_parameters():
            if name in self.partitioner.partitions[self.rank].param_names:
                owned_params[name] = param.data.clone()
                # Update gRPC server's parameter store
                self.grpc_server.update_parameters(name, param.data)

        logger.info(f"Worker owns {len(owned_params)} parameter groups")

        # Create optimizer for owned parameters
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate
        )

        for step, (inputs, targets) in enumerate(dataset[:num_steps]):
            step_start = time.time()

            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # === PHASE 1: Real Distributed Training with gRPC ===

            # 1. All-gather parameters from all workers via gRPC
            all_gather_start = time.time()
            all_params = {}

            for name, param in self.model.named_parameters():
                param_owner_rank = self._get_param_owner_rank(name)

                if param_owner_rank == self.rank:
                    # We own this parameter, use our local copy
                    all_params[name] = param.data.clone()
                else:
                    # Fetch from the owner via gRPC with retry
                    fetched = None
                    for retry in range(3):
                        fetched = await self.grpc_client.get_parameters(
                            worker_address=self.worker_addresses[param_owner_rank],
                            parameter_name=name,
                            shard_start=0,
                            shard_end=-1
                        )
                        if fetched is not None:
                            break
                        # Wait before retry (peer might still be starting up)
                        if retry < 2:
                            await asyncio.sleep(0.5)

                    if fetched is not None:
                        all_params[name] = fetched
                    else:
                        # Fallback: use current parameter value
                        logger.warning(f"Failed to fetch {name} from rank {param_owner_rank} after retries, using local")
                        all_params[name] = param.data.clone()

            all_gather_time = time.time() - all_gather_start

            # 2. Load all parameters into model for forward/backward
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.data.copy_(all_params[name])

            # 3. Forward pass
            forward_start = time.time()
            logits, loss = self.model(inputs, targets)
            losses.append(loss.item())
            forward_time = time.time() - forward_start

            # 4. Backward pass to compute gradients
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            backward_time = time.time() - backward_start

            # 5. Reduce-scatter: Send gradients to parameter owners via gRPC
            reduce_scatter_start = time.time()

            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue

                param_owner_rank = self._get_param_owner_rank(name)

                if param_owner_rank != self.rank:
                    # Send our gradient to the owner with retry
                    success = False
                    for retry in range(3):
                        success = await self.grpc_client.send_gradients(
                            worker_address=self.worker_addresses[param_owner_rank],
                            gradients=param.grad,
                            step=step,
                            parameter_name=name,  # Pass parameter name for proper accumulation
                            shard_start=0,
                            shard_end=-1
                        )
                        if success:
                            break
                        if retry < 2:
                            await asyncio.sleep(0.1)

                    if not success:
                        logger.warning(f"Failed to send gradient for {name} to rank {param_owner_rank}")

            reduce_scatter_time = time.time() - reduce_scatter_start

            # 6. Retrieve accumulated gradients for parameters we own
            grad_retrieval_start = time.time()

            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue

                param_owner_rank = self._get_param_owner_rank(name)

                if param_owner_rank == self.rank:
                    # We own this parameter - get accumulated gradients from other workers
                    # Wait briefly for gradients to arrive
                    await asyncio.sleep(0.01)

                    accumulated_grad = await self.grpc_server.servicer.get_accumulated_gradients(
                        step=step,
                        shard_id=name,  # Use parameter name
                        reduce_op="sum"
                    )

                    if accumulated_grad is not None:
                        # Add accumulated gradients from other workers to our own gradient
                        param.grad.data.add_(accumulated_grad)

            grad_retrieval_time = time.time() - grad_retrieval_start

            # 7. Optimizer step (only updates parameters we own)
            optimizer_start = time.time()
            optimizer.step()
            optimizer_time = time.time() - optimizer_start

            # 8. Clear gradient accumulator for this step
            await self.grpc_server.servicer.clear_gradients(step)

            # 9. Update gRPC server's parameter store with our updated parameters
            for name, param in self.model.named_parameters():
                if self._get_param_owner_rank(name) == self.rank:
                    self.grpc_server.update_parameters(name, param.data)

            # Calculate throughput
            step_time = time.time() - step_start
            throughput = 1.0 / step_time if step_time > 0 else 0.0

            # Report telemetry
            if self.telemetry_reporter:
                memory_usage = 0.0
                if self.shard_manager:
                    mem_info = self.shard_manager.get_memory_usage()
                    memory_usage = mem_info.get('total_gb', 0.0)

                await self.telemetry_reporter.report_step_async(
                    step=step,
                    loss=loss.item() if isinstance(loss, torch.Tensor) else loss,
                    throughput=throughput,
                    memory_usage_gb=memory_usage
                )

            # Print progress
            if (step + 1) % 10 == 0 or step == 0:
                avg_loss = sum(losses[-10:]) / len(losses[-10:])
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0.0

                # Communication overhead
                comm_time = all_gather_time + reduce_scatter_time + grad_retrieval_time
                compute_time = forward_time + backward_time + optimizer_time
                total_time = comm_time + compute_time
                comm_overhead = (comm_time / total_time * 100) if total_time > 0 else 0

                logger.info(
                    f"Rank {self.rank} | "
                    f"Step {step+1:3d}/{num_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg: {avg_loss:.4f} | "
                    f"Speed: {steps_per_sec:.2f} steps/s | "
                    f"Comm: {comm_overhead:.1f}%"
                )

                # Detailed timing on first step and every 50 steps
                if step == 0 or (step + 1) % 50 == 0:
                    logger.info(
                        f"  Timing breakdown: "
                        f"all-gather={all_gather_time*1000:.1f}ms, "
                        f"forward={forward_time*1000:.1f}ms, "
                        f"backward={backward_time*1000:.1f}ms, "
                        f"reduce-scatter={reduce_scatter_time*1000:.1f}ms, "
                        f"grad-retrieval={grad_retrieval_time*1000:.1f}ms, "
                        f"optimizer={optimizer_time*1000:.1f}ms"
                    )

            # Save checkpoint
            if self.shard_manager and (step + 1) % checkpoint_interval == 0:
                try:
                    checkpoint_path = self.shard_manager.save_checkpoint(
                        global_step=step + 1
                    )
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {e}")

            self._current_step = step + 1

        total_time = time.time() - start_time

        # Training results
        results = {
            'num_steps': num_steps,
            'total_time': total_time,
            'steps_per_sec': num_steps / total_time if total_time > 0 else 0.0,
            'final_loss': losses[-1] if losses else None,
            'initial_loss': losses[0] if losses else None,
            'avg_loss': sum(losses) / len(losses) if losses else None,
            'losses': losses
        }

        logger.info(
            f"Distributed training complete! "
            f"Rank {self.rank}: {num_steps} steps in {total_time:.2f}s "
            f"({results['steps_per_sec']:.2f} steps/s)"
        )

        # Distributed barrier: wait for all workers to finish training
        if self.coordinator_client:
            logger.info(f"Rank {self.rank} finished training, waiting at barrier...")
            barrier_success = await self.coordinator_client.wait_at_barrier(
                step="training_complete",
                poll_interval=1.0,
                timeout=300.0  # 5 minutes
            )
            if barrier_success:
                logger.info(f"Rank {self.rank} barrier complete, all workers finished")
            else:
                logger.warning(f"Rank {self.rank} barrier timeout")
        else:
            # Fallback if no coordinator
            logger.info(f"Rank {self.rank} finished, waiting 10s for peers...")
            await asyncio.sleep(10)

        return results

    def get_worker_stats(self) -> Dict[str, Any]:
        """
        Get worker statistics.

        Returns:
            Statistics dictionary
        """
        if not self._setup_complete or not self.worker_coordinator:
            return {}

        stats = self.worker_coordinator.get_all_stats()
        if stats and len(stats) > self.rank:
            return stats[self.rank]

        return {}

    def get_current_step(self) -> int:
        """
        Get current training step.

        Returns:
            Current step number
        """
        return self._current_step

    def is_setup(self) -> bool:
        """
        Check if trainer is set up.

        Returns:
            True if setup complete, False otherwise
        """
        return self._setup_complete

    def _get_param_owner_rank(self, param_name: str) -> int:
        """
        Determine which worker owns a parameter.

        In ZeRO-3, each parameter is owned by exactly one worker based on
        the parameter partitioning.

        Args:
            param_name: Name of the parameter

        Returns:
            Rank of the worker that owns this parameter
        """
        if not self.partitioner:
            return 0

        for rank, partition in enumerate(self.partitioner.partitions):
            if param_name in partition.param_names:
                return rank

        # Fallback: if parameter not found in any partition, assign to rank 0
        logger.warning(f"Parameter {param_name} not found in any partition, assigning to rank 0")
        return 0
