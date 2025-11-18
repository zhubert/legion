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
from communication.async_collectives import AsyncParameterFetcher, AsyncGradientPusher


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
        self.worker_coordinator: Optional[WorkerCoordinator] = None
        self.async_param_fetcher: Optional[AsyncParameterFetcher] = None
        self.async_grad_pusher: Optional[AsyncGradientPusher] = None

        # Training state
        self._setup_complete = False
        self._current_step = 0

        # Async parameter server config
        self.staleness_bound = 5  # Maximum allowed version divergence

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
            # Use async parameter server for distributed training
            logger.info("Using async parameter server architecture")

            # Initialize async parameter fetcher and gradient pusher
            self.async_param_fetcher = AsyncParameterFetcher(
                self.grpc_client,
                timeout=10.0,
                enable_cache=True
            )
            self.async_grad_pusher = AsyncGradientPusher(
                self.grpc_client,
                timeout=10.0,
                max_retries=3
            )
            logger.info("Async parameter server components initialized")

            # Configure gRPC server for version-tracked gradient accumulation
            if self.grpc_server:
                self.grpc_server.servicer.set_world_size(self.world_size)
                self.grpc_server.servicer.set_aggregation_threshold(0.75)  # 75% threshold
                logger.info(f"gRPC server configured: world_size={self.world_size}, threshold=75%")

            # For gRPC mode, we still need a collective coordinator for simulation compatibility
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
            # Real distributed training with async parameter server
            results = await self._train_async_parameter_server(dataset, num_steps, checkpoint_interval)
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

    async def _train_async_parameter_server(
        self,
        dataset: List[Tuple[torch.Tensor, torch.Tensor]],
        num_steps: int,
        checkpoint_interval: int
    ) -> Dict[str, Any]:
        """
        Training loop using async parameter server architecture.

        Key features:
        - Bounded staleness: Workers limited to K steps ahead of global median
        - Async parameter fetching: Parallel requests with cache fallback
        - Async gradient pushing: Non-blocking with retry
        - Work stealing: Fast workers help slow workers when ahead
        """
        logger.info(f"Starting async parameter server training: rank {self.rank}/{self.world_size}")

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

        # Create optimizer for owned parameters only (ZeRO-3)
        owned_param_objects = [
            dict(self.model.named_parameters())[name]
            for name in self.partitioner.partitions[self.rank].param_names
            if name in dict(self.model.named_parameters())
        ]
        optimizer = torch.optim.Adam(owned_param_objects, lr=self.learning_rate)
        logger.info(f"Optimizer tracking {len(owned_param_objects)} owned parameters")

        # Build parameter ownership map (param_name -> owner_address)
        param_owners = {}
        for rank, partition in enumerate(self.partitioner.partitions):
            owner_address = self.worker_addresses[rank]
            for param_name in partition.param_names:
                param_owners[param_name] = owner_address

        for step, (inputs, targets) in enumerate(dataset[:num_steps]):
            step_start = time.time()

            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # === PHASE 1: Check bounded staleness and get backup assignment ===
            if self.coordinator_client:
                # Report current version to coordinator
                version_response = await self.coordinator_client.http_client.post(
                    f"{self.coordinator_client.base_url}/training/version/update",
                    json={
                        "worker_id": self.worker_id,
                        "version": step,
                        "is_healthy": True
                    }
                )
                version_data = version_response.json()
                global_version = version_data.get("global_version", 0)
                is_ahead = version_data.get("is_ahead", False)
                backup_assignment = version_data.get("backup_assignment")

                if is_ahead and backup_assignment:
                    logger.info(
                        f"Step {step}: Worker ahead (v{step} > v{global_version}+{self.staleness_bound}), "
                        f"assigned to help {backup_assignment}"
                    )
                    # Work stealing: Compute backup gradients for slow worker
                    await self._compute_backup_gradients(
                        backup_worker_id=backup_assignment,
                        backup_version=global_version,
                        inputs=inputs,
                        targets=targets,
                        param_owners=param_owners
                    )
                    # Skip own training step when doing backup computation
                    logger.info(f"Step {step}: Completed backup computation for {backup_assignment}")
                    continue
            else:
                is_ahead = False
                backup_assignment = None

            # === PHASE 2: Async parameter fetching ===
            fetch_start = time.time()

            # Build parameter fetch requests (address, param_name, shard_start, shard_end)
            parameter_requests = []
            for param_name in self.model.state_dict().keys():
                if param_name in param_owners:
                    owner_address = param_owners[param_name]
                    # For now, fetch full parameter (shard_start=0, shard_end=-1)
                    parameter_requests.append((owner_address, param_name, 0, -1))

            # Fetch all parameters in parallel
            fetched_params = await self.async_param_fetcher.fetch_parameters_async(
                parameter_requests,
                version=step,
                staleness_tolerance=self.staleness_bound
            )

            fetch_time = time.time() - fetch_start

            # === PHASE 3: Load parameters into model ===
            load_start = time.time()
            with torch.no_grad():
                for param_key, param_tensor in fetched_params.items():
                    if param_tensor is not None:
                        # Extract param name from key (format: "param_name[start:end]")
                        param_name = param_key.split('[')[0]
                        if param_name in dict(self.model.named_parameters()):
                            self.model.state_dict()[param_name].copy_(param_tensor)
            load_time = time.time() - load_start

            # === PHASE 4: Forward pass ===
            forward_start = time.time()
            logits, loss = self.model(inputs, targets)
            losses.append(loss.item())
            forward_time = time.time() - forward_start

            # === PHASE 5: Backward pass ===
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            backward_time = time.time() - backward_start

            # === PHASE 6: Async gradient pushing ===
            push_start = time.time()

            # Build gradient push requests (address, param_name, gradient, shard_start, shard_end)
            gradient_requests = []
            for param_name, param in self.model.named_parameters():
                if param.grad is not None and param_name in param_owners:
                    owner_address = param_owners[param_name]
                    gradient_requests.append((owner_address, param_name, param.grad, 0, -1))

            # Push all gradients in parallel (non-blocking)
            push_results = await self.async_grad_pusher.push_gradients_async(
                gradient_requests,
                version=step
            )

            push_time = time.time() - push_start

            # === PHASE 7: Wait for gradient aggregation and fetch reduced gradients ===
            # For owned parameters, wait for aggregation threshold then retrieve
            aggregation_start = time.time()

            for param_name in self.partitioner.partitions[self.rank].param_names:
                if param_name in dict(self.model.named_parameters()):
                    # Wait for aggregation (threshold-based, not 100%)
                    aggregated_grad = await self.grpc_server.servicer.get_accumulated_gradients(
                        version=step,
                        param_name=param_name,
                        wait_for_threshold=True,
                        timeout=30.0
                    )

                    if aggregated_grad is not None:
                        # Apply aggregated gradient to parameter
                        param = dict(self.model.named_parameters())[param_name]
                        param.grad = aggregated_grad

            aggregation_time = time.time() - aggregation_start

            # === PHASE 8: Optimizer step (only owned parameters) ===
            optimizer_start = time.time()
            optimizer.step()
            optimizer_time = time.time() - optimizer_start

            # === PHASE 9: Update parameter server with new values ===
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
                comm_time = fetch_time + push_time + aggregation_time
                compute_time = forward_time + backward_time + optimizer_time + load_time
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
                        f"fetch={fetch_time*1000:.1f}ms, "
                        f"load={load_time*1000:.1f}ms, "
                        f"forward={forward_time*1000:.1f}ms, "
                        f"backward={backward_time*1000:.1f}ms, "
                        f"push={push_time*1000:.1f}ms, "
                        f"aggregation={aggregation_time*1000:.1f}ms, "
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
            f"Async PS training complete! "
            f"Rank {self.rank}: {num_steps} steps in {total_time:.2f}s "
            f"({results['steps_per_sec']:.2f} steps/s)"
        )

        # Distributed barrier: wait for all workers to finish training
        if self.coordinator_client:
            logger.info(f"Rank {self.rank} finished training, waiting at barrier...")
            barrier_success = await self.coordinator_client.wait_at_barrier(
                step="training_complete",
                poll_interval=5.0,
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

    async def _compute_backup_gradients(
        self,
        backup_worker_id: str,
        backup_version: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        param_owners: Dict[str, str]
    ):
        """
        Compute backup gradients for a slow worker (work stealing).

        When this worker is ahead of the staleness bound, it helps a slow worker
        by fetching their older parameters, running forward/backward, and sending
        gradients to help them catch up.

        Args:
            backup_worker_id: ID of slow worker to help
            backup_version: Version/step of slow worker to compute for
            inputs: Input batch
            targets: Target batch
            param_owners: Map of param_name -> owner_address
        """
        logger.info(f"Computing backup gradients for {backup_worker_id} at version {backup_version}")

        try:
            # 1. Fetch parameters at the slow worker's version
            # Note: We fetch from current parameter owners, but ideally should fetch
            # parameters as they were at backup_version (future enhancement)
            parameter_requests = []
            for param_name in self.model.state_dict().keys():
                if param_name in param_owners:
                    owner_address = param_owners[param_name]
                    parameter_requests.append((owner_address, param_name, 0, -1))

            fetched_params = await self.async_param_fetcher.fetch_parameters_async(
                parameter_requests,
                version=backup_version,
                staleness_tolerance=self.staleness_bound
            )

            # 2. Load parameters into model
            with torch.no_grad():
                for param_key, param_tensor in fetched_params.items():
                    if param_tensor is not None:
                        param_name = param_key.split('[')[0]
                        if param_name in dict(self.model.named_parameters()):
                            self.model.state_dict()[param_name].copy_(param_tensor)

            # 3. Forward pass with slow worker's batch (we use our own batch as proxy)
            # TODO: Ideally fetch the slow worker's actual batch for this version
            logits, loss = self.model(inputs, targets)
            logger.debug(f"Backup computation loss: {loss.item():.4f}")

            # 4. Backward pass to compute backup gradients
            # Clear gradients first
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.zero_()

            loss.backward()

            # 5. Send backup gradients to parameter owners
            # Tag these as backup gradients for the slow worker's version
            gradient_requests = []
            for param_name, param in self.model.named_parameters():
                if param.grad is not None and param_name in param_owners:
                    owner_address = param_owners[param_name]
                    gradient_requests.append((owner_address, param_name, param.grad, 0, -1))

            # Push backup gradients with the slow worker's version
            push_results = await self.async_grad_pusher.push_gradients_async(
                gradient_requests,
                version=backup_version
            )

            # Count successful pushes
            successful_pushes = sum(1 for success in push_results.values() if success)
            logger.info(
                f"Backup gradients sent: {successful_pushes}/{len(push_results)} successful "
                f"for {backup_worker_id} v{backup_version}"
            )

        except Exception as e:
            logger.error(f"Failed to compute backup gradients for {backup_worker_id}: {e}")

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
