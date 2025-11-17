"""
Main worker client for Legion distributed training.

Orchestrates all worker components and manages the complete worker lifecycle.
"""

import asyncio
import signal
import logging
from typing import Optional, List, Tuple
import torch

from worker.config import WorkerConfig
from worker.coordinator_client import CoordinatorClient
from worker.heartbeat import HeartbeatManager
from worker.shard_manager import ShardManager
from worker.telemetry import TelemetryReporter
from worker.trainer import DistributedTrainer

from communication.grpc_server import WorkerGRPCServer
from communication.grpc_client import WorkerGRPCClient

from core.model import create_model


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkerClient:
    """
    Main worker client orchestrating all components.

    Manages worker lifecycle including registration, heartbeat,
    training, and graceful shutdown.
    """

    def __init__(self, config: WorkerConfig):
        """
        Initialize worker client.

        Args:
            config: Worker configuration
        """
        self.config = config

        # Set logging level
        logging.getLogger().setLevel(config.log_level)

        # Components (initialized in start())
        self.coordinator_client: Optional[CoordinatorClient] = None
        self.heartbeat_manager: Optional[HeartbeatManager] = None
        self.shard_manager: Optional[ShardManager] = None
        self.telemetry_reporter: Optional[TelemetryReporter] = None
        self.trainer: Optional[DistributedTrainer] = None

        # gRPC components (Phase 1 integration)
        self.grpc_server: Optional[WorkerGRPCServer] = None
        self.grpc_client: Optional[WorkerGRPCClient] = None

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()

        logger.info(f"Worker client initialized: {config.worker_id}")

    async def start(self):
        """
        Start worker client.

        Initializes all components, registers with coordinator,
        and starts background tasks.
        """
        if self._running:
            logger.warning("Worker already running")
            return

        logger.info("=" * 60)
        logger.info("Starting Legion Worker")
        logger.info("=" * 60)
        logger.info(f"Worker ID: {self.config.worker_id}")
        logger.info(f"Coordinator: {self.config.coordinator_url}")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Model: {self.config.model_size}")
        logger.info("=" * 60)

        # Initialize coordinator client
        logger.info("Initializing coordinator client...")
        self.coordinator_client = CoordinatorClient(
            coordinator_url=self.config.coordinator_url,
            worker_id=self.config.worker_id,
            ip_address=self.config.ip_address,
            port=self.config.port,
            timeout=self.config.coordinator_timeout,
            retry_attempts=self.config.coordinator_retry_attempts,
            retry_delay=self.config.coordinator_retry_delay,
            gpu_info=self.config.get_gpu_info(),
            cpu_cores=self.config.cpu_cores,
            ram_gb=self.config.ram_gb,
            bandwidth_mbps=self.config.bandwidth_mbps
        )

        # Register with coordinator
        logger.info("Registering with coordinator...")
        success = await self.coordinator_client.register()
        if not success:
            raise RuntimeError("Failed to register with coordinator")
        logger.info("✓ Registration successful")

        # Initialize heartbeat manager
        logger.info("Starting heartbeat manager...")
        self.heartbeat_manager = HeartbeatManager(
            coordinator_client=self.coordinator_client,
            interval_seconds=self.config.heartbeat_interval,
            max_failures=self.config.heartbeat_max_failures
        )
        await self.heartbeat_manager.start()
        logger.info("✓ Heartbeat manager started")

        # Initialize telemetry reporter
        if self.config.telemetry_enabled:
            logger.info("Initializing telemetry reporter...")
            self.telemetry_reporter = TelemetryReporter(
                coordinator_client=self.coordinator_client,
                report_interval_steps=self.config.telemetry_interval_steps,
                buffer_size=self.config.telemetry_buffer_size,
                enabled=True
            )
            logger.info("✓ Telemetry reporter initialized")

        # Initialize gRPC client for worker-to-worker communication
        logger.info("Initializing gRPC client...")
        self.grpc_client = WorkerGRPCClient(
            worker_id=self.config.worker_id,
            timeout=30.0
        )
        logger.info("✓ gRPC client initialized")

        # Initialize gRPC server (will be started when training begins)
        # Note: We'll create the server with an empty parameter store initially
        # and update it when the model is partitioned
        logger.info(f"Initializing gRPC server on port {self.config.port}...")
        self.grpc_server = WorkerGRPCServer(
            worker_id=self.config.worker_id,
            parameter_store={},  # Empty initially, populated during training setup
            host="0.0.0.0",
            port=self.config.port
        )
        await self.grpc_server.start()
        logger.info(f"✓ gRPC server started on {self.config.ip_address}:{self.config.port}")

        self._running = True

        logger.info("")
        logger.info("Worker started successfully!")
        logger.info("=" * 60)

    async def stop(self):
        """
        Stop worker client.

        Performs graceful shutdown: stops training, saves checkpoint,
        stops background tasks, and deregisters from coordinator.
        """
        if not self._running:
            return

        logger.info("=" * 60)
        logger.info("Shutting down worker...")
        logger.info("=" * 60)

        # Stop heartbeat manager
        if self.heartbeat_manager:
            logger.info("Stopping heartbeat manager...")
            await self.heartbeat_manager.stop()
            logger.info("✓ Heartbeat manager stopped")

        # Report final metrics
        if self.telemetry_reporter and self.telemetry_reporter.is_enabled():
            logger.info("Reporting final metrics...")
            await self.telemetry_reporter.report(force=True)
            logger.info("✓ Final metrics reported")

        # Save final checkpoint
        if self.shard_manager and self.shard_manager.is_loaded():
            logger.info("Saving final checkpoint...")
            try:
                step = self.trainer.get_current_step() if self.trainer else 0
                checkpoint_path = self.shard_manager.save_checkpoint(
                    global_step=step
                )
                logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

        # Stop gRPC server and client
        if self.grpc_server:
            logger.info("Stopping gRPC server...")
            await self.grpc_server.stop()
            logger.info("✓ gRPC server stopped")

        if self.grpc_client:
            logger.info("Closing gRPC client...")
            await self.grpc_client.close()
            logger.info("✓ gRPC client closed")

        # Deregister from coordinator
        if self.coordinator_client:
            logger.info("Deregistering from coordinator...")
            await self.coordinator_client.deregister()
            await self.coordinator_client.close()
            logger.info("✓ Deregistered from coordinator")

        self._running = False
        self._shutdown_event.set()

        logger.info("=" * 60)
        logger.info("Worker shutdown complete")
        logger.info("=" * 60)

    async def run_training(
        self,
        dataset: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        num_steps: Optional[int] = None,
        use_distributed: bool = False
    ):
        """
        Run distributed training.

        Args:
            dataset: Training dataset (optional, will be created if not provided)
            num_steps: Number of training steps
            use_distributed: Whether to use distributed multi-worker training
        """
        if not self._running:
            raise RuntimeError("Worker not started. Call start() first.")

        # Get cluster information from coordinator
        rank = 0
        world_size = 1
        worker_addresses = []

        if use_distributed:
            logger.info("Fetching cluster information from coordinator...")
            try:
                # Get list of online workers
                workers_response = await self.coordinator_client.get_workers(status="online")
                if workers_response and 'workers' in workers_response:
                    workers = workers_response['workers']
                    world_size = len(workers)

                    # Sort workers by worker_id to ensure consistent rank assignment
                    workers = sorted(workers, key=lambda w: w.get('worker_id', ''))

                    # Find our rank
                    for i, worker in enumerate(workers):
                        if worker.get('worker_id') == self.config.worker_id:
                            rank = i
                        # Build worker addresses list
                        ip = worker.get('ip_address', 'localhost')
                        port = worker.get('port', 50051)
                        worker_addresses.append(f"{ip}:{port}")

                    logger.info(
                        f"Cluster configured: rank {rank}/{world_size}, "
                        f"{len(worker_addresses)} workers"
                    )
                else:
                    logger.warning("No cluster information available, falling back to single-worker mode")
                    use_distributed = False
            except Exception as e:
                logger.error(f"Failed to get cluster information: {e}")
                logger.warning("Falling back to single-worker mode")
                use_distributed = False

        logger.info("Initializing training components...")

        # If distributed, verify connectivity to peers
        if use_distributed and len(worker_addresses) > 1:
            logger.info("Verifying connectivity to peer workers...")
            for i, addr in enumerate(worker_addresses):
                if i == rank:
                    continue  # Skip self

                try:
                    latency = await self.grpc_client.ping(addr)
                    if latency is not None:
                        logger.info(f"✓ Connected to worker at {addr} (latency: {latency:.1f}ms)")
                    else:
                        logger.warning(f"✗ Failed to ping worker at {addr}")
                except Exception as e:
                    logger.warning(f"✗ Cannot reach worker at {addr}: {e}")

        # Create dataset if not provided
        if dataset is None:
            dataset_type = self.config.dataset_type
            dataset_name = self.config.dataset_name

            # Determine dataset type from config
            if dataset_type == "huggingface" and dataset_name:
                # HuggingFace dataset (fineweb, pile, shakespeare, etc.)
                logger.info(f"Loading HuggingFace dataset: {dataset_name}")
                from core.dataset import create_huggingface_dataset
                dataset = create_huggingface_dataset(
                    dataset_name=dataset_name,
                    rank=rank,
                    world_size=world_size,
                    num_batches=num_steps or self.config.num_steps,
                    batch_size=self.config.batch_size,
                    tokenizer_name=self.config.tokenizer_name,
                    seq_len=self.config.seq_len,
                    seed=42
                )
                logger.info(
                    f"Loaded HuggingFace dataset '{dataset_name}' for rank {rank}/{world_size} "
                    f"(effective global batch size: {self.config.batch_size * world_size})"
                )
            elif dataset_type == "distributed_dummy" or (use_distributed and world_size > 1):
                # Distributed dummy dataset (for testing multi-worker)
                from core.dataset import create_distributed_dataset
                dataset = create_distributed_dataset(
                    vocab_size=1000,
                    seq_len=self.config.seq_len,
                    num_batches=num_steps or self.config.num_steps,
                    batch_size=self.config.batch_size,
                    rank=rank,
                    world_size=world_size,
                    seed=42
                )
                logger.info(
                    f"Created distributed dummy dataset shard for rank {rank}/{world_size} "
                    f"(effective global batch size: {self.config.batch_size * world_size})"
                )
            else:
                # Single-worker dummy dataset (default)
                from core.dataset import create_dummy_dataset
                dataset = create_dummy_dataset(
                    vocab_size=1000,
                    seq_len=self.config.seq_len,
                    num_batches=num_steps or self.config.num_steps,
                    batch_size=self.config.batch_size
                )
                logger.info(f"Created single-worker dummy dataset (batch size: {self.config.batch_size})")

        # Create model for shard manager
        model = create_model(self.config.model_size)
        total_params = model.count_parameters()

        # Initialize shard manager (we own all parameters for now)
        self.shard_manager = ShardManager(
            worker_id=self.config.worker_id,
            model=model,
            shard_start=0,
            shard_end=total_params,
            device=self.config.device,
            checkpoint_dir=self.config.checkpoint_dir
        )
        self.shard_manager.load_shard()
        logger.info("✓ Shard manager initialized")

        # Initialize trainer
        self.trainer = DistributedTrainer(
            worker_id=self.config.worker_id,
            rank=rank,
            world_size=world_size,
            model_size=self.config.model_size,
            device=self.config.device,
            learning_rate=self.config.learning_rate,
            compression=self.config.compression,
            latency_ms=0.0,  # No latency simulation
            shard_manager=self.shard_manager,
            telemetry_reporter=self.telemetry_reporter,
            coordinator_client=self.coordinator_client if use_distributed else None,
            # gRPC components for distributed training
            grpc_client=self.grpc_client if use_distributed else None,
            grpc_server=self.grpc_server if use_distributed else None,
            worker_addresses=worker_addresses if use_distributed else None
        )
        self.trainer.setup()
        logger.info("✓ Trainer initialized")

        # Run training
        logger.info("")
        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info("=" * 60)

        results = await self.trainer.train(
            dataset=dataset,
            num_steps=num_steps or self.config.num_steps,
            checkpoint_interval=self.config.checkpoint_interval_steps
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("Training Results")
        logger.info("=" * 60)
        logger.info(f"Total steps: {results['num_steps']}")
        logger.info(f"Total time: {results['total_time']:.2f}s")
        logger.info(f"Speed: {results['steps_per_sec']:.2f} steps/s")
        logger.info(f"Initial loss: {results['initial_loss']:.4f}")
        logger.info(f"Final loss: {results['final_loss']:.4f}")
        logger.info(f"Average loss: {results['avg_loss']:.4f}")
        logger.info("=" * 60)

        return results

    async def wait_for_shutdown(self):
        """
        Wait for shutdown signal.

        Blocks until shutdown is triggered via signal or stop().
        """
        await self._shutdown_event.wait()

    def get_status(self) -> dict:
        """
        Get worker status.

        Returns:
            Status dictionary with component information
        """
        status = {
            'worker_id': self.config.worker_id,
            'running': self._running,
            'coordinator_url': self.config.coordinator_url,
            'device': self.config.device,
            'model_size': self.config.model_size
        }

        if self.coordinator_client:
            status['registered'] = self.coordinator_client.is_registered()

        if self.heartbeat_manager:
            status['heartbeat'] = self.heartbeat_manager.get_status()

        if self.telemetry_reporter:
            status['telemetry'] = self.telemetry_reporter.get_status()

        if self.shard_manager:
            status['shard'] = self.shard_manager.get_shard_info()

        if self.trainer:
            status['training'] = {
                'current_step': self.trainer.get_current_step(),
                'setup_complete': self.trainer.is_setup()
            }

        return status

    def is_running(self) -> bool:
        """
        Check if worker is running.

        Returns:
            True if running, False otherwise
        """
        return self._running


# Signal handling for graceful shutdown

_worker_instance: Optional[WorkerClient] = None


def setup_signal_handlers(worker: WorkerClient):
    """
    Set up signal handlers for graceful shutdown.

    Args:
        worker: Worker client instance
    """
    global _worker_instance
    _worker_instance = worker

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        if _worker_instance:
            asyncio.create_task(_worker_instance.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# Main entry point

async def main(config: Optional[WorkerConfig] = None):
    """
    Main entry point for worker client.

    Args:
        config: Optional worker configuration (creates default if None)
    """
    if config is None:
        config = WorkerConfig()

    worker = WorkerClient(config)
    setup_signal_handlers(worker)

    try:
        # Start worker
        await worker.start()

        # Note: Dataset creation moved inside run_training() where rank/world_size are known
        # Run training
        await worker.run_training(num_steps=config.num_steps)

        # Wait for shutdown signal
        # await worker.wait_for_shutdown()

    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
