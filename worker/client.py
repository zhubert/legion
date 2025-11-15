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

from sim.model import create_model


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
        dataset: List[Tuple[torch.Tensor, torch.Tensor]],
        num_steps: Optional[int] = None
    ):
        """
        Run distributed training.

        Args:
            dataset: Training dataset (list of input/target batches)
            num_steps: Number of training steps
        """
        if not self._running:
            raise RuntimeError("Worker not started. Call start() first.")

        # For now, assume we're the only worker (rank 0, world_size 1)
        # In Phase 1.3 we'll add multi-worker support with gRPC
        rank = 0
        world_size = 1

        logger.info("Initializing training components...")

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
            latency_ms=0.0,  # No latency simulation for single worker
            shard_manager=self.shard_manager,
            telemetry_reporter=self.telemetry_reporter
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

        # Create dummy dataset for testing
        from sim.train import create_dummy_dataset
        dataset = create_dummy_dataset(
            vocab_size=1000,
            seq_len=config.seq_len,
            num_batches=config.num_steps,
            batch_size=config.batch_size
        )

        # Run training
        await worker.run_training(dataset, num_steps=config.num_steps)

        # Wait for shutdown signal
        # await worker.wait_for_shutdown()

    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
