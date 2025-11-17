"""
Test script for 2-worker distributed training.

This script demonstrates end-to-end distributed training with:
1. Coordinator server
2. Two worker clients
3. gRPC communication for parameter exchange
4. Distributed dataset sharding

Usage:
    # Terminal 1: Start coordinator
    python -m coordinator.server

    # Terminal 2: Run this test script
    python scripts/test_two_workers.py
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from worker.client import WorkerClient
from worker.config import WorkerConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def wait_for_coordinator():
    """Wait for coordinator to be ready."""
    import httpx

    logger.info("Waiting for coordinator to be ready...")
    for _ in range(30):  # Try for 30 seconds
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8000/health")
                if response.status_code == 200:
                    logger.info("✓ Coordinator is ready")
                    return True
        except:
            await asyncio.sleep(1)

    logger.error("✗ Coordinator not available")
    return False


async def wait_for_training_ready(min_workers: int = 2):
    """Wait for enough workers to be online."""
    import httpx

    logger.info(f"Waiting for {min_workers} workers to be online...")
    for _ in range(60):  # Try for 60 seconds
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:8000/training/ready?min_workers={min_workers}"
                )
                if response.status_code == 200:
                    data = response.json()
                    if data['ready']:
                        logger.info(f"✓ Training ready: {data['active_workers']} workers online")
                        return True
                    else:
                        logger.info(f"  Waiting... {data['active_workers']}/{min_workers} workers")
        except Exception as e:
            logger.warning(f"Error checking readiness: {e}")

        await asyncio.sleep(2)

    logger.error("✗ Training not ready (timeout)")
    return False


async def run_worker(worker_id: str, port: int, num_steps: int = 50):
    """
    Run a single worker client.

    Args:
        worker_id: Unique worker identifier
        port: gRPC port for this worker
        num_steps: Number of training steps
    """
    logger.info(f"=" * 60)
    logger.info(f"Starting worker: {worker_id}")
    logger.info(f"=" * 60)

    # Create worker configuration
    config = WorkerConfig(
        worker_id=worker_id,
        coordinator_url="http://localhost:8000",
        ip_address="127.0.0.1",  # Force localhost for local testing
        port=port,
        model_size="tiny",
        num_steps=num_steps,
        batch_size=4,
        seq_len=32,
        learning_rate=0.001,
        device="cpu",
        telemetry_enabled=True,
        checkpoint_enabled=False,
        heartbeat_interval=30,
        dataset_type="distributed_dummy"  # Use distributed dummy dataset
    )

    # Create and start worker
    worker = WorkerClient(config)

    try:
        # Start worker (registers with coordinator, starts gRPC server)
        await worker.start()
        logger.info(f"✓ Worker {worker_id} started successfully")

        # Wait for training to be ready
        if not await wait_for_training_ready(min_workers=2):
            logger.error(f"Worker {worker_id} timeout waiting for peers")
            return None

        # Extra wait to ensure all gRPC servers are fully up
        logger.info(f"Worker {worker_id} waiting for peer gRPC servers to be ready...")
        await asyncio.sleep(2)

        # Run distributed training
        logger.info(f"Worker {worker_id} starting distributed training...")
        results = await worker.run_training(
            num_steps=num_steps,
            use_distributed=True  # Enable distributed mode
        )

        logger.info(f"=" * 60)
        logger.info(f"Worker {worker_id} Training Complete!")
        logger.info(f"=" * 60)
        logger.info(f"  Steps: {results['num_steps']}")
        logger.info(f"  Time: {results['total_time']:.2f}s")
        logger.info(f"  Speed: {results['steps_per_sec']:.2f} steps/s")
        logger.info(f"  Initial loss: {results['initial_loss']:.4f}")
        logger.info(f"  Final loss: {results['final_loss']:.4f}")
        logger.info(f"  Improvement: {(1 - results['final_loss']/results['initial_loss'])*100:.1f}%")
        logger.info(f"=" * 60)

        return results

    except Exception as e:
        logger.error(f"Worker {worker_id} failed: {e}", exc_info=True)
        return None

    finally:
        # Cleanup
        logger.info(f"Stopping worker {worker_id}...")
        await worker.stop()
        logger.info(f"✓ Worker {worker_id} stopped")


async def main():
    """
    Main test function.

    Runs 2 workers concurrently for distributed training.
    """
    logger.info("=" * 80)
    logger.info("Legion 2-Worker Distributed Training Test")
    logger.info("=" * 80)

    # Wait for coordinator
    if not await wait_for_coordinator():
        logger.error("Coordinator not available. Please start it first:")
        logger.error("  python -m coordinator.server")
        return 1

    logger.info("")
    logger.info("Starting 2 workers for distributed training...")
    logger.info("")

    # Run two workers concurrently
    results = await asyncio.gather(
        run_worker("worker_1", port=50051, num_steps=50),
        run_worker("worker_2", port=50052, num_steps=50),
        return_exceptions=True
    )

    # Check results
    logger.info("")
    logger.info("=" * 80)
    logger.info("Test Results")
    logger.info("=" * 80)

    success = True
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Worker {i+1} failed with exception: {result}")
            success = False
        elif result is None:
            logger.error(f"Worker {i+1} returned no results")
            success = False
        else:
            logger.info(f"Worker {i+1}: ✓ Completed successfully")
            logger.info(f"  Final loss: {result['final_loss']:.4f}")

    logger.info("")
    if success:
        logger.info("=" * 80)
        logger.info("✓ Test PASSED - 2-worker distributed training successful!")
        logger.info("=" * 80)
        return 0
    else:
        logger.info("=" * 80)
        logger.info("✗ Test FAILED - Check logs above for errors")
        logger.info("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
