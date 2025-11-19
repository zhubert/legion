"""
Assembler service for Legion distributed training.

This service runs independently and listens to the coordinator's WebSocket
for checkpoint assembly requests. When all workers complete training, the
coordinator broadcasts an 'assemble_checkpoint' event, and this service
assembles the complete model from worker shards.
"""

import asyncio
import logging
import argparse
import json
from typing import Optional
import websockets

from worker.assembler import CheckpointAssembler


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AssemblerService:
    """
    Service that listens for checkpoint assembly requests from coordinator.

    Connects to coordinator WebSocket and assembles checkpoints when
    all workers complete training.
    """

    def __init__(
        self,
        coordinator_url: str = "ws://localhost:8000/ws/events",
        checkpoint_dir: str = "./checkpoints"
    ):
        """
        Initialize assembler service.

        Args:
            coordinator_url: WebSocket URL of coordinator
            checkpoint_dir: Directory containing checkpoint shards
        """
        self.coordinator_url = coordinator_url
        self.checkpoint_dir = checkpoint_dir
        self.assembler = CheckpointAssembler(checkpoint_dir)
        self.running = False

        logger.info(f"Assembler service initialized")
        logger.info(f"  Coordinator: {coordinator_url}")
        logger.info(f"  Checkpoint dir: {checkpoint_dir}")

    async def start(self):
        """Start the assembler service and listen for messages."""
        self.running = True

        logger.info("Starting assembler service...")
        logger.info(f"Connecting to coordinator at {self.coordinator_url}")

        while self.running:
            try:
                async with websockets.connect(self.coordinator_url) as websocket:
                    logger.info("Connected to coordinator WebSocket")

                    # Listen for messages
                    async for message in websocket:
                        await self._handle_message(message)

            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection to coordinator closed, reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"WebSocket error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _handle_message(self, message: str):
        """
        Handle incoming WebSocket message from coordinator.

        Args:
            message: JSON message from coordinator
        """
        try:
            data = json.loads(message)
            event = data.get('event')

            if event == 'assemble_checkpoint':
                await self._handle_assemble_checkpoint(data)
            else:
                # Ignore other events
                logger.debug(f"Ignoring event: {event}")

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse message: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _handle_assemble_checkpoint(self, data: dict):
        """
        Handle checkpoint assembly request.

        Args:
            data: Message data containing global_step and worker info
        """
        global_step = data.get('global_step')
        num_workers = data.get('num_workers')
        worker_ids = data.get('worker_ids', [])

        logger.info("=" * 60)
        logger.info("Checkpoint Assembly Request Received")
        logger.info("=" * 60)
        logger.info(f"Global step: {global_step}")
        logger.info(f"Workers: {num_workers}")
        logger.info(f"Worker IDs: {worker_ids}")
        logger.info("=" * 60)

        if global_step is None:
            logger.error("No global_step provided in assembly request")
            return

        # Workers save checkpoint shards BEFORE reaching the barrier
        # So by the time we get this message, all shards should already be on disk
        try:
            # Assemble checkpoint from worker shards
            logger.info(f"Assembling checkpoint for step {global_step}...")
            output_path = self.assembler.assemble_checkpoint(
                global_step=global_step,
                validate=True
            )

            logger.info("=" * 60)
            logger.info("Checkpoint Assembly Complete!")
            logger.info("=" * 60)
            logger.info(f"Output: {output_path}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error("=" * 60)
            logger.error("Checkpoint Assembly Failed!")
            logger.error("=" * 60)
            logger.error(f"Error: {e}")
            logger.error("=" * 60)

    async def stop(self):
        """Stop the assembler service."""
        logger.info("Stopping assembler service...")
        self.running = False


async def main(
    coordinator_url: str = "ws://localhost:8000/ws/events",
    checkpoint_dir: str = "./checkpoints"
):
    """
    Main entry point for assembler service.

    Args:
        coordinator_url: WebSocket URL of coordinator
        checkpoint_dir: Directory containing checkpoint shards
    """
    service = AssemblerService(
        coordinator_url=coordinator_url,
        checkpoint_dir=checkpoint_dir
    )

    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        await service.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legion Checkpoint Assembler Service")
    parser.add_argument(
        "--coordinator-url",
        type=str,
        default="ws://localhost:8000/ws/events",
        help="WebSocket URL of coordinator (default: ws://localhost:8000/ws/events)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory containing checkpoint shards (default: ./checkpoints)"
    )

    args = parser.parse_args()

    asyncio.run(main(
        coordinator_url=args.coordinator_url,
        checkpoint_dir=args.checkpoint_dir
    ))
