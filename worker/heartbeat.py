"""
Heartbeat manager for maintaining worker liveness with coordinator.

Sends periodic heartbeats to prove the worker is still alive and responsive.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from worker.coordinator_client import CoordinatorClient


logger = logging.getLogger(__name__)


class HeartbeatManager:
    """
    Manages periodic heartbeat sending to coordinator.

    Runs in background, automatically retries on failure with exponential backoff,
    and tracks health status.
    """

    def __init__(
        self,
        coordinator_client: CoordinatorClient,
        interval_seconds: int = 30,
        max_failures: int = 3
    ):
        """
        Initialize heartbeat manager.

        Args:
            coordinator_client: Client for coordinator communication
            interval_seconds: Interval between heartbeats
            max_failures: Maximum consecutive failures before marking unhealthy
        """
        self.coordinator_client = coordinator_client
        self.interval_seconds = interval_seconds
        self.max_failures = max_failures

        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_success: Optional[datetime] = None
        self._last_failure: Optional[datetime] = None
        self._consecutive_failures = 0
        self._total_sent = 0
        self._total_failed = 0

    async def start(self):
        """
        Start heartbeat manager.

        Begins sending periodic heartbeats in background task.
        """
        if self._running:
            logger.warning("Heartbeat manager already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info(
            f"Heartbeat manager started (interval: {self.interval_seconds}s)"
        )

    async def stop(self):
        """
        Stop heartbeat manager.

        Cancels background task and waits for cleanup.
        """
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Heartbeat manager stopped")

    async def _heartbeat_loop(self):
        """
        Main heartbeat loop.

        Sends heartbeats periodically with exponential backoff on failure.
        """
        backoff = 1.0  # Start with 1 second backoff

        while self._running:
            try:
                # Send heartbeat
                success = await self.coordinator_client.heartbeat()
                self._total_sent += 1

                if success:
                    # Success - reset failure counter and backoff
                    self._last_success = datetime.now()
                    self._consecutive_failures = 0
                    backoff = 1.0

                    logger.debug(
                        f"Heartbeat successful (total: {self._total_sent}, "
                        f"failures: {self._total_failed})"
                    )

                    # Wait for next interval
                    await asyncio.sleep(self.interval_seconds)

                else:
                    # Failure - increment counter and apply backoff
                    self._last_failure = datetime.now()
                    self._consecutive_failures += 1
                    self._total_failed += 1

                    logger.warning(
                        f"Heartbeat failed ({self._consecutive_failures}/{self.max_failures})"
                    )

                    # Exponential backoff (max 60s)
                    backoff = min(backoff * 2, 60.0)
                    await asyncio.sleep(backoff)

            except asyncio.CancelledError:
                logger.info("Heartbeat loop cancelled")
                break

            except Exception as e:
                logger.error(f"Unexpected error in heartbeat loop: {e}")
                self._last_failure = datetime.now()
                self._consecutive_failures += 1
                self._total_failed += 1

                # Backoff on error
                backoff = min(backoff * 2, 60.0)
                await asyncio.sleep(backoff)

    def is_healthy(self) -> bool:
        """
        Check if heartbeat is healthy.

        Returns:
            True if healthy (recent success and not too many failures)
        """
        # Not healthy if never succeeded
        if self._last_success is None:
            return False

        # Not healthy if too many consecutive failures
        if self._consecutive_failures >= self.max_failures:
            return False

        # Not healthy if last success was too long ago
        # (more than 2x the interval)
        time_since_success = datetime.now() - self._last_success
        if time_since_success > timedelta(seconds=self.interval_seconds * 2):
            return False

        return True

    def get_status(self) -> dict:
        """
        Get heartbeat manager status.

        Returns:
            Status dictionary with statistics
        """
        return {
            'running': self._running,
            'healthy': self.is_healthy(),
            'last_success': self._last_success.isoformat() if self._last_success else None,
            'last_failure': self._last_failure.isoformat() if self._last_failure else None,
            'consecutive_failures': self._consecutive_failures,
            'total_sent': self._total_sent,
            'total_failed': self._total_failed,
            'failure_rate': self._total_failed / self._total_sent if self._total_sent > 0 else 0.0
        }

    def get_last_success_time(self) -> Optional[datetime]:
        """
        Get timestamp of last successful heartbeat.

        Returns:
            Datetime of last success or None
        """
        return self._last_success

    def get_consecutive_failures(self) -> int:
        """
        Get number of consecutive failures.

        Returns:
            Count of consecutive failures
        """
        return self._consecutive_failures

    def is_running(self) -> bool:
        """
        Check if heartbeat manager is running.

        Returns:
            True if running, False otherwise
        """
        return self._running
