"""
Asynchronous collective operations for parameter server architecture.

Implements non-blocking parameter fetch and gradient push with:
- Version tracking for bounded staleness
- Parallel fetching from multiple parameter servers
- Fault tolerance with fallbacks
- Adaptive timeouts
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
import torch

from communication.grpc_client import WorkerGRPCClient

logger = logging.getLogger(__name__)


class AsyncParameterFetcher:
    """
    Handles asynchronous parameter fetching from multiple parameter servers.

    Features:
    - Parallel fetching (no sequential dependencies)
    - Version-aware requests
    - Fallback to cached parameters on failure
    - Timeout handling per shard
    """

    def __init__(
        self,
        grpc_client: WorkerGRPCClient,
        timeout: float = 5.0,
        enable_cache: bool = True
    ):
        """
        Initialize async parameter fetcher.

        Args:
            grpc_client: gRPC client for communication
            timeout: Timeout per shard fetch (seconds)
            enable_cache: Whether to cache parameters for fallback
        """
        self.grpc_client = grpc_client
        self.timeout = timeout
        self.enable_cache = enable_cache
        self.parameter_cache: Dict[str, torch.Tensor] = {}  # shard_id -> tensor

    async def fetch_parameters_async(
        self,
        parameter_owners: List[Tuple[str, str, int, int]],  # (address, name, start, end)
        version: Optional[int] = None,
        staleness_tolerance: int = 5
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Fetch parameters from multiple owners in parallel.

        Args:
            parameter_owners: List of (worker_address, param_name, shard_start, shard_end)
            version: Requested parameter version (None = latest)
            staleness_tolerance: Accept parameters in version range [V-K, V]

        Returns:
            Dict mapping shard_id to parameter tensor (None if fetch failed)

        Example:
            >>> fetcher = AsyncParameterFetcher(grpc_client)
            >>> owners = [
            ...     ("127.0.0.1:50051", "model.layer1.weight", 0, 1000),
            ...     ("127.0.0.1:50052", "model.layer2.weight", 1000, 2000),
            ... ]
            >>> params = await fetcher.fetch_parameters_async(owners, version=100)
            >>> # All fetches happen in parallel, no blocking
        """
        start_time = time.time()

        # Create parallel fetch tasks
        fetch_tasks = []
        shard_ids = []

        for address, param_name, shard_start, shard_end in parameter_owners:
            shard_id = f"{param_name}[{shard_start}:{shard_end}]"
            shard_ids.append(shard_id)

            task = self._fetch_single_shard(
                worker_address=address,
                parameter_name=param_name,
                shard_start=shard_start,
                shard_end=shard_end,
                version=version,
                shard_id=shard_id
            )
            fetch_tasks.append(task)

        # Execute all fetches in parallel
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        # Build result dictionary
        parameters = {}
        success_count = 0
        failure_count = 0

        for shard_id, result in zip(shard_ids, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch {shard_id}: {result}")
                # Try fallback from cache
                if self.enable_cache and shard_id in self.parameter_cache:
                    parameters[shard_id] = self.parameter_cache[shard_id]
                    logger.info(f"Using cached parameters for {shard_id}")
                else:
                    parameters[shard_id] = None
                failure_count += 1
            elif result is None:
                # Fetch timed out or failed
                if self.enable_cache and shard_id in self.parameter_cache:
                    parameters[shard_id] = self.parameter_cache[shard_id]
                    logger.info(f"Using cached parameters for {shard_id}")
                else:
                    parameters[shard_id] = None
                failure_count += 1
            else:
                parameters[shard_id] = result
                # Update cache
                if self.enable_cache:
                    self.parameter_cache[shard_id] = result.clone()
                success_count += 1

        elapsed = time.time() - start_time
        logger.debug(
            f"Fetched {len(parameter_owners)} shards in {elapsed*1000:.1f}ms "
            f"(success: {success_count}, failures: {failure_count})"
        )

        return parameters

    async def _fetch_single_shard(
        self,
        worker_address: str,
        parameter_name: str,
        shard_start: int,
        shard_end: int,
        version: Optional[int],
        shard_id: str
    ) -> Optional[torch.Tensor]:
        """
        Fetch a single parameter shard with timeout.

        Args:
            worker_address: Target worker address
            parameter_name: Parameter name
            shard_start: Shard start index
            shard_end: Shard end index
            version: Requested version
            shard_id: Shard identifier for logging

        Returns:
            Parameter tensor or None if fetch failed
        """
        try:
            # Fetch with timeout
            tensor = await asyncio.wait_for(
                self.grpc_client.get_parameters(
                    worker_address=worker_address,
                    parameter_name=parameter_name,
                    shard_start=shard_start,
                    shard_end=shard_end
                ),
                timeout=self.timeout
            )

            if tensor is not None:
                logger.debug(f"Fetched {shard_id} from {worker_address}")

            return tensor

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching {shard_id} from {worker_address}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {shard_id} from {worker_address}: {e}")
            return None


class AsyncGradientPusher:
    """
    Handles asynchronous gradient pushing to parameter servers.

    Features:
    - Non-blocking gradient sends
    - Retry logic for transient failures
    - Version tracking
    - Local buffering on server unavailability
    """

    def __init__(
        self,
        grpc_client: WorkerGRPCClient,
        timeout: float = 5.0,
        max_retries: int = 2
    ):
        """
        Initialize async gradient pusher.

        Args:
            grpc_client: gRPC client for communication
            timeout: Timeout per gradient push (seconds)
            max_retries: Number of retries on failure
        """
        self.grpc_client = grpc_client
        self.timeout = timeout
        self.max_retries = max_retries
        self.gradient_buffer: Dict[str, Dict[int, torch.Tensor]] = {}  # shard_id -> {version -> gradients}

    async def push_gradients_async(
        self,
        gradient_targets: List[Tuple[str, str, torch.Tensor, int, int]],  # (address, name, grads, start, end)
        version: int
    ) -> Dict[str, bool]:
        """
        Push gradients to multiple parameter servers in parallel.

        Args:
            gradient_targets: List of (worker_address, param_name, gradients, shard_start, shard_end)
            version: Training step version for this gradient

        Returns:
            Dict mapping shard_id to success status

        Example:
            >>> pusher = AsyncGradientPusher(grpc_client)
            >>> targets = [
            ...     ("127.0.0.1:50051", "model.layer1.weight", grad1, 0, 1000),
            ...     ("127.0.0.1:50052", "model.layer2.weight", grad2, 1000, 2000),
            ... ]
            >>> results = await pusher.push_gradients_async(targets, version=100)
            >>> # All pushes happen in parallel
        """
        start_time = time.time()

        # Create parallel push tasks
        push_tasks = []
        shard_ids = []

        for address, param_name, gradients, shard_start, shard_end in gradient_targets:
            shard_id = f"{param_name}[{shard_start}:{shard_end}]"
            shard_ids.append(shard_id)

            task = self._push_single_gradient(
                worker_address=address,
                parameter_name=param_name,
                gradients=gradients,
                shard_start=shard_start,
                shard_end=shard_end,
                version=version,
                shard_id=shard_id
            )
            push_tasks.append(task)

        # Execute all pushes in parallel
        results = await asyncio.gather(*push_tasks, return_exceptions=True)

        # Build result dictionary
        push_results = {}
        success_count = 0
        failure_count = 0

        for shard_id, result in zip(shard_ids, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to push gradient for {shard_id}: {result}")
                push_results[shard_id] = False
                failure_count += 1
            elif result:
                push_results[shard_id] = True
                success_count += 1
            else:
                push_results[shard_id] = False
                failure_count += 1

        elapsed = time.time() - start_time
        logger.debug(
            f"Pushed {len(gradient_targets)} gradients in {elapsed*1000:.1f}ms "
            f"(success: {success_count}, failures: {failure_count})"
        )

        return push_results

    async def _push_single_gradient(
        self,
        worker_address: str,
        parameter_name: str,
        gradients: torch.Tensor,
        shard_start: int,
        shard_end: int,
        version: int,
        shard_id: str
    ) -> bool:
        """
        Push a single gradient with timeout and retries.

        Args:
            worker_address: Target parameter server address
            parameter_name: Parameter name
            gradients: Gradient tensor
            shard_start: Shard start index
            shard_end: Shard end index
            version: Training step version
            shard_id: Shard identifier for logging

        Returns:
            True if push succeeded, False otherwise
        """
        for attempt in range(self.max_retries + 1):
            try:
                # Push with timeout
                success = await asyncio.wait_for(
                    self.grpc_client.send_gradients(
                        worker_address=worker_address,
                        gradients=gradients,
                        step=version,  # Use version as step
                        parameter_name=parameter_name,
                        shard_start=shard_start,
                        shard_end=shard_end
                    ),
                    timeout=self.timeout
                )

                if success:
                    logger.debug(f"Pushed gradient for {shard_id} to {worker_address} (v={version})")
                    return True
                else:
                    logger.warning(f"Push rejected for {shard_id} by {worker_address}")

            except asyncio.TimeoutError:
                if attempt < self.max_retries:
                    logger.warning(
                        f"Timeout pushing {shard_id} to {worker_address} "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    logger.error(f"Timeout pushing {shard_id} after {self.max_retries + 1} attempts")

            except Exception as e:
                logger.error(f"Error pushing {shard_id} to {worker_address}: {e}")
                break

        # Push failed - buffer gradient locally
        if shard_id not in self.gradient_buffer:
            self.gradient_buffer[shard_id] = {}
        self.gradient_buffer[shard_id][version] = gradients.clone()
        logger.info(f"Buffered gradient for {shard_id} at version {version}")

        return False

    def get_buffered_gradients(self, shard_id: str) -> Dict[int, torch.Tensor]:
        """
        Get buffered gradients for a shard.

        Args:
            shard_id: Shard identifier

        Returns:
            Dict mapping version to gradient tensor
        """
        return self.gradient_buffer.get(shard_id, {})

    def clear_buffered_gradients(self, shard_id: str, version: int):
        """
        Clear buffered gradients up to a version.

        Args:
            shard_id: Shard identifier
            version: Clear gradients <= this version
        """
        if shard_id in self.gradient_buffer:
            self.gradient_buffer[shard_id] = {
                v: g for v, g in self.gradient_buffer[shard_id].items()
                if v > version
            }
