"""
Coordinator client for worker-coordinator communication.

Handles all HTTP communication with the coordinator server.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

import httpx


logger = logging.getLogger(__name__)


class CoordinatorClient:
    """
    Client for communicating with the Legion coordinator.

    Provides methods for worker registration, heartbeat, metrics reporting,
    and cluster discovery.
    """

    def __init__(
        self,
        coordinator_url: str,
        worker_id: str,
        ip_address: str,
        port: int,
        timeout: float = 30.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        gpu_info: Optional[Dict[str, Any]] = None,
        cpu_cores: Optional[int] = None,
        ram_gb: Optional[float] = None,
        bandwidth_mbps: Optional[float] = None
    ):
        """
        Initialize coordinator client.

        Args:
            coordinator_url: URL of coordinator server
            worker_id: Unique worker identifier
            ip_address: Worker IP address
            port: Worker gRPC port
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            retry_delay: Base delay between retries (exponential backoff)
            gpu_info: GPU information dict
            cpu_cores: Number of CPU cores
            ram_gb: RAM in GB
            bandwidth_mbps: Network bandwidth in Mbps
        """
        self.coordinator_url = coordinator_url.rstrip('/')
        self.worker_id = worker_id
        self.ip_address = ip_address
        self.port = port
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Hardware info
        self.gpu_info = gpu_info
        self.cpu_cores = cpu_cores
        self.ram_gb = ram_gb
        self.bandwidth_mbps = bandwidth_mbps

        # HTTP client (async)
        self._client: Optional[httpx.AsyncClient] = None
        self._registered = False

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def wait_at_barrier(
        self,
        step: str = "training_complete",
        global_step: Optional[int] = None,
        poll_interval: float = 5.0,
        timeout: float = 300.0
    ) -> tuple[bool, Optional[int]]:
        """
        Wait at a distributed barrier until all workers reach it.

        Args:
            step: Barrier step name
            global_step: Training step number (for checkpoint creation)
            poll_interval: How often to poll in seconds (default: 5.0)
            timeout: Maximum time to wait in seconds (default: 300.0)

        Returns:
            Tuple of (success: bool, agreed_global_step: Optional[int])
            - success: True if barrier completed, False if timeout
            - agreed_global_step: The global_step agreed upon by all workers (if provided)
        """
        import asyncio
        elapsed = 0.0

        while elapsed < timeout:
            try:
                params = {"worker_id": self.worker_id, "step": step}
                if global_step is not None:
                    params["global_step"] = global_step

                response = await self._request_with_retry(
                    "POST",
                    "/training/barrier",
                    params=params
                )

                data = response.json()

                if data and data.get("all_ready"):
                    agreed_step = data.get("global_step")
                    logger.info(f"Barrier '{step}' complete - all workers ready (global_step={agreed_step})")
                    return (True, agreed_step)

                reached = data.get("reached", 0)
                total = data.get("total", 0)
                waiting = data.get("waiting_for", [])

                logger.debug(
                    f"Barrier '{step}': {reached}/{total} workers ready, "
                    f"waiting for: {waiting}"
                )

            except Exception as e:
                logger.warning(f"Barrier check failed: {e}")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        logger.error(f"Barrier '{step}' timeout after {timeout}s")
        return (False, None)

    async def close(self):
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint (e.g., "/workers/register")
            **kwargs: Additional arguments for httpx request

        Returns:
            HTTP response

        Raises:
            httpx.HTTPError: If all retry attempts fail
        """
        client = await self._get_client()
        url = f"{self.coordinator_url}{endpoint}"

        last_exception = None
        for attempt in range(self.retry_attempts):
            try:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response

            except httpx.HTTPError as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Request to {endpoint} failed (attempt {attempt + 1}/{self.retry_attempts}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Request to {endpoint} failed after {self.retry_attempts} attempts: {e}"
                    )

        raise last_exception

    async def register(self) -> bool:
        """
        Register worker with coordinator.

        Returns:
            True if registration successful, False otherwise
        """
        try:
            payload = {
                "worker_id": self.worker_id,
                "ip_address": self.ip_address,
                "port": self.port,
                "gpu_info": self.gpu_info,
                "cpu_cores": self.cpu_cores,
                "ram_gb": self.ram_gb,
                "bandwidth_mbps": self.bandwidth_mbps
            }

            response = await self._request_with_retry(
                "POST",
                "/workers/register",
                json=payload
            )

            self._registered = True
            logger.info(f"Worker {self.worker_id} registered successfully")
            return True

        except httpx.HTTPError as e:
            logger.error(f"Failed to register worker: {e}")
            return False

    async def deregister(self) -> bool:
        """
        Deregister worker from coordinator.

        Returns:
            True if deregistration successful, False otherwise
        """
        try:
            await self._request_with_retry(
                "DELETE",
                f"/workers/{self.worker_id}"
            )

            self._registered = False
            logger.info(f"Worker {self.worker_id} deregistered successfully")
            return True

        except httpx.HTTPError as e:
            logger.error(f"Failed to deregister worker: {e}")
            return False

    async def heartbeat(self) -> bool:
        """
        Send heartbeat to coordinator.

        Returns:
            True if heartbeat successful, False otherwise
        """
        try:
            await self._request_with_retry(
                "POST",
                f"/workers/{self.worker_id}/heartbeat"
            )
            logger.debug(f"Heartbeat sent for worker {self.worker_id}")
            return True

        except httpx.HTTPError as e:
            logger.warning(f"Heartbeat failed: {e}")
            return False

    async def report_latency(
        self,
        other_worker_id: str,
        latency_ms: float
    ) -> bool:
        """
        Report latency measurement to another worker.

        Args:
            other_worker_id: ID of the other worker
            latency_ms: Measured latency in milliseconds

        Returns:
            True if report successful, False otherwise
        """
        try:
            payload = {
                "worker_a": self.worker_id,
                "worker_b": other_worker_id,
                "latency_ms": latency_ms
            }

            await self._request_with_retry(
                "POST",
                "/latency/report",
                json=payload
            )

            logger.debug(
                f"Reported latency to {other_worker_id}: {latency_ms:.2f}ms"
            )
            return True

        except httpx.HTTPError as e:
            logger.warning(f"Failed to report latency: {e}")
            return False

    async def report_metric(
        self,
        global_step: int,
        loss: Optional[float] = None,
        throughput: Optional[float] = None,
        memory_usage_gb: Optional[float] = None
    ) -> bool:
        """
        Report training metric to coordinator.

        Deprecated: Use report_metrics_batch for better performance.

        Args:
            global_step: Current training step
            loss: Loss value
            throughput: Throughput in samples/sec
            memory_usage_gb: Memory usage in GB

        Returns:
            True if report successful, False otherwise
        """
        try:
            payload = {
                "worker_id": self.worker_id,
                "global_step": global_step,
                "loss": loss,
                "throughput": throughput,
                "memory_usage_gb": memory_usage_gb
            }

            await self._request_with_retry(
                "POST",
                "/metrics/report",
                json=payload
            )

            logger.debug(
                f"Reported metric for step {global_step}: loss={loss}"
            )
            return True

        except httpx.HTTPError as e:
            logger.warning(f"Failed to report metric: {e}")
            return False

    async def report_metrics_batch(
        self,
        metrics: List[Dict[str, Any]]
    ) -> bool:
        """
        Report multiple training metrics to coordinator in a single request.

        This is much more efficient than calling report_metric multiple times.

        Args:
            metrics: List of metric dictionaries with keys:
                     - global_step (required)
                     - loss (optional)
                     - throughput (optional)
                     - memory_usage_gb (optional)

        Returns:
            True if report successful, False otherwise
        """
        if not metrics:
            return True

        try:
            payload = {
                "worker_id": self.worker_id,
                "metrics": metrics
            }

            await self._request_with_retry(
                "POST",
                "/metrics/report-batch",
                json=payload
            )

            logger.debug(
                f"Reported {len(metrics)} metrics in batch "
                f"(steps {metrics[0]['global_step']}-{metrics[-1]['global_step']})"
            )
            return True

        except httpx.HTTPError as e:
            logger.warning(f"Failed to report metrics batch: {e}")
            return False

    async def get_worker_info(self, worker_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get worker information from coordinator.

        Args:
            worker_id: Worker ID (defaults to self)

        Returns:
            Worker info dictionary or None if not found
        """
        try:
            worker_id = worker_id or self.worker_id
            response = await self._request_with_retry(
                "GET",
                f"/workers/{worker_id}"
            )

            return response.json()

        except httpx.HTTPError as e:
            logger.warning(f"Failed to get worker info: {e}")
            return None

    async def get_all_workers(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all workers from coordinator.

        Args:
            status: Filter by status ('online', 'offline', or None for all)

        Returns:
            List of worker info dictionaries
        """
        try:
            params = {"status": status} if status else {}
            response = await self._request_with_retry(
                "GET",
                "/workers",
                params=params
            )

            data = response.json()
            return data.get('workers', [])

        except httpx.HTTPError as e:
            logger.warning(f"Failed to get workers: {e}")
            return []

    async def get_workers(self, status: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get workers from coordinator.

        Args:
            status: Filter by status ('online', 'offline', or None for all)

        Returns:
            Dictionary with 'workers' key containing list of worker info
        """
        try:
            params = {"status": status} if status else {}
            response = await self._request_with_retry(
                "GET",
                "/workers",
                params=params
            )

            return response.json()

        except httpx.HTTPError as e:
            logger.warning(f"Failed to get workers: {e}")
            return None

    async def get_cluster_members(self) -> List[Dict[str, Any]]:
        """
        Get members of this worker's cluster.

        Returns:
            List of worker info dictionaries in same cluster
        """
        # First get self info to find our region
        self_info = await self.get_worker_info()
        if not self_info:
            return []

        region = self_info.get('region')
        if not region:
            return []

        # Get all workers and filter by region
        all_workers = await self.get_all_workers(status='online')
        cluster_members = [
            w for w in all_workers
            if w.get('region') == region and w.get('worker_id') != self.worker_id
        ]

        return cluster_members

    async def get_shard_assignment(self) -> Optional[Tuple[int, int]]:
        """
        Get assigned parameter shard from coordinator.

        Returns:
            Tuple of (shard_start, shard_end) or None if not assigned
        """
        worker_info = await self.get_worker_info()
        if not worker_info:
            return None

        shard_start = worker_info.get('shard_start')
        shard_end = worker_info.get('shard_end')

        if shard_start is not None and shard_end is not None:
            return (shard_start, shard_end)

        return None

    async def compute_clusters(self) -> bool:
        """
        Trigger cluster computation on coordinator.

        Returns:
            True if successful, False otherwise
        """
        try:
            await self._request_with_retry(
                "POST",
                "/clusters/compute"
            )

            logger.info("Cluster computation triggered")
            return True

        except httpx.HTTPError as e:
            logger.warning(f"Failed to compute clusters: {e}")
            return False

    async def get_training_config(self) -> Optional[Dict[str, Any]]:
        """
        Get worker-specific training configuration from coordinator.

        This includes the global training config plus worker-specific settings
        like rank, world_size, and any custom overrides (e.g., batch size).

        Returns:
            Training configuration dict or None if failed
        """
        try:
            response = await self._request_with_retry(
                "GET",
                f"/training/config/worker/{self.worker_id}"
            )

            data = response.json()

            logger.info(
                f"Fetched training config from coordinator: "
                f"rank={data.get('rank')}, world_size={data.get('world_size')}"
            )
            return data

        except httpx.HTTPError as e:
            logger.error(f"Failed to get training config: {e}")
            return None

    def is_registered(self) -> bool:
        """
        Check if worker is registered.

        Returns:
            True if registered, False otherwise
        """
        return self._registered
