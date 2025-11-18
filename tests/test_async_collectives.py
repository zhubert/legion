"""
Tests for asynchronous collective operations.

Tests the async parameter server architecture:
- Async parameter fetching with parallel requests
- Async gradient pushing with retries
- Fallback to cached parameters
- Timeout handling
"""

import pytest
import asyncio
import torch

from communication.async_collectives import AsyncParameterFetcher, AsyncGradientPusher
from communication.grpc_client import WorkerGRPCClient
from communication.grpc_server import WorkerGRPCServer


class TestAsyncParameterFetcher:
    """Test AsyncParameterFetcher for parallel parameter requests."""

    @pytest.mark.asyncio
    async def test_parallel_parameter_fetch(self):
        """Test fetching parameters from multiple servers in parallel."""
        # Create mock parameter servers
        param_store1 = {
            "layer1.weight": torch.randn(100, 50)
        }
        param_store2 = {
            "layer2.weight": torch.randn(50, 25)
        }

        server1 = WorkerGRPCServer(
            worker_id="worker_0",
            parameter_store=param_store1,
            host="127.0.0.1",
            port=50051
        )
        server2 = WorkerGRPCServer(
            worker_id="worker_1",
            parameter_store=param_store2,
            host="127.0.0.1",
            port=50052
        )

        try:
            await server1.start()
            await server2.start()
            await asyncio.sleep(0.2)  # Let servers start

            # Create client and fetcher
            client = WorkerGRPCClient(worker_id="test_client")
            fetcher = AsyncParameterFetcher(client, timeout=5.0)

            # Fetch parameters in parallel
            parameter_owners = [
                ("127.0.0.1:50051", "layer1.weight", 0, -1),
                ("127.0.0.1:50052", "layer2.weight", 0, -1),
            ]

            import time
            start = time.time()
            params = await fetcher.fetch_parameters_async(parameter_owners)
            elapsed = time.time() - start

            # Verify both fetched successfully
            assert "layer1.weight[0:-1]" in params
            assert "layer2.weight[0:-1]" in params
            assert params["layer1.weight[0:-1]"] is not None
            assert params["layer2.weight[0:-1]"] is not None

            # Verify shapes
            assert params["layer1.weight[0:-1]"].shape == (100, 50)
            assert params["layer2.weight[0:-1]"].shape == (50, 25)

            # Parallel fetching should be faster than sequential
            # (though with localhost, difference is small)
            print(f"Parallel fetch took {elapsed*1000:.1f}ms")
            assert elapsed < 1.0  # Should be fast on localhost

        finally:
            await server1.stop()
            await server2.stop()
            await client.close()

    @pytest.mark.asyncio
    async def test_parameter_fetch_with_failure(self):
        """Test fallback to cached parameters when server fails."""
        # Create one working server
        param_store = {
            "layer1.weight": torch.randn(100, 50)
        }

        server = WorkerGRPCServer(
            worker_id="worker_0",
            parameter_store=param_store,
            host="127.0.0.1",
            port=50051
        )

        try:
            await server.start()
            await asyncio.sleep(0.2)

            client = WorkerGRPCClient(worker_id="test_client")
            fetcher = AsyncParameterFetcher(client, timeout=2.0, enable_cache=True)

            # First fetch - should succeed and populate cache
            parameter_owners = [
                ("127.0.0.1:50051", "layer1.weight", 0, -1),
            ]

            params = await fetcher.fetch_parameters_async(parameter_owners)
            assert params["layer1.weight[0:-1]"] is not None

            # Stop server
            await server.stop()
            await asyncio.sleep(0.1)

            # Second fetch - server down, should use cache
            params = await fetcher.fetch_parameters_async(parameter_owners)
            assert params["layer1.weight[0:-1]"] is not None  # From cache
            assert params["layer1.weight[0:-1]"].shape == (100, 50)

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_parameter_fetch_timeout(self):
        """Test timeout handling for slow servers."""
        client = WorkerGRPCClient(worker_id="test_client")
        fetcher = AsyncParameterFetcher(client, timeout=0.5)  # Very short timeout

        # Try to fetch from non-existent server
        parameter_owners = [
            ("127.0.0.1:59999", "layer1.weight", 0, -1),  # Non-existent port
        ]

        params = await fetcher.fetch_parameters_async(parameter_owners)

        # Should return None (no cache, server down)
        assert params["layer1.weight[0:-1]"] is None

        await client.close()


class TestAsyncGradientPusher:
    """Test AsyncGradientPusher for parallel gradient sends."""

    @pytest.mark.asyncio
    async def test_parallel_gradient_push(self):
        """Test pushing gradients to multiple servers in parallel."""
        # Create mock parameter servers
        param_store1 = {"layer1.weight": torch.randn(100, 50)}
        param_store2 = {"layer2.weight": torch.randn(50, 25)}

        server1 = WorkerGRPCServer(
            worker_id="worker_0",
            parameter_store=param_store1,
            host="127.0.0.1",
            port=50051
        )
        server2 = WorkerGRPCServer(
            worker_id="worker_1",
            parameter_store=param_store2,
            host="127.0.0.1",
            port=50052
        )

        # Set world size for aggregation
        server1.servicer.set_world_size(3)
        server2.servicer.set_world_size(3)

        try:
            await server1.start()
            await server2.start()
            await asyncio.sleep(0.2)

            # Create client and pusher
            client = WorkerGRPCClient(worker_id="test_client")
            pusher = AsyncGradientPusher(client, timeout=5.0)

            # Push gradients in parallel
            grad1 = torch.randn(100, 50)
            grad2 = torch.randn(50, 25)

            gradient_targets = [
                ("127.0.0.1:50051", "layer1.weight", grad1, 0, -1),
                ("127.0.0.1:50052", "layer2.weight", grad2, 0, -1),
            ]

            import time
            start = time.time()
            results = await pusher.push_gradients_async(gradient_targets, version=1)
            elapsed = time.time() - start

            # Verify both pushed successfully
            assert results["layer1.weight[0:-1]"] is True
            assert results["layer2.weight[0:-1]"] is True

            print(f"Parallel push took {elapsed*1000:.1f}ms")
            assert elapsed < 1.0  # Should be fast on localhost

        finally:
            await server1.stop()
            await server2.stop()
            await client.close()

    @pytest.mark.asyncio
    async def test_gradient_push_with_retry(self):
        """Test retry logic when gradient push fails transiently."""
        client = WorkerGRPCClient(worker_id="test_client")
        pusher = AsyncGradientPusher(client, timeout=0.5, max_retries=2)

        # Try to push to non-existent server
        grad = torch.randn(10, 10)
        gradient_targets = [
            ("127.0.0.1:59999", "layer1.weight", grad, 0, -1),
        ]

        results = await pusher.push_gradients_async(gradient_targets, version=1)

        # Should fail after retries
        assert results["layer1.weight[0:-1]"] is False

        # Check that gradient was buffered
        buffered = pusher.get_buffered_gradients("layer1.weight[0:-1]")
        assert 1 in buffered  # Version 1 should be buffered
        assert buffered[1].shape == (10, 10)

        await client.close()


class TestAsyncCollectivesIntegration:
    """Integration tests for async parameter fetch + gradient push."""

    @pytest.mark.asyncio
    async def test_full_async_cycle(self):
        """Test complete cycle: fetch params → compute → push gradients."""
        # Create parameter server
        param_store = {
            "model.weight": torch.randn(50, 25)
        }

        server = WorkerGRPCServer(
            worker_id="worker_0",
            parameter_store=param_store,
            host="127.0.0.1",
            port=50051
        )
        server.servicer.set_world_size(2)

        try:
            await server.start()
            await asyncio.sleep(0.2)

            client = WorkerGRPCClient(worker_id="worker_1")
            fetcher = AsyncParameterFetcher(client)
            pusher = AsyncGradientPusher(client)

            # 1. Fetch parameters
            parameter_owners = [
                ("127.0.0.1:50051", "model.weight", 0, -1),
            ]
            params = await fetcher.fetch_parameters_async(parameter_owners)
            param = params["model.weight[0:-1]"]
            assert param is not None
            assert param.shape == (50, 25)

            # 2. Simulate computation (just create fake gradients)
            gradients = torch.randn_like(param)

            # 3. Push gradients back
            gradient_targets = [
                ("127.0.0.1:50051", "model.weight", gradients, 0, -1),
            ]
            results = await pusher.push_gradients_async(gradient_targets, version=1)
            assert results["model.weight[0:-1]"] is True

            # 4. Verify gradients were accumulated on server
            accumulated = await server.servicer.get_accumulated_gradients(
                version=1,
                param_name="model.weight",
                wait_for_threshold=False
            )
            assert accumulated is not None
            assert accumulated.shape == (50, 25)

        finally:
            await server.stop()
            await client.close()

    @pytest.mark.asyncio
    async def test_multiple_workers_gradient_aggregation(self):
        """Test gradient aggregation from multiple async workers."""
        # Create parameter server
        param_store = {"model.weight": torch.randn(10, 10)}
        server = WorkerGRPCServer(
            worker_id="server",
            parameter_store=param_store,
            host="127.0.0.1",
            port=50051
        )
        server.servicer.set_world_size(3)  # Expect 3 workers
        server.servicer.set_aggregation_threshold(0.67)  # 67% = 2/3 workers

        try:
            await server.start()
            await asyncio.sleep(0.2)

            # Create 3 workers pushing gradients
            workers = []
            for i in range(3):
                client = WorkerGRPCClient(worker_id=f"worker_{i}")
                pusher = AsyncGradientPusher(client)
                workers.append((client, pusher))

            # Workers push gradients asynchronously
            tasks = []
            for i, (client, pusher) in enumerate(workers):
                grad = torch.ones(10, 10) * (i + 1)  # Unique gradients per worker
                gradient_targets = [
                    ("127.0.0.1:50051", "model.weight", grad, 0, -1),
                ]
                task = pusher.push_gradients_async(gradient_targets, version=1)
                tasks.append(task)

            # All push in parallel
            results = await asyncio.gather(*tasks)

            # All should succeed
            for result in results:
                assert result["model.weight[0:-1]"] is True

            # Wait a bit for accumulation
            await asyncio.sleep(0.1)

            # Server should have aggregated gradients from all 3 workers
            accumulated = await server.servicer.get_accumulated_gradients(
                version=1,
                param_name="model.weight",
                wait_for_threshold=False
            )

            assert accumulated is not None
            # Should be sum of all 3 workers: 1 + 2 + 3 = 6 per element
            expected = torch.ones(10, 10) * 6
            assert torch.allclose(accumulated, expected)

        finally:
            await server.stop()
            for client, _ in workers:
                await client.close()
