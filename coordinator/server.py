"""
Coordinator server for Legion distributed training.

Provides REST API and WebSocket endpoints for:
- Worker registration and discovery
- Health monitoring
- Regional clustering
- Checkpoint coordination
- Telemetry aggregation
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from coordinator.database import Database
from coordinator.registry import WorkerRegistry
from coordinator.clustering import ClusterManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pydantic models for API

class WorkerRegistration(BaseModel):
    """Worker registration request."""
    worker_id: str = Field(..., description="Unique worker identifier")
    ip_address: str = Field(..., description="Worker IP address")
    port: int = Field(..., description="Worker gRPC port", gt=0, lt=65536)
    gpu_info: Optional[Dict[str, Any]] = Field(None, description="GPU information")
    cpu_cores: Optional[int] = Field(None, description="Number of CPU cores", gt=0)
    ram_gb: Optional[float] = Field(None, description="RAM in GB", gt=0)
    bandwidth_mbps: Optional[float] = Field(None, description="Network bandwidth in Mbps", gt=0)


class HeartbeatRequest(BaseModel):
    """Worker heartbeat request."""
    worker_id: str = Field(..., description="Worker identifier")


class LatencyReport(BaseModel):
    """Latency measurement report."""
    worker_a: str = Field(..., description="First worker ID")
    worker_b: str = Field(..., description="Second worker ID")
    latency_ms: float = Field(..., description="Measured latency in milliseconds", ge=0)


class ShardAssignment(BaseModel):
    """Parameter shard assignment."""
    worker_id: str = Field(..., description="Worker identifier")
    shard_start: int = Field(..., description="Start index of shard", ge=0)
    shard_end: int = Field(..., description="End index of shard", ge=0)


class MetricReport(BaseModel):
    """Training metric report."""
    worker_id: str = Field(..., description="Worker identifier")
    global_step: int = Field(..., description="Training step", ge=0)
    loss: Optional[float] = Field(None, description="Loss value")
    throughput: Optional[float] = Field(None, description="Throughput (samples/sec)")
    memory_usage_gb: Optional[float] = Field(None, description="Memory usage in GB")


class MetricBatchReport(BaseModel):
    """Batch training metric report."""
    worker_id: str = Field(..., description="Worker identifier")
    metrics: List[Dict[str, Any]] = Field(..., description="List of metrics to report")


# WebSocket connection manager

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


# Global state
db: Optional[Database] = None
registry: Optional[WorkerRegistry] = None
cluster_manager: Optional[ClusterManager] = None
ws_manager: ConnectionManager = ConnectionManager()


# Background tasks

async def heartbeat_monitor():
    """Background task to monitor worker heartbeats."""
    while True:
        try:
            offline_workers = registry.check_stale_workers()

            if offline_workers:
                # Broadcast worker offline events
                for worker in offline_workers:
                    await ws_manager.broadcast({
                        'event': 'worker_offline',
                        'worker_id': worker.worker_id,
                        'timestamp': worker.last_heartbeat.isoformat() if worker.last_heartbeat else None
                    })

        except Exception as e:
            logger.error(f"Error in heartbeat monitor: {e}")

        await asyncio.sleep(30)  # Check every 30 seconds


# Lifespan context manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup/shutdown)."""
    global db, registry, cluster_manager

    # Startup
    logger.info("Starting coordinator server...")
    db = Database("coordinator.db")
    registry = WorkerRegistry(db, heartbeat_timeout=90)
    cluster_manager = ClusterManager(latency_threshold_ms=50.0)

    # Start background tasks
    heartbeat_task = asyncio.create_task(heartbeat_monitor())

    logger.info("Coordinator server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down coordinator server...")
    heartbeat_task.cancel()
    db.close()
    logger.info("Coordinator server shutdown complete")


# Create FastAPI app

app = FastAPI(
    title="Legion Coordinator",
    description="Coordinator server for Legion distributed training",
    version="0.1.0",
    lifespan=lifespan
)


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Legion Coordinator",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    worker_counts = registry.get_worker_count()
    return {
        "status": "healthy",
        "workers": worker_counts
    }


@app.post("/workers/register")
async def register_worker(registration: WorkerRegistration):
    """
    Register a new worker.

    Returns 201 if successful, 409 if worker already exists.
    """
    success = registry.register_worker(
        worker_id=registration.worker_id,
        ip_address=registration.ip_address,
        port=registration.port,
        gpu_info=registration.gpu_info,
        cpu_cores=registration.cpu_cores,
        ram_gb=registration.ram_gb,
        bandwidth_mbps=registration.bandwidth_mbps
    )

    if not success:
        raise HTTPException(status_code=409, detail="Worker already registered")

    # Broadcast worker joined event
    await ws_manager.broadcast({
        'event': 'worker_joined',
        'worker_id': registration.worker_id,
        'ip_address': registration.ip_address
    })

    return JSONResponse(
        status_code=201,
        content={
            "message": "Worker registered successfully",
            "worker_id": registration.worker_id
        }
    )


@app.delete("/workers/{worker_id}")
async def deregister_worker(worker_id: str):
    """
    Deregister a worker.

    Returns 200 if successful, 404 if worker not found.
    """
    success = registry.deregister_worker(worker_id)

    if not success:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Broadcast worker left event
    await ws_manager.broadcast({
        'event': 'worker_left',
        'worker_id': worker_id
    })

    return {"message": "Worker deregistered successfully"}


@app.post("/workers/{worker_id}/heartbeat")
async def worker_heartbeat(worker_id: str):
    """
    Process worker heartbeat.

    Returns 200 if successful, 404 if worker not found.
    """
    success = registry.heartbeat(worker_id)

    if not success:
        raise HTTPException(status_code=404, detail="Worker not found")

    return {"message": "Heartbeat received"}


@app.get("/workers")
async def list_workers(status: Optional[str] = None):
    """
    List all workers, optionally filtered by status.

    Query params:
    - status: Filter by status ('online', 'offline', or omit for all)
    """
    workers = registry.get_all_workers(status=status)

    return {
        "workers": [w.to_dict() for w in workers],
        "count": len(workers)
    }


@app.get("/workers/{worker_id}")
async def get_worker(worker_id: str):
    """
    Get worker information.

    Returns 404 if worker not found.
    """
    worker = registry.get_worker(worker_id)

    if worker is None:
        raise HTTPException(status_code=404, detail="Worker not found")

    return worker.to_dict()


@app.post("/workers/assign-shard")
async def assign_shard(assignment: ShardAssignment):
    """
    Assign parameter shard to worker.

    Returns 200 if successful, 404 if worker not found.
    """
    success = registry.assign_shard(
        worker_id=assignment.worker_id,
        shard_start=assignment.shard_start,
        shard_end=assignment.shard_end
    )

    if not success:
        raise HTTPException(status_code=404, detail="Worker not found")

    return {
        "message": "Shard assigned successfully",
        "worker_id": assignment.worker_id,
        "shard": f"[{assignment.shard_start}:{assignment.shard_end}]"
    }


@app.post("/latency/report")
async def report_latency(latency: LatencyReport):
    """
    Report latency measurement between two workers.

    Used for regional clustering.
    """
    cluster_manager.update_latency(
        worker_a=latency.worker_a,
        worker_b=latency.worker_b,
        latency_ms=latency.latency_ms
    )

    return {"message": "Latency recorded"}


@app.post("/clusters/compute")
async def compute_clusters():
    """
    Compute regional clusters based on latency measurements.

    Assigns workers to clusters and updates their region.
    """
    workers = registry.get_online_workers()

    if not workers:
        stats = cluster_manager.get_cluster_stats([])
        return {
            "message": "No online workers to cluster",
            "stats": stats
        }

    # Compute clusters
    clusters = cluster_manager.compute_clusters(workers)

    # Assign regions to workers
    for cluster in clusters:
        for worker_id in cluster.worker_ids:
            registry.assign_region(worker_id, cluster.region)

    # Get stats
    stats = cluster_manager.get_cluster_stats(clusters)

    # Broadcast clustering update
    await ws_manager.broadcast({
        'event': 'clusters_updated',
        'num_clusters': len(clusters),
        'total_workers': len(workers)
    })

    return {
        "message": "Clusters computed successfully",
        "stats": stats
    }


@app.get("/clusters")
async def get_clusters():
    """
    Get current cluster assignments.

    Returns workers grouped by region.
    """
    regions = registry.get_workers_by_region()

    clusters = []
    for region, workers in regions.items():
        clusters.append({
            'region': region,
            'workers': [w.worker_id for w in workers],
            'size': len(workers)
        })

    return {
        "clusters": clusters,
        "count": len(clusters)
    }


@app.post("/metrics/report")
async def report_metric(metric: MetricReport):
    """
    Report single training metric from worker.

    Deprecated: Use /metrics/report-batch for better performance.
    """
    success = db.record_metric(
        worker_id=metric.worker_id,
        global_step=metric.global_step,
        loss=metric.loss,
        throughput=metric.throughput,
        memory_usage_gb=metric.memory_usage_gb
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to record metric")

    # Broadcast metric update
    await ws_manager.broadcast({
        'event': 'metric_update',
        'worker_id': metric.worker_id,
        'global_step': metric.global_step,
        'loss': metric.loss
    })

    return {"message": "Metric recorded"}


@app.post("/metrics/report-batch")
async def report_metrics_batch(batch: MetricBatchReport):
    """
    Report multiple training metrics from worker in a single request.

    This is more efficient than multiple individual reports.
    """
    success = db.record_metrics_batch(
        worker_id=batch.worker_id,
        metrics=batch.metrics
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to record metrics")

    # Broadcast batch metric update
    if batch.metrics:
        latest_metric = batch.metrics[-1]
        await ws_manager.broadcast({
            'event': 'metrics_batch_update',
            'worker_id': batch.worker_id,
            'count': len(batch.metrics),
            'latest_step': latest_metric.get('global_step'),
            'latest_loss': latest_metric.get('loss')
        })

    return {
        "message": f"{len(batch.metrics)} metrics recorded",
        "count": len(batch.metrics)
    }


@app.get("/metrics")
async def get_metrics(limit: int = 100):
    """
    Get recent training metrics.

    Query params:
    - limit: Maximum number of metrics to return (default: 100)
    """
    metrics = db.get_recent_metrics(limit=limit)

    return {
        "metrics": metrics,
        "count": len(metrics)
    }


@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time events.

    Events:
    - worker_joined: New worker registered
    - worker_left: Worker deregistered
    - worker_offline: Worker marked offline (stale heartbeat)
    - clusters_updated: Cluster assignments updated
    - metric_update: New training metric
    """
    await ws_manager.connect(websocket)

    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for ping/pong
            await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# Development server

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Run the coordinator server.

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8000)
    """
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
