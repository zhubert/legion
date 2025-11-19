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
from coordinator.version_manager import VersionManager
from coordinator.training_config import TrainingConfig, TrainingConfigManager


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


class VersionUpdate(BaseModel):
    """Worker version update."""
    worker_id: str = Field(..., description="Worker identifier")
    version: int = Field(..., description="Current training step version", ge=0)
    is_healthy: bool = Field(default=True, description="Worker health status")


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
training_config_manager: Optional[TrainingConfigManager] = None
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
    global db, registry, cluster_manager, version_manager, training_config_manager

    # Startup
    logger.info("Starting coordinator server...")
    db = Database("coordinator.db")
    registry = WorkerRegistry(db, heartbeat_timeout=90)
    cluster_manager = ClusterManager(latency_threshold_ms=50.0)
    version_manager = VersionManager(staleness_bound=5)
    training_config_manager = TrainingConfigManager()

    # Clean up stale workers from previous runs (> 5 minutes old)
    from datetime import datetime, timedelta
    stale_workers = []
    for worker in db.get_all_workers():
        if worker['last_heartbeat']:
            try:
                last_heartbeat = datetime.fromisoformat(worker['last_heartbeat'])
                if datetime.now() - last_heartbeat > timedelta(minutes=5):
                    stale_workers.append(worker['worker_id'])
            except:
                pass

    for worker_id in stale_workers:
        logger.info(f"Removing stale worker from previous run: {worker_id}")
        db.delete_worker(worker_id)

    if stale_workers:
        logger.info(f"Cleaned up {len(stale_workers)} stale workers")

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


@app.post("/training/barrier")
async def training_barrier(
    worker_id: str,
    step: str = "training_complete",
    global_step: Optional[int] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Distributed barrier synchronization for workers.

    Workers call this when they reach a barrier point (e.g., training complete).
    The coordinator tracks which workers have reached the barrier.

    Args:
    - worker_id: ID of worker reaching barrier
    - step: Barrier step name (default: "training_complete")
    - global_step: Training step number (used for checkpoint creation)

    Returns:
    - reached: Number of workers at this barrier
    - total: Total number of online workers
    - all_ready: True if all workers have reached the barrier
    """
    # Get all online workers
    active_workers = registry.get_online_workers()
    worker_ids = {w.worker_id for w in active_workers}
    worker_ids_list = sorted(list(worker_ids))  # For consistent ordering
    total_workers = len(worker_ids)

    # Use persistent barrier state from database
    barrier_id = f"{step}_{hash(frozenset(worker_ids_list))}"  # Unique ID per worker set

    # Get or create barrier
    barrier = db.get_barrier(barrier_id)
    if barrier is None:
        # Create new barrier
        db.create_or_update_barrier(
            barrier_id=barrier_id,
            step=step,
            worker_ids=worker_ids_list,
            arrived_workers=[worker_id],
            status="waiting"
        )
        arrived_workers = {worker_id}
        # Store global_step in barrier metadata if provided
        if global_step is not None:
            barrier_global_step = global_step
        else:
            barrier_global_step = None
    else:
        # Update existing barrier
        arrived_workers = set(barrier['arrived_workers'])
        arrived_workers.add(worker_id)
        db.create_or_update_barrier(
            barrier_id=barrier_id,
            step=step,
            worker_ids=worker_ids_list,
            arrived_workers=sorted(list(arrived_workers)),
            status="waiting" if len(arrived_workers) < total_workers else "complete"
        )
        # Use global_step from barrier if not provided
        barrier_global_step = global_step if global_step is not None else barrier.get('global_step')

    reached = len(arrived_workers)
    all_ready = reached >= total_workers

    logger.info(
        f"Barrier '{step}': worker {worker_id} reached "
        f"({reached}/{total_workers} workers)"
    )

    # Trigger checkpoint assembly when training_complete barrier is reached
    if all_ready and step == "training_complete" and barrier_global_step is not None:
        logger.info(
            f"Barrier '{step}' complete, all {total_workers} workers ready. "
            f"Triggering checkpoint assembly at step {barrier_global_step}"
        )

        # Create checkpoint metadata before broadcasting assembly request
        # Only create if it doesn't already exist (prevents overwriting with fewer workers)
        try:
            from pathlib import Path
            from worker.assembler import create_checkpoint_metadata

            metadata_path = Path("./checkpoints") / f"step_{barrier_global_step}" / "metadata.json"

            if not metadata_path.exists():
                # Build worker info for metadata using workers that reached the barrier
                # Use worker_ids_list which is the consistent ordering from the barrier
                worker_metadata = []
                for idx, worker_id in enumerate(worker_ids_list):
                    worker_metadata.append({
                        'rank': idx,
                        'worker_id': worker_id,
                        'shard_file': f"shard_rank_{idx}.pt"
                    })

                # Create metadata.json
                metadata_path_str = create_checkpoint_metadata(
                    checkpoint_dir="./checkpoints",
                    global_step=barrier_global_step,
                    workers=worker_metadata,
                    model_config=None  # TODO: Get from training config
                )
                logger.info(f"Created checkpoint metadata: {metadata_path_str} with {len(worker_metadata)} workers")
            else:
                logger.info(f"Checkpoint metadata already exists: {metadata_path}, skipping creation")

        except Exception as e:
            logger.error(f"Failed to create checkpoint metadata: {e}")
            # Continue with assembly anyway - assembler will fail with clear error

        # Broadcast checkpoint assembly request to assembler service
        await ws_manager.broadcast({
            'event': 'assemble_checkpoint',
            'global_step': barrier_global_step,
            'num_workers': total_workers,
            'worker_ids': worker_ids_list
        })

        logger.info(f"Checkpoint assembly request broadcast for step {barrier_global_step}")

        db.delete_barrier(barrier_id)
    elif all_ready:
        logger.info(f"Barrier '{step}' complete, all {total_workers} workers ready")
        db.delete_barrier(barrier_id)

    return {
        "reached": reached,
        "total": total_workers,
        "all_ready": all_ready,
        "waiting_for": list(worker_ids - arrived_workers) if not all_ready else [],
        "global_step": barrier_global_step if all_ready else None
    }


@app.get("/training/ready")
async def check_training_ready(min_workers: int = 2):
    """
    Check if enough workers are ready to start distributed training.

    This endpoint allows workers to synchronize and wait for sufficient
    peers before beginning training.

    Query params:
    - min_workers: Minimum number of online workers required (default: 2)

    Returns:
    - ready: Boolean indicating if training can start
    - active_workers: Number of currently online workers
    - min_workers: Required minimum
    - workers: List of active worker information
    """
    # Get all online workers
    active_workers = registry.get_online_workers()
    num_active = len(active_workers)

    # Sort workers by worker_id for consistent rank assignment
    workers_sorted = sorted(active_workers, key=lambda w: w.worker_id)

    # Build worker addresses for distributed training
    worker_info = [
        {
            'worker_id': w.worker_id,
            'ip_address': w.ip_address,
            'port': w.port,
            'rank': i  # Assign rank based on sorted order
        }
        for i, w in enumerate(workers_sorted)
    ]

    ready = num_active >= min_workers

    logger.info(
        f"Training readiness check: {num_active}/{min_workers} workers "
        f"(ready={ready})"
    )

    return {
        "ready": ready,
        "active_workers": num_active,
        "min_workers": min_workers,
        "workers": worker_info
    }


# Version Management Endpoints (for Async Parameter Server)

@app.post("/training/version/update")
async def update_worker_version(version_update: VersionUpdate):
    """
    Update a worker's current training version.

    Workers should call this endpoint after completing each training step
    to enable bounded staleness tracking.

    Returns:
    - global_version: Current global version (median of all workers)
    - worker_version: This worker's reported version
    - is_ahead: Whether worker is beyond staleness bound
    - backup_assignment: Worker ID to help (if ahead), or null
    """
    # Update version in version manager
    # Use provided is_healthy, but override if worker is offline in registry
    worker_info = registry.get_worker(version_update.worker_id)
    if worker_info and worker_info.status != "online":
        is_healthy = False
    else:
        is_healthy = version_update.is_healthy

    version_manager.update_worker_version(
        worker_id=version_update.worker_id,
        version=version_update.version,
        is_healthy=is_healthy
    )

    # Compute global version
    global_version = version_manager.get_global_version()

    # Check if worker is too far ahead
    is_ahead = version_manager.is_worker_too_far_ahead(version_update.worker_id)

    # Get backup assignment if ahead
    backup_assignment = None
    if is_ahead:
        backup_assignment = version_manager.get_backup_assignment(version_update.worker_id)

    logger.debug(
        f"Version update: {version_update.worker_id} → v{version_update.version} "
        f"(global=v{global_version}, ahead={is_ahead})"
    )

    return {
        "status": "success",
        "worker_id": version_update.worker_id,
        "version": version_update.version,
        "global_version": global_version,
        "worker_version": version_update.version,
        "is_ahead": is_ahead,
        "backup_assignment": backup_assignment
    }


@app.get("/training/version/global")
async def get_global_version():
    """
    Get the current global training version.

    Global version is computed as the median of all active workers,
    preventing one straggler from blocking the entire cluster.

    Returns:
    - global_version: Median version across active workers
    - num_workers: Number of workers contributing to global version
    - staleness_bound: Maximum allowed version gap
    """
    global_version = version_manager.get_global_version()
    num_workers = len([w for w in version_manager.get_all_workers() if w.is_healthy])

    return {
        "global_version": global_version,
        "num_workers": num_workers,
        "staleness_bound": version_manager.staleness_bound
    }


@app.get("/training/version/stats")
async def get_version_stats():
    """
    Get detailed version statistics across the cluster.

    Returns min, max, mean, median versions and staleness metrics.
    Useful for monitoring training progress and identifying stragglers.
    """
    stats = version_manager.get_staleness_stats()

    # Add progress rate
    stats["progress_rate_versions_per_sec"] = version_manager.get_version_progress_rate()

    return stats


@app.get("/training/version/workers")
async def get_worker_versions():
    """
    Get version information for all tracked workers.

    Returns list of workers with their current versions, timestamps,
    and health status.
    """
    workers = version_manager.get_all_workers()

    return {
        "workers": [
            {
                "worker_id": w.worker_id,
                "version": w.version,
                "timestamp": w.timestamp,
                "is_healthy": w.is_healthy
            }
            for w in workers
        ],
        "count": len(workers)
    }


@app.get("/training/version/slow-workers")
async def get_slow_workers():
    """
    Get list of workers below the global version (stragglers).

    These workers are candidates for receiving backup computation help.

    Returns:
    - slow_workers: List of worker IDs sorted by version (slowest first)
    - global_version: Current global version for reference
    """
    global_version = version_manager.get_global_version()
    slow_workers = version_manager.get_slow_workers(global_version)

    return {
        "slow_workers": slow_workers,
        "global_version": global_version,
        "count": len(slow_workers)
    }


@app.get("/training/version/ahead-workers")
async def get_ahead_workers():
    """
    Get list of workers ahead of global version + staleness_bound.

    These workers should perform work stealing instead of progressing.

    Returns:
    - ahead_workers: List of worker IDs that are too far ahead
    - global_version: Current global version
    - staleness_bound: Maximum allowed gap
    """
    global_version = version_manager.get_global_version()
    ahead_workers = version_manager.get_ahead_workers(global_version)

    return {
        "ahead_workers": ahead_workers,
        "global_version": global_version,
        "staleness_bound": version_manager.staleness_bound,
        "count": len(ahead_workers)
    }


@app.post("/training/version/assign-backups")
async def assign_backup_work():
    """
    Assign ahead workers to help slow workers (work stealing).

    Coordinator matches fastest workers (ahead of staleness bound)
    with slowest workers for backup gradient computation.

    Returns:
    - assignments: Dict mapping ahead_worker_id → slow_worker_id
    - count: Number of assignments made
    """
    assignments = version_manager.assign_work_stealing()

    return {
        "assignments": assignments,
        "count": len(assignments)
    }


# Checkpoint endpoints

class CheckpointCreateRequest(BaseModel):
    """Request to create a checkpoint."""
    global_step: int = Field(..., description="Training step to checkpoint", ge=0)
    checkpoint_dir: str = Field(default="./checkpoints", description="Base checkpoint directory")
    model_metadata: Optional[Dict[str, Any]] = Field(None, description="Optional model configuration metadata")
    assemble: bool = Field(default=True, description="Whether to assemble checkpoint after saving shards")


@app.post("/checkpoint/create")
async def create_checkpoint(request: CheckpointCreateRequest, background_tasks: BackgroundTasks):
    """
    Trigger checkpoint creation across all workers.

    This endpoint:
    1. Gets list of all online workers
    2. Broadcasts checkpoint request via WebSocket
    3. Creates metadata.json with worker info
    4. Optionally triggers assembler in background

    Returns checkpoint ID for tracking.
    """
    import uuid
    import json
    from pathlib import Path
    from worker.assembler import create_checkpoint_metadata

    # Get online workers
    workers = registry.get_workers()
    if not workers:
        raise HTTPException(status_code=400, detail="No workers online to checkpoint")

    # Generate checkpoint ID
    checkpoint_id = f"ckpt_{request.global_step}_{uuid.uuid4().hex[:8]}"

    # Prepare worker info for metadata
    worker_info = []
    for worker in workers:
        rank = workers.index(worker)  # Use position as rank
        worker_info.append({
            'rank': rank,
            'worker_id': worker.worker_id,
            'shard_file': f"shard_rank_{rank}.pt",
            'shard_start': worker.shard_start,
            'shard_end': worker.shard_end
        })

    # Save checkpoint metadata to database
    worker_shards = {
        w['worker_id']: {
            'rank': w['rank'],
            'shard_start': w['shard_start'],
            'shard_end': w['shard_end']
        }
        for w in worker_info
    }

    db.save_checkpoint_metadata(
        checkpoint_id=checkpoint_id,
        version=1,
        global_step=request.global_step,
        worker_shards=worker_shards,
        metadata=request.model_metadata
    )

    # Create checkpoint metadata file
    try:
        create_checkpoint_metadata(
            checkpoint_dir=request.checkpoint_dir,
            global_step=request.global_step,
            workers=worker_info,
            model_config=request.model_metadata
        )
    except Exception as e:
        logger.error(f"Failed to create checkpoint metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create checkpoint metadata: {e}")

    # Broadcast checkpoint request to workers
    await ws_manager.broadcast({
        'event': 'checkpoint_request',
        'checkpoint_id': checkpoint_id,
        'global_step': request.global_step,
        'checkpoint_dir': request.checkpoint_dir,
        'workers': worker_info
    })

    logger.info(
        f"Checkpoint request broadcast: step={request.global_step}, "
        f"workers={len(workers)}, checkpoint_id={checkpoint_id}"
    )

    # Update status to shards_saved (workers will save in background)
    db.update_checkpoint_status(checkpoint_id, "shards_saved")

    # Optionally trigger assembly in background
    if request.assemble:
        background_tasks.add_task(
            assemble_checkpoint_task,
            request.checkpoint_dir,
            request.global_step,
            checkpoint_id
        )

    return {
        "checkpoint_id": checkpoint_id,
        "global_step": request.global_step,
        "num_workers": len(workers),
        "status": "shards_saving",
        "will_assemble": request.assemble
    }


async def assemble_checkpoint_task(
    checkpoint_dir: str,
    global_step: int,
    checkpoint_id: str
):
    """
    Background task to assemble checkpoint from shards.

    Args:
        checkpoint_dir: Base checkpoint directory
        global_step: Training step
        checkpoint_id: Checkpoint ID for tracking
    """
    import asyncio
    from worker.assembler import CheckpointAssembler

    # Wait a bit for workers to save shards
    await asyncio.sleep(5)

    try:
        db.update_checkpoint_status(checkpoint_id, "assembling")

        assembler = CheckpointAssembler(checkpoint_dir)
        output_path = assembler.assemble_checkpoint(global_step)

        db.update_checkpoint_status(checkpoint_id, "assembled", output_path)

        # Broadcast completion
        await ws_manager.broadcast({
            'event': 'checkpoint_assembled',
            'checkpoint_id': checkpoint_id,
            'global_step': global_step,
            'output_path': output_path
        })

        logger.info(f"Checkpoint assembled: {output_path}")

    except Exception as e:
        logger.error(f"Failed to assemble checkpoint: {e}")
        db.update_checkpoint_status(checkpoint_id, "failed")

        await ws_manager.broadcast({
            'event': 'checkpoint_failed',
            'checkpoint_id': checkpoint_id,
            'global_step': global_step,
            'error': str(e)
        })


@app.get("/checkpoint/{global_step}/status")
async def get_checkpoint_status(global_step: int):
    """
    Get status of a checkpoint by global step.

    Returns checkpoint information including assembly status.
    """
    checkpoint = db.get_checkpoint_by_step(global_step)

    if not checkpoint:
        raise HTTPException(status_code=404, detail=f"Checkpoint not found for step {global_step}")

    return checkpoint


@app.get("/checkpoints")
async def list_checkpoints():
    """
    List all checkpoints.

    Returns list of checkpoint info sorted by global step (newest first).
    """
    checkpoints = db.list_checkpoints()
    return {
        "checkpoints": checkpoints,
        "count": len(checkpoints)
    }


@app.get("/training/config")
async def get_training_config():
    """
    Get current global training configuration.

    Returns the training configuration that all workers should follow.
    """
    return training_config_manager.to_dict()


@app.put("/training/config")
async def update_training_config(updates: Dict[str, Any]):
    """
    Update global training configuration.

    Args:
        updates: Dictionary of configuration fields to update

    Returns:
        Updated configuration or error if validation fails
    """
    errors = training_config_manager.update_config(updates)

    if errors:
        raise HTTPException(
            status_code=400,
            detail={"message": "Invalid configuration", "errors": errors}
        )

    # Broadcast configuration update to workers
    await ws_manager.broadcast({
        'event': 'training_config_updated',
        'config': training_config_manager.to_dict()
    })

    return {
        "message": "Training configuration updated",
        "config": training_config_manager.to_dict()
    }


@app.get("/training/config/worker/{worker_id}")
async def get_worker_training_config(worker_id: str):
    """
    Get training configuration assignment for a specific worker.

    This includes rank, world_size, and any worker-specific overrides
    (e.g., custom batch size based on hardware).

    Args:
        worker_id: Worker identifier

    Returns:
        Training configuration customized for this worker
    """
    # Get worker info to determine rank
    worker = registry.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")

    # Get online workers to determine rank and world_size
    workers = registry.get_online_workers()

    # Sort workers by worker_id to ensure consistent rank assignment
    workers = sorted(workers, key=lambda w: w.worker_id)

    # Find rank
    rank = next((i for i, w in enumerate(workers) if w.worker_id == worker_id), 0)
    world_size = len(workers)

    # Get configuration with worker-specific settings
    config = training_config_manager.get_worker_assignment(
        worker_id=worker_id,
        rank=rank,
        world_size=world_size
    )

    return {
        "worker_id": worker_id,
        "rank": rank,
        "world_size": world_size,
        "config": config
    }


@app.put("/training/config/worker/{worker_id}/batch_size")
async def set_worker_batch_size(worker_id: str, batch_size: int):
    """
    Set custom batch size for a worker based on its hardware capabilities.

    Args:
        worker_id: Worker identifier
        batch_size: Custom batch size for this worker

    Returns:
        Updated worker configuration
    """
    if batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size must be positive")

    training_config_manager.set_worker_batch_size(worker_id, batch_size)

    logger.info(f"Set batch size for worker {worker_id}: {batch_size}")

    return {
        "message": f"Batch size set for worker {worker_id}",
        "worker_id": worker_id,
        "batch_size": batch_size
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
    - checkpoint_request: Coordinator requests workers save checkpoint
    - checkpoint_assembled: Checkpoint assembly completed
    - checkpoint_failed: Checkpoint assembly failed
    - training_config_updated: Training configuration updated
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
