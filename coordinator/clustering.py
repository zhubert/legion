"""
Regional clustering for workers based on network latency.

Groups workers into regional clusters to minimize communication overhead.
"""

from typing import List, Dict, Set, Tuple
import logging
from dataclasses import dataclass

from coordinator.registry import WorkerInfo


logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    """Regional cluster of workers."""
    cluster_id: str
    region: str
    worker_ids: List[str]
    avg_latency: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses."""
        return {
            'cluster_id': self.cluster_id,
            'region': self.region,
            'worker_ids': self.worker_ids,
            'avg_latency': self.avg_latency,
            'size': len(self.worker_ids)
        }


class ClusterManager:
    """
    Manages regional clustering of workers.

    Uses latency measurements to group workers into efficient clusters.
    """

    def __init__(self, latency_threshold_ms: float = 50.0):
        """
        Initialize cluster manager.

        Args:
            latency_threshold_ms: Maximum latency within a cluster (default: 50ms)
        """
        self.latency_threshold_ms = latency_threshold_ms
        self.latency_matrix: Dict[Tuple[str, str], float] = {}

    def update_latency(self, worker_a: str, worker_b: str, latency_ms: float):
        """
        Update latency measurement between two workers.

        Args:
            worker_a: First worker ID
            worker_b: Second worker ID
            latency_ms: Measured latency in milliseconds
        """
        # Store both directions (symmetric)
        self.latency_matrix[(worker_a, worker_b)] = latency_ms
        self.latency_matrix[(worker_b, worker_a)] = latency_ms

        logger.debug(f"Latency {worker_a} <-> {worker_b}: {latency_ms:.2f}ms")

    def get_latency(self, worker_a: str, worker_b: str) -> float:
        """
        Get latency between two workers.

        Args:
            worker_a: First worker ID
            worker_b: Second worker ID

        Returns:
            Latency in milliseconds, or infinity if not measured
        """
        return self.latency_matrix.get((worker_a, worker_b), float('inf'))

    def compute_clusters(self, workers: List[WorkerInfo]) -> List[Cluster]:
        """
        Compute regional clusters using greedy clustering algorithm.

        Algorithm:
        1. Start with all workers unassigned
        2. Pick an unassigned worker as cluster seed
        3. Add all workers within latency threshold
        4. Repeat until all workers assigned

        Args:
            workers: List of workers to cluster

        Returns:
            List of clusters
        """
        if not workers:
            return []

        unassigned: Set[str] = {w.worker_id for w in workers}
        clusters: List[Cluster] = []
        cluster_counter = 0

        while unassigned:
            # Pick seed worker (first unassigned)
            seed_id = next(iter(unassigned))
            unassigned.remove(seed_id)

            # Start new cluster
            cluster_workers = [seed_id]

            # Add workers within latency threshold
            to_check = list(unassigned)
            for worker_id in to_check:
                # Check average latency to all workers in cluster
                latencies = [
                    self.get_latency(worker_id, cluster_worker)
                    for cluster_worker in cluster_workers
                ]
                avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

                if avg_latency <= self.latency_threshold_ms:
                    cluster_workers.append(worker_id)
                    unassigned.remove(worker_id)

            # Compute average intra-cluster latency
            if len(cluster_workers) > 1:
                total_latency = 0.0
                count = 0
                for i, w1 in enumerate(cluster_workers):
                    for w2 in cluster_workers[i+1:]:
                        total_latency += self.get_latency(w1, w2)
                        count += 1
                avg_latency = total_latency / count if count > 0 else 0.0
            else:
                avg_latency = 0.0

            # Create cluster
            cluster = Cluster(
                cluster_id=f"cluster_{cluster_counter}",
                region=f"region_{cluster_counter}",
                worker_ids=cluster_workers,
                avg_latency=avg_latency
            )
            clusters.append(cluster)
            cluster_counter += 1

            logger.info(
                f"Created {cluster.cluster_id} with {len(cluster_workers)} workers "
                f"(avg latency: {avg_latency:.2f}ms)"
            )

        return clusters

    def assign_clusters_simple(self, workers: List[WorkerInfo]) -> Dict[str, str]:
        """
        Simple cluster assignment without latency measurements.

        Assigns all workers to a single global cluster. Useful for testing
        or when latency measurements are not available.

        Args:
            workers: List of workers to assign

        Returns:
            Dictionary mapping worker_id to region
        """
        assignments = {}
        for worker in workers:
            assignments[worker.worker_id] = "global"

        logger.info(f"Assigned {len(workers)} workers to global cluster")
        return assignments

    def assign_clusters_by_latency(self, workers: List[WorkerInfo]) -> Dict[str, str]:
        """
        Assign workers to clusters based on latency measurements.

        Args:
            workers: List of workers to assign

        Returns:
            Dictionary mapping worker_id to region
        """
        clusters = self.compute_clusters(workers)

        assignments = {}
        for cluster in clusters:
            for worker_id in cluster.worker_ids:
                assignments[worker_id] = cluster.region

        logger.info(
            f"Assigned {len(workers)} workers to {len(clusters)} regional clusters"
        )

        return assignments

    def get_cluster_stats(self, clusters: List[Cluster]) -> Dict:
        """
        Get statistics about clusters.

        Args:
            clusters: List of clusters

        Returns:
            Statistics dictionary
        """
        if not clusters:
            return {
                'num_clusters': 0,
                'total_workers': 0,
                'avg_cluster_size': 0.0,
                'avg_intra_cluster_latency': 0.0
            }

        total_workers = sum(len(c.worker_ids) for c in clusters)
        avg_cluster_size = total_workers / len(clusters)
        avg_latency = sum(c.avg_latency for c in clusters) / len(clusters)

        return {
            'num_clusters': len(clusters),
            'total_workers': total_workers,
            'avg_cluster_size': avg_cluster_size,
            'avg_intra_cluster_latency': avg_latency,
            'clusters': [c.to_dict() for c in clusters]
        }
