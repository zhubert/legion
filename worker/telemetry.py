"""
Telemetry reporter for collecting and reporting training metrics.

Collects metrics during training and periodically reports them to the coordinator.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from worker.coordinator_client import CoordinatorClient


logger = logging.getLogger(__name__)


@dataclass
class MetricRecord:
    """Single metric record."""
    step: int
    loss: Optional[float] = None
    throughput: Optional[float] = None
    memory_usage_gb: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'step': self.step,
            'loss': self.loss,
            'throughput': self.throughput,
            'memory_usage_gb': self.memory_usage_gb,
            'timestamp': self.timestamp.isoformat()
        }


class TelemetryReporter:
    """
    Collects and reports training metrics to coordinator.

    Buffers metrics locally and reports them periodically or when buffer is full.
    """

    def __init__(
        self,
        coordinator_client: CoordinatorClient,
        report_interval_steps: int = 10,
        buffer_size: int = 100,
        enabled: bool = True
    ):
        """
        Initialize telemetry reporter.

        Args:
            coordinator_client: Client for coordinator communication
            report_interval_steps: Report every N steps
            buffer_size: Maximum buffered metrics before forcing report
            enabled: Whether telemetry is enabled
        """
        self.coordinator_client = coordinator_client
        self.report_interval_steps = report_interval_steps
        self.buffer_size = buffer_size
        self.enabled = enabled

        # Metric buffer
        self._buffer: deque = deque(maxlen=buffer_size)
        self._last_reported_step = -1

        # Statistics
        self._total_recorded = 0
        self._total_reported = 0
        self._failed_reports = 0

    def record_step(
        self,
        step: int,
        loss: Optional[float] = None,
        throughput: Optional[float] = None,
        memory_usage_gb: Optional[float] = None,
        **extra_metrics
    ):
        """
        Record metrics for a training step.

        Args:
            step: Training step number
            loss: Loss value
            throughput: Throughput in samples/sec
            memory_usage_gb: Memory usage in GB
            **extra_metrics: Additional metrics (ignored for now)
        """
        if not self.enabled:
            return

        record = MetricRecord(
            step=step,
            loss=loss,
            throughput=throughput,
            memory_usage_gb=memory_usage_gb
        )

        self._buffer.append(record)
        self._total_recorded += 1

        loss_str = f"{loss:.4f}" if loss is not None else "None"
        throughput_str = f"{throughput:.2f}" if throughput is not None else "None"

        logger.debug(
            f"Recorded metric for step {step}: "
            f"loss={loss_str}, throughput={throughput_str}"
        )

    async def report(self, force: bool = False) -> bool:
        """
        Report buffered metrics to coordinator in a single batch request.

        Args:
            force: Force reporting even if interval not reached

        Returns:
            True if report successful, False otherwise
        """
        if not self.enabled:
            return True

        if not self._buffer:
            return True

        # Check if we should report
        latest_step = self._buffer[-1].step

        should_report = (
            force or
            len(self._buffer) >= self.buffer_size or
            (latest_step - self._last_reported_step >= self.report_interval_steps)
        )

        if not should_report:
            return True

        # Convert buffer to list of metric dicts for batch reporting
        metrics_to_report = []
        while self._buffer:
            record = self._buffer.popleft()
            metrics_to_report.append({
                'global_step': record.step,
                'loss': record.loss,
                'throughput': record.throughput,
                'memory_usage_gb': record.memory_usage_gb
            })

        try:
            # Send all metrics in a single batch request
            success = await self.coordinator_client.report_metrics_batch(
                metrics=metrics_to_report
            )

            if success:
                self._total_reported += len(metrics_to_report)
                self._last_reported_step = metrics_to_report[-1]['global_step']

                logger.info(
                    f"Reported {len(metrics_to_report)} metrics in batch "
                    f"(steps {metrics_to_report[0]['global_step']}-{metrics_to_report[-1]['global_step']})"
                )
                return True
            else:
                # Put metrics back in buffer if failed
                self._failed_reports += len(metrics_to_report)
                for metric in reversed(metrics_to_report):
                    if len(self._buffer) < self.buffer_size:
                        record = MetricRecord(
                            step=metric['global_step'],
                            loss=metric.get('loss'),
                            throughput=metric.get('throughput'),
                            memory_usage_gb=metric.get('memory_usage_gb')
                        )
                        self._buffer.appendleft(record)

                logger.warning(f"Failed to report {len(metrics_to_report)} metrics")
                return False

        except Exception as e:
            logger.error(f"Error reporting metrics batch: {e}")
            self._failed_reports += len(metrics_to_report)

            # Put metrics back in buffer
            for metric in reversed(metrics_to_report):
                if len(self._buffer) < self.buffer_size:
                    record = MetricRecord(
                        step=metric['global_step'],
                        loss=metric.get('loss'),
                        throughput=metric.get('throughput'),
                        memory_usage_gb=metric.get('memory_usage_gb')
                    )
                    self._buffer.appendleft(record)

            return False

    async def report_step_async(
        self,
        step: int,
        loss: Optional[float] = None,
        throughput: Optional[float] = None,
        memory_usage_gb: Optional[float] = None
    ):
        """
        Record and potentially report metrics for a step (async).

        This is a convenience method that records the metric and
        automatically reports if the interval is reached.

        Args:
            step: Training step number
            loss: Loss value
            throughput: Throughput in samples/sec
            memory_usage_gb: Memory usage in GB
        """
        self.record_step(
            step=step,
            loss=loss,
            throughput=throughput,
            memory_usage_gb=memory_usage_gb
        )

        await self.report()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of collected metrics.

        Returns:
            Dictionary with metric statistics
        """
        if not self._buffer:
            return {
                'count': 0,
                'latest_step': self._last_reported_step
            }

        # Calculate statistics from buffer
        losses = [r.loss for r in self._buffer if r.loss is not None]
        throughputs = [r.throughput for r in self._buffer if r.throughput is not None]
        memory_usages = [r.memory_usage_gb for r in self._buffer if r.memory_usage_gb is not None]

        summary = {
            'count': len(self._buffer),
            'latest_step': self._buffer[-1].step,
            'oldest_step': self._buffer[0].step,
        }

        if losses:
            summary['loss'] = {
                'mean': sum(losses) / len(losses),
                'min': min(losses),
                'max': max(losses),
                'latest': losses[-1]
            }

        if throughputs:
            summary['throughput'] = {
                'mean': sum(throughputs) / len(throughputs),
                'min': min(throughputs),
                'max': max(throughputs),
                'latest': throughputs[-1]
            }

        if memory_usages:
            summary['memory_usage_gb'] = {
                'mean': sum(memory_usages) / len(memory_usages),
                'min': min(memory_usages),
                'max': max(memory_usages),
                'latest': memory_usages[-1]
            }

        return summary

    def get_status(self) -> Dict[str, Any]:
        """
        Get telemetry reporter status.

        Returns:
            Status dictionary with statistics
        """
        return {
            'enabled': self.enabled,
            'buffered_metrics': len(self._buffer),
            'total_recorded': self._total_recorded,
            'total_reported': self._total_reported,
            'failed_reports': self._failed_reports,
            'last_reported_step': self._last_reported_step,
            'report_interval_steps': self.report_interval_steps,
            'buffer_size': self.buffer_size
        }

    def clear_buffer(self):
        """Clear metric buffer."""
        self._buffer.clear()
        logger.info("Metric buffer cleared")

    def is_enabled(self) -> bool:
        """
        Check if telemetry is enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self.enabled

    def enable(self):
        """Enable telemetry."""
        self.enabled = True
        logger.info("Telemetry enabled")

    def disable(self):
        """Disable telemetry."""
        self.enabled = False
        logger.info("Telemetry disabled")
