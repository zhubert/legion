"""
Service orchestrator for Legion distributed training.

Starts and manages all required services with unified, color-coded output:
- Coordinator server
- Worker 1
- Worker 2
- Checkpoint assembler service

Usage:
    # Start all services with console output
    python scripts/start_services.py

    # Also write to log files
    python scripts/start_services.py --logs-dir logs

    # Custom number of workers
    python scripts/start_services.py --workers 3

    # Skip assembler service
    python scripts/start_services.py --no-assembler
"""

import asyncio
import argparse
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ANSI color codes
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


class ServiceManager:
    """Manages multiple service processes with unified output."""

    def __init__(self, logs_dir: Optional[Path] = None):
        self.processes = {}
        self.tasks = {}
        self.logs_dir = logs_dir
        self.shutdown_event = asyncio.Event()

        if self.logs_dir:
            self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _format_line(self, service: str, line: str, color: str) -> str:
        """Format a log line with color, timestamp, and service prefix."""
        ts = self._timestamp()
        # Remove trailing newline if present
        line = line.rstrip('\n')
        return f"{Colors.DIM}{ts}{Colors.RESET} {color}{service:12}{Colors.RESET} │ {line}"

    async def _stream_output(self, service: str, stream, color: str, log_file=None):
        """Stream output from a process with formatting."""
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break

                line = line.decode('utf-8', errors='replace')
                formatted = self._format_line(service, line, color)
                print(formatted, flush=True)

                if log_file:
                    log_file.write(f"{self._timestamp()} {service} │ {line}")
                    log_file.flush()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            error_line = f"Error streaming {service}: {e}"
            print(self._format_line(service, error_line, Colors.RED), flush=True)

    async def start_service(
        self,
        service: str,
        command: list[str],
        color: str,
        cwd: Optional[Path] = None,
        env: Optional[dict] = None
    ):
        """Start a service process and stream its output."""
        print(f"{Colors.BOLD}{color}▶ Starting {service}...{Colors.RESET}")

        log_file = None
        if self.logs_dir:
            log_path = self.logs_dir / f"{service.replace(' ', '_').lower()}.log"
            log_file = open(log_path, 'a')
            log_file.write(f"\n{'='*80}\n")
            log_file.write(f"Started at {datetime.now().isoformat()}\n")
            log_file.write(f"{'='*80}\n")

        try:
            # Start the process
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
                env=env
            )

            self.processes[service] = process

            # Stream stdout
            stdout_task = asyncio.create_task(
                self._stream_output(service, process.stdout, color, log_file)
            )
            self.tasks[service] = stdout_task

            print(f"{Colors.BOLD}{color}✓ {service} started (PID: {process.pid}){Colors.RESET}")

            # Wait for process to complete
            return_code = await process.wait()

            if return_code != 0 and not self.shutdown_event.is_set():
                print(f"{Colors.BOLD}{Colors.RED}✗ {service} exited with code {return_code}{Colors.RESET}")
            else:
                print(f"{Colors.BOLD}{color}■ {service} stopped{Colors.RESET}")

        except Exception as e:
            print(f"{Colors.BOLD}{Colors.RED}✗ {service} failed to start: {e}{Colors.RESET}")
        finally:
            if log_file:
                log_file.close()

    async def stop_all(self):
        """Stop all running services gracefully."""
        print(f"\n{Colors.BOLD}{Colors.YELLOW}⚠ Shutting down all services...{Colors.RESET}")
        self.shutdown_event.set()

        # Cancel output streaming tasks
        for service, task in self.tasks.items():
            task.cancel()

        # Send SIGTERM to all processes
        for service, process in self.processes.items():
            if process.returncode is None:
                print(f"{Colors.DIM}Stopping {service}...{Colors.RESET}")
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass

        # Wait for processes to terminate (with timeout)
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *[task for task in self.tasks.values()],
                    return_exceptions=True
                ),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            # Force kill if they don't terminate
            for service, process in self.processes.items():
                if process.returncode is None:
                    print(f"{Colors.DIM}Force killing {service}...{Colors.RESET}")
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass

        print(f"{Colors.BOLD}{Colors.GREEN}✓ All services stopped{Colors.RESET}")


async def main():
    parser = argparse.ArgumentParser(
        description="Start all Legion services with unified output"
    )
    parser.add_argument(
        '--logs-dir',
        type=Path,
        default=Path('logs'),
        help='Directory to write log files (default: logs/)'
    )
    parser.add_argument(
        '--no-logs',
        action='store_true',
        help='Disable log file writing (console output only)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='Number of worker clients to start (default: 2)'
    )
    parser.add_argument(
        '--no-assembler',
        action='store_true',
        help='Skip starting the checkpoint assembler service'
    )
    args = parser.parse_args()

    # Disable logs if requested
    if args.no_logs:
        args.logs_dir = None

    # Validate worker count
    if args.workers < 1:
        print(f"{Colors.RED}Error: Must have at least 1 worker{Colors.RESET}")
        return 1

    manager = ServiceManager(logs_dir=args.logs_dir)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        loop.create_task(manager.stop_all())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}Legion Service Orchestrator{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")

    if args.logs_dir:
        print(f"Logs: {args.logs_dir.absolute()}")
    print(f"Workers: {args.workers}")
    print(f"Assembler: {'disabled' if args.no_assembler else 'enabled'}")
    print()

    # Get project root
    project_root = Path(__file__).parent.parent

    # Clear checkpoints directory for fresh run
    checkpoints_dir = project_root / "checkpoints"
    if checkpoints_dir.exists():
        import shutil
        print(f"{Colors.DIM}Clearing checkpoints directory...{Colors.RESET}")
        shutil.rmtree(checkpoints_dir)
        print(f"{Colors.DIM}✓ Checkpoints cleared{Colors.RESET}")
    checkpoints_dir.mkdir(exist_ok=True)

    try:
        # Start all services concurrently
        services = []

        # 1. Coordinator server
        services.append(
            manager.start_service(
                "coordinator",
                ["python", "-m", "coordinator.server"],
                Colors.BLUE,
                cwd=project_root
            )
        )

        # 2. Workers (dynamic count)
        worker_colors = [Colors.GREEN, Colors.YELLOW, Colors.CYAN, Colors.MAGENTA]
        base_port = 50051
        for i in range(args.workers):
            color = worker_colors[i % len(worker_colors)]
            port = base_port + i
            # Set unique port via environment variable
            worker_env = {
                **dict(os.environ),
                'LEGION_WORKER_PORT': str(port)
            }
            services.append(
                manager.start_service(
                    f"worker-{i+1}",
                    ["python", "-m", "worker.client"],
                    color,
                    cwd=project_root,
                    env=worker_env
                )
            )

        # 3. Checkpoint assembler service (optional)
        if not args.no_assembler:
            services.append(
                manager.start_service(
                    "assembler",
                    ["python", "-m", "worker.assembler_service"],
                    Colors.MAGENTA,
                    cwd=project_root
                )
            )

        # Wait for all services
        await asyncio.gather(*services, return_exceptions=True)

    except KeyboardInterrupt:
        pass
    finally:
        await manager.stop_all()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
