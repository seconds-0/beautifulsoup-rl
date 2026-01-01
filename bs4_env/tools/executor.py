from __future__ import annotations

"""Code execution infrastructure for BeautifulSoup RL environment.

This module provides executors for running Python code in isolated environments.
Two implementations are provided:
- LocalSubprocessExecutor: For local development and testing
- PrimeSandboxExecutor: For production use with Prime's sandbox infrastructure
"""

import contextlib
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExecResult:
    """Result of code execution.

    Attributes:
        stdout: Standard output from the code.
        stderr: Standard error from the code.
        exit_code: Exit code (0 for success).
        runtime_ms: Execution time in milliseconds.
        timed_out: Whether execution was terminated due to timeout.
        error: Any error message from the executor itself.
    """

    stdout: str
    stderr: str
    exit_code: int
    runtime_ms: int = 0
    timed_out: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "runtime_ms": self.runtime_ms,
            "timed_out": self.timed_out,
            "error": self.error,
        }


class Executor(ABC):
    """Abstract base class for code executors."""

    @abstractmethod
    def run(
        self,
        code: str,
        globals_dict: dict[str, Any],
        timeout_s: float = 30.0,
    ) -> ExecResult:
        """Execute Python code with injected globals.

        Args:
            code: Python code to execute.
            globals_dict: Dictionary of global variables to inject.
            timeout_s: Maximum execution time in seconds.

        Returns:
            ExecResult with stdout, stderr, exit_code, etc.
        """
        pass


class LocalSubprocessExecutor(Executor):
    """Execute code in a local subprocess.

    This executor is suitable for development and testing. It provides
    basic isolation via subprocess but does NOT provide security isolation.
    Do not use this for untrusted code in production.
    """

    def __init__(
        self,
        python_path: str | None = None,
        max_output_chars: int = 10000,
    ):
        """Initialize the executor.

        Args:
            python_path: Path to Python interpreter. If None, uses sys.executable.
            max_output_chars: Maximum characters to capture from stdout/stderr.
        """
        self.python_path = python_path or sys.executable
        self.max_output_chars = max_output_chars

    def run(
        self,
        code: str,
        globals_dict: dict[str, Any],
        timeout_s: float = 30.0,
    ) -> ExecResult:
        """Execute code in a subprocess.

        Args:
            code: Python code to execute.
            globals_dict: Global variables to inject (HTML, QUERY, CONSTRAINTS).
            timeout_s: Maximum execution time.

        Returns:
            ExecResult with execution results.
        """
        start_time = time.time()

        # Build the runner script
        from bs4_env.tools.harness import build_runner_script

        runner_code = build_runner_script(code, globals_dict)

        # Write to temp file (more reliable than stdin for complex code)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(runner_code)
            script_path = f.name

        try:
            # Run the script
            # Inherit current Python's environment for package access
            # Note: We DON'T change HOME as that breaks user site-packages lookup
            import os

            env = os.environ.copy()

            result = subprocess.run(
                [self.python_path, script_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )

            runtime_ms = int((time.time() - start_time) * 1000)

            return ExecResult(
                stdout=result.stdout[: self.max_output_chars],
                stderr=result.stderr[: self.max_output_chars],
                exit_code=result.returncode,
                runtime_ms=runtime_ms,
                timed_out=False,
            )

        except subprocess.TimeoutExpired:
            runtime_ms = int((time.time() - start_time) * 1000)
            return ExecResult(
                stdout="",
                stderr=f"Execution timed out after {timeout_s} seconds",
                exit_code=-1,
                runtime_ms=runtime_ms,
                timed_out=True,
            )

        except Exception as e:
            runtime_ms = int((time.time() - start_time) * 1000)
            return ExecResult(
                stdout="",
                stderr="",
                exit_code=-1,
                runtime_ms=runtime_ms,
                error=f"Executor error: {str(e)}",
            )

        finally:
            # Clean up temp file
            with contextlib.suppress(Exception):
                Path(script_path).unlink()


class PrimeSandboxExecutor(Executor):
    """Execute code in Prime's cloud sandbox infrastructure.

    This executor provides proper security isolation using Prime Intellect's
    sandbox infrastructure. Required for production use and bounty submission.

    Usage:
        # As context manager (recommended)
        with PrimeSandboxExecutor() as executor:
            result = executor.run(code, globals_dict)

        # Manual lifecycle
        executor = PrimeSandboxExecutor()
        try:
            result = executor.run(code, globals_dict)
        finally:
            executor.close()

    Note:
        - Requires `prime-sandboxes` package: pip install prime-sandboxes
        - API key from PRIME_API_KEY env var or passed to constructor
        - Network access is disabled during code execution for security
    """

    def __init__(
        self,
        api_key: str | None = None,
        docker_image: str = "python:3.11-slim",
        cpu_cores: int = 1,
        memory_gb: int = 2,
        max_output_chars: int = 10000,
        timeout_minutes: int = 30,
    ):
        """Initialize the executor.

        Args:
            api_key: Prime API key. If None, reads from PRIME_API_KEY env var.
            docker_image: Docker image for sandbox. Must have Python installed.
            cpu_cores: Number of CPU cores to allocate.
            memory_gb: Memory allocation in GB.
            max_output_chars: Maximum characters to capture from stdout/stderr.
            timeout_minutes: Sandbox lifecycle timeout in minutes.
        """
        self.docker_image = docker_image
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.max_output_chars = max_output_chars
        self.timeout_minutes = timeout_minutes
        self._sandbox_id: str | None = None
        self._deps_installed: bool = False

        # Lazy initialization - defer import until needed
        self._api_key = api_key
        self._sandbox_client = None

    def _get_sandbox_client(self):
        """Get or create the sandbox client (lazy initialization)."""
        if self._sandbox_client is None:
            try:
                from prime_sandboxes import APIClient, SandboxClient
            except ImportError:
                raise ImportError(
                    "prime-sandboxes package not installed. "
                    "Install with: pip install beautiful-soup-env[prime]"
                ) from None

            api_client = APIClient(api_key=self._api_key)
            self._sandbox_client = SandboxClient(api_client)
        return self._sandbox_client

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, cleanup sandbox."""
        self.close()
        return False

    def close(self):
        """Cleanup sandbox resources.

        Safe to call multiple times. Best-effort cleanup - exceptions
        during deletion are suppressed.
        """
        if self._sandbox_id is not None:
            try:
                self._get_sandbox_client().delete(self._sandbox_id)
            except Exception:
                pass  # Best effort cleanup
            self._sandbox_id = None
            self._deps_installed = False

    def _ensure_sandbox(self) -> str:
        """Create sandbox if not exists, return sandbox ID.

        Returns:
            The sandbox ID string.

        Raises:
            ImportError: If prime-sandboxes is not installed.
        """
        import uuid

        if self._sandbox_id is None:
            try:
                from prime_sandboxes import CreateSandboxRequest
            except ImportError:
                raise ImportError(
                    "prime-sandboxes package not installed. "
                    "Install with: pip install beautiful-soup-env[prime]"
                ) from None

            client = self._get_sandbox_client()
            request = CreateSandboxRequest(
                name=f"bs4-env-{uuid.uuid4().hex[:8]}",
                docker_image=self.docker_image,
                cpu_cores=self.cpu_cores,
                memory_gb=self.memory_gb,
                network_access=False,  # Security: no network during execution
                timeout_minutes=self.timeout_minutes,
            )
            sandbox = client.create(request)
            client.wait_for_creation(sandbox.id)
            self._sandbox_id = sandbox.id

        return self._sandbox_id

    def run(
        self,
        code: str,
        globals_dict: dict[str, Any],
        timeout_s: float = 30.0,
    ) -> ExecResult:
        """Execute code in Prime's sandbox.

        Args:
            code: Python code to execute.
            globals_dict: Global variables to inject (HTML, QUERY, CONSTRAINTS).
            timeout_s: Maximum execution time in seconds.

        Returns:
            ExecResult with execution results.
        """
        import uuid

        start_time = time.time()

        try:
            sandbox_id = self._ensure_sandbox()
            client = self._get_sandbox_client()

            # Build the runner script
            from bs4_env.tools.harness import build_runner_script

            script = build_runner_script(code, globals_dict)

            # Upload script to sandbox
            script_path = f"/tmp/run_{uuid.uuid4().hex[:8]}.py"

            # Write script to temp file then upload
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script)
                local_path = f.name

            try:
                client.upload_file(sandbox_id, script_path, local_path)
            finally:
                with contextlib.suppress(Exception):
                    Path(local_path).unlink()

            # Execute with timeout using shell timeout command
            # Exit code 124 means timeout killed the process
            timeout_int = int(timeout_s)
            result = client.execute_command(
                sandbox_id,
                f"timeout {timeout_int} python {script_path}",
                timeout=timeout_int + 5,  # Buffer for command overhead
            )

            runtime_ms = int((time.time() - start_time) * 1000)

            # Exit code 124 means the timeout command killed the process
            timed_out = result.exit_code == 124

            return ExecResult(
                stdout=(result.stdout or "")[: self.max_output_chars],
                stderr=(result.stderr or "")[: self.max_output_chars],
                exit_code=result.exit_code if not timed_out else -1,
                runtime_ms=runtime_ms,
                timed_out=timed_out,
            )

        except ImportError:
            # Re-raise import errors for clear messaging
            raise

        except Exception as e:
            runtime_ms = int((time.time() - start_time) * 1000)
            return ExecResult(
                stdout="",
                stderr=str(e)[: self.max_output_chars],
                exit_code=-1,
                runtime_ms=runtime_ms,
                timed_out=False,
                error=f"Sandbox execution error: {e}",
            )


class PooledSubprocessExecutor(Executor):
    """Persistent worker pool executor for high-throughput training.

    Uses a multiprocessing pool to reduce subprocess spawn overhead during
    training. Workers persist across calls, significantly improving throughput
    for batch processing.

    Usage:
        with PooledSubprocessExecutor(num_workers=4) as executor:
            for code, globals_dict in batch:
                result = executor.run(code, globals_dict)

    Note:
        Must be used as a context manager. Workers are terminated on exit.
    """

    def __init__(
        self,
        num_workers: int = 4,
        max_output_chars: int = 10000,
    ):
        """Initialize the pooled executor.

        Args:
            num_workers: Number of worker processes in the pool.
            max_output_chars: Maximum characters to capture from stdout/stderr.
        """
        self.num_workers = num_workers
        self.max_output_chars = max_output_chars
        self._pool = None

    def __enter__(self):
        """Enter context manager, create worker pool."""
        from multiprocessing import Pool

        self._pool = Pool(self.num_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, terminate workers."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
        return False

    def run(
        self,
        code: str,
        globals_dict: dict[str, Any],
        timeout_s: float = 30.0,
    ) -> ExecResult:
        """Execute code in a worker from the pool.

        Args:
            code: Python code to execute.
            globals_dict: Global variables to inject.
            timeout_s: Maximum execution time in seconds.

        Returns:
            ExecResult with execution results.

        Raises:
            RuntimeError: If not used as context manager.
        """
        from multiprocessing import TimeoutError as MPTimeoutError

        if self._pool is None:
            raise RuntimeError(
                "PooledSubprocessExecutor must be used as a context manager. "
                "Example: with PooledSubprocessExecutor() as executor: ..."
            )

        start_time = time.time()

        try:
            # Submit to pool and wait with timeout
            async_result = self._pool.apply_async(
                _execute_in_worker,
                (code, globals_dict, self.max_output_chars),
            )
            result = async_result.get(timeout=timeout_s)
            return result

        except MPTimeoutError:
            runtime_ms = int((time.time() - start_time) * 1000)
            return ExecResult(
                stdout="",
                stderr=f"Execution timed out after {timeout_s} seconds",
                exit_code=-1,
                runtime_ms=runtime_ms,
                timed_out=True,
            )

        except Exception as e:
            runtime_ms = int((time.time() - start_time) * 1000)
            return ExecResult(
                stdout="",
                stderr="",
                exit_code=-1,
                runtime_ms=runtime_ms,
                error=f"Pool execution error: {e}",
            )


def _execute_in_worker(
    code: str,
    globals_dict: dict[str, Any],
    max_output_chars: int,
) -> ExecResult:
    """Worker function for PooledSubprocessExecutor.

    This function runs in a separate process from the pool.
    It uses exec() to run the code in the worker's namespace.

    Args:
        code: Python code to execute.
        globals_dict: Global variables to inject.
        max_output_chars: Maximum output characters to capture.

    Returns:
        ExecResult with execution results.
    """
    import io
    import sys
    import traceback

    start_time = time.time()

    # Capture stdout/stderr
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    exit_code = 0

    try:
        # Build execution namespace with injected globals
        namespace = dict(globals_dict)

        # Execute the code
        exec(code, namespace)

    except Exception:
        exit_code = 1
        traceback.print_exc()

    finally:
        stdout = sys.stdout.getvalue()[:max_output_chars]
        stderr = sys.stderr.getvalue()[:max_output_chars]
        sys.stdout, sys.stderr = old_stdout, old_stderr

    runtime_ms = int((time.time() - start_time) * 1000)

    return ExecResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        runtime_ms=runtime_ms,
        timed_out=False,
    )


def get_executor(backend: str = "local", **kwargs) -> Executor:
    """Factory function to get an executor by backend name.

    Args:
        backend: Either "local", "prime", or "pooled".
        **kwargs: Additional arguments passed to executor constructor.

    Returns:
        An Executor instance.

    Raises:
        ValueError: If backend is unknown.
    """
    if backend == "local":
        return LocalSubprocessExecutor(**kwargs)
    elif backend == "prime":
        return PrimeSandboxExecutor(**kwargs)
    elif backend == "pooled":
        return PooledSubprocessExecutor(**kwargs)
    else:
        raise ValueError(f"Unknown executor backend: {backend}")
