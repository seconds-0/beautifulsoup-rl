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
    """Execute code in Prime's sandbox infrastructure.

    This executor provides proper security isolation and is required for
    production use and bounty submission.

    NOTE: This is a stub implementation. You must wire this to the actual
    Prime sandbox APIs for production use.
    """

    def __init__(
        self,
        network_access: bool = False,
        max_output_chars: int = 10000,
    ):
        """Initialize the executor.

        Args:
            network_access: Whether to allow network access. Should be False.
            max_output_chars: Maximum characters to capture.
        """
        self.network_access = network_access
        self.max_output_chars = max_output_chars

        # Check if we can import Prime/Verifiers
        self._prime_available = self._check_prime_available()

    def _check_prime_available(self) -> bool:
        """Check if Prime sandbox APIs are available."""
        try:
            # TODO: Replace with actual import check
            # import prime.sandbox
            return False
        except ImportError:
            return False

    def run(
        self,
        code: str,
        globals_dict: dict[str, Any],
        timeout_s: float = 30.0,
    ) -> ExecResult:
        """Execute code in Prime's sandbox.

        Args:
            code: Python code to execute.
            globals_dict: Global variables to inject.
            timeout_s: Maximum execution time.

        Returns:
            ExecResult with execution results.

        Raises:
            NotImplementedError: This is a stub. Wire to Prime APIs.
        """
        if not self._prime_available:
            raise NotImplementedError(
                "PrimeSandboxExecutor is not yet wired to Prime's sandbox APIs. "
                "To use this executor, you must:\n"
                "1. Install Prime/Verifiers packages\n"
                "2. Configure Prime credentials\n"
                "3. Implement the sandbox integration in this method\n"
                "\n"
                "For local development, use LocalSubprocessExecutor instead:\n"
                "  config = EnvConfig(executor_backend='local')"
            )

        # TODO: Implement actual Prime sandbox execution
        # The implementation should:
        # 1. Build runner script using harness.build_runner_script()
        # 2. Create a Prime sandbox session
        # 3. Execute the script with network_access disabled
        # 4. Capture stdout, stderr, exit_code
        # 5. Return ExecResult

        raise NotImplementedError("Prime sandbox integration not implemented")


def get_executor(backend: str = "local", **kwargs) -> Executor:
    """Factory function to get an executor by backend name.

    Args:
        backend: Either "local" or "prime".
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
    else:
        raise ValueError(f"Unknown executor backend: {backend}")
