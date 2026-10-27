"""Durable execution primitives shared by model and local tool calls.

The scientific layers decide *what* to investigate.  This module only decides
whether an individual external operation may be retried and how its durable
attempt is recorded.  It deliberately does not turn a failed operation into
scientific evidence.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, replace
import errno
import http.client
import socket
import time
from typing import Any, Callable, TypeVar
import urllib.error


T = TypeVar("T")


class HTTPStatusError(RuntimeError):
    """HTTP failure retaining the status code for retry classification."""

    def __init__(self, status_code: int, detail: str = "") -> None:
        self.status_code = int(status_code)
        self.detail = detail[:2_000]
        super().__init__(f"HTTP {self.status_code}: {self.detail}".strip())


@dataclass(frozen=True)
class ErrorClassification:
    category: str
    retryable: bool
    status_code: int | None = None


@dataclass(frozen=True)
class RetryPolicy:
    """Per-operation timeout and bounded exponential-backoff policy."""

    timeout_seconds: float = 180.0
    max_retries: int = 2
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.backoff_seconds < 0 or self.backoff_multiplier < 1 or self.max_backoff_seconds < 0:
            raise ValueError("backoff values are invalid")

    def for_action(self, action: Any) -> "RetryPolicy":
        """Allow a typed action to override execution knobs, never its scope."""

        parameters = getattr(action, "parameters", {})
        if not isinstance(parameters, dict):
            return self
        overrides: dict[str, Any] = {}
        for name, caster in {
            "timeout_seconds": float,
            "max_retries": int,
            "backoff_seconds": float,
            "backoff_multiplier": float,
            "max_backoff_seconds": float,
        }.items():
            if name in parameters:
                try:
                    overrides[name] = caster(parameters[name])
                except (TypeError, ValueError):
                    continue
        return replace(self, **overrides) if overrides else self


@dataclass(frozen=True)
class RetryOutcome:
    value: Any
    attempts: int
    retry_events: tuple[dict[str, Any], ...]


def classify_error(error: BaseException) -> ErrorClassification:
    """Classify transport failures without treating scientific errors as retryable."""

    status_code = getattr(error, "status_code", None)
    if status_code is None and isinstance(error, urllib.error.HTTPError):
        status_code = int(error.code)
    if status_code is not None:
        status_code = int(status_code)
        return ErrorClassification(
            category="rate_limited" if status_code == 429 else "server_error" if 500 <= status_code <= 599 else "http_error",
            retryable=status_code == 429 or 500 <= status_code <= 599,
            status_code=status_code,
        )

    if isinstance(error, (TimeoutError, socket.timeout, FutureTimeout)):
        return ErrorClassification("timeout", True)
    if isinstance(error, socket.gaierror):
        return ErrorClassification("dns_error", True)
    if isinstance(error, (ConnectionError, http.client.HTTPException)):
        return ErrorClassification("connection_error", True)
    if isinstance(error, urllib.error.URLError):
        reason = error.reason
        # urllib wraps DNS failures, refused connections, resets, and network
        # reachability failures in URLError.  These are transport failures,
        # not evidence or tool-contract failures.
        if isinstance(reason, socket.gaierror) or isinstance(reason, OSError):
            return ErrorClassification("dns_or_network_error", True)
        return ErrorClassification("url_error", True)
    if isinstance(error, OSError) and getattr(error, "errno", None) in {
        errno.EAI_AGAIN if hasattr(errno, "EAI_AGAIN") else -3,
        errno.ECONNRESET, errno.ECONNREFUSED, errno.ENETUNREACH,
        errno.ETIMEDOUT, errno.EHOSTUNREACH,
    }:
        return ErrorClassification("dns_or_network_error", True)
    return ErrorClassification("non_retryable_error", False)


def retry_metadata(error: BaseException) -> dict[str, Any]:
    """Return serializable retry diagnostics attached by ``execute_with_retry``."""

    value = getattr(error, "_retry_metadata", None)
    return dict(value) if isinstance(value, dict) else {}


def _call_once(operation: Callable[[], T], timeout_seconds: float) -> T:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(operation)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeout as error:
        future.cancel()
        raise TimeoutError(f"operation exceeded {timeout_seconds:g}s timeout") from error
    finally:
        # A timed-out worker cannot be forcibly killed.  Do not block the
        # controller on it; the action attempt remains durable and the next
        # retry has its own bounded wait.
        executor.shutdown(wait=False, cancel_futures=True)


def execute_with_retry(
    operation: Callable[[], T],
    policy: RetryPolicy,
    *,
    operation_name: str = "operation",
    sleep: Callable[[float], None] = time.sleep,
) -> RetryOutcome:
    """Run an operation with bounded timeout/retry and structured diagnostics."""

    retry_events: list[dict[str, Any]] = []
    for attempt in range(1, policy.max_retries + 2):
        try:
            return RetryOutcome(_call_once(operation, policy.timeout_seconds), attempt, tuple(retry_events))
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            classification = classify_error(error)
            event = {
                "operation": operation_name,
                "attempt": attempt,
                "category": classification.category,
                "retryable": classification.retryable,
                "status_code": classification.status_code,
                "error_type": type(error).__name__,
                "error": str(error)[:1_000],
            }
            retry_events.append(event)
            if not classification.retryable or attempt > policy.max_retries:
                setattr(error, "_retry_metadata", {"attempts": attempt, "events": retry_events, "final": event})
                raise
            delay = min(
                policy.max_backoff_seconds,
                policy.backoff_seconds * (policy.backoff_multiplier ** (attempt - 1)),
            )
            event["backoff_seconds"] = delay
            if delay:
                sleep(delay)
    raise AssertionError("unreachable")
