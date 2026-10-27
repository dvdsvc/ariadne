"""Compact semantic tracing for a LangGraph run.

The public ``tasks`` stream is used only as a clock and node boundary.  The
persisted record is deliberately a semantic message trace: node inputs and
outputs, tool calls, status, and elapsed time.  Large row arrays and repeated
message bodies are represented by a small head/tail sample plus a digest.
Checkpoint and task protocol fields are not written to the trace.
"""

from __future__ import annotations

import json
import hashlib
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _namespace(value: Any) -> tuple[str, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(str(item) for item in value)
    if value:
        return (str(value),)
    return ()


def _namespace_label(namespace: Sequence[str]) -> str:
    return "/".join(namespace) if namespace else "root"


def _jsonable(value: Any) -> Any:
    """Convert LangGraph/LangChain values to JSON without dropping payloads."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


_MAX_TEXT = 4_000
_MAX_LIST_ITEMS = 8

# The progress artifact is intentionally much smaller than the raw semantic
# trace.  It answers "where is the research and what crossed the boundary?"
# without copying the same LangGraph state into every task event.
_PROGRESS_TEXT = 280
_PROGRESS_LIST_ITEMS = 4
_SIGNAL_KEYS = {
    "question",
    "technical_question",
    "conclusion_boundary",
    "required_outputs",
    "current_stage",
    "active_stage",
    "researcher_active_stage_id",
    "decision",
    "reason",
    "status",
    "summary",
    "claim",
    "finding",
    "interpretation",
    "next_action",
    "target_epistemic_question",
    "target_question",
    "tool_name",
    "tool_call_id",
    "item_id",
    "revision",
}


def _text_preview(value: Any, limit: int = _PROGRESS_TEXT) -> str:
    text = str(value).replace("\x00", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…"


def _progress_value(value: Any, *, key: str = "") -> Any:
    """Project a value for the small progress artifact.

    Large data arrays are represented by shape and a digest.  Message content
    and decision text remain readable, while full payloads stay in JSONL.
    """

    value = _jsonable(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return _text_preview(value)
        if isinstance(decoded, (dict, list)):
            return _progress_value(decoded, key=key)
        return _text_preview(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        if len(value) <= _PROGRESS_LIST_ITEMS:
            return [_progress_value(item, key=key) for item in value]
        return {
            "count": len(value),
            "sha256": _digest(value),
            "sample": [
                _progress_value(item, key=key)
                for item in value[:2]
            ],
        }
    if isinstance(value, dict):
        projected: dict[str, Any] = {}
        for name, item in value.items():
            name = str(name)
            lowered = name.lower()
            if (
                name in _SIGNAL_KEYS
                or lowered.endswith("_id")
                or lowered.endswith("_ids")
                or name in {"messages", "tool_calls", "tools", "stages"}
            ):
                projected[name] = _progress_value(item, key=name)
        if projected:
            return projected
        return {
            "keys": [str(name) for name in value.keys()][:20],
            "count": len(value),
            "sha256": _digest(value),
        }
    return _text_preview(value)


def _message_progress(value: Any) -> Any:
    value = _jsonable(value)
    if not isinstance(value, dict):
        return _progress_value(value)
    result: dict[str, Any] = {}
    for key in ("type", "name"):
        if key in value:
            result[key] = _progress_value(value[key], key=key)
    content = value.get("content")
    if isinstance(content, str) and content.strip():
        result["content"] = _text_preview(content, 160)
    elif isinstance(content, (int, float, bool)):
        result["content"] = content
    elif isinstance(content, (dict, list)):
        # Tool observations can contain thousands of rows.  Preserve the
        # result shape and digest, not the repeated payload.
        result["content_shape"] = {
            "type": "object" if isinstance(content, dict) else "array",
            "keys": list(content.keys())[:12] if isinstance(content, dict) else None,
            "count": len(content),
            "sha256": _digest(content),
        }
    tool_calls = value.get("tool_calls")
    if isinstance(tool_calls, list):
        result["tool_calls"] = []
        for call in tool_calls[:6]:
            if not isinstance(call, dict):
                result["tool_calls"].append(_progress_value(call))
                continue
            result["tool_calls"].append(
                {
                    "name": call.get("name"),
                    "arg_keys": sorted(str(key) for key in (call.get("args") or {}).keys())
                    if isinstance(call.get("args"), dict)
                    else [],
                    "args": {
                        str(key): (
                            _text_preview(item, 120)
                            if isinstance(item, str)
                            else {
                                "keys": list(item.keys())[:20],
                                "sha256": _digest(item),
                            }
                            if str(key) == "arguments_json" and isinstance(item, dict)
                            else _progress_value(item, key=str(key))
                        )
                        for key, item in (call.get("args") or {}).items()
                        if str(key) in {
                            "local_intent",
                            "request",
                            "source_id",
                            "scientific_input",
                            "temporary_tool_name",
                            "purpose",
                            "arguments_json",
                        }
                    } if isinstance(call.get("args"), dict) else {},
                }
            )
    return result


def _message_list_progress(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    # Keep the conversation order, but only the compact semantic part of each
    # message.  The full message history remains in the raw JSONL artifact.
    # Only the most recent action/result pair is useful for the step view;
    # earlier messages are repeated state and remain in the raw trace.
    return [_message_progress(item) for item in value[-2:]]


def _context_progress(value: Any, *, detail: str = "full") -> dict[str, Any]:
    value = _jsonable(value)
    if not isinstance(value, dict):
        return {"value": _progress_value(value)}
    result: dict[str, Any] = {"fields": [str(key) for key in value.keys()]}
    if detail == "step":
        for key in (
            "researcher_active_stage_id",
            "researcher_progress_decision",
            "department_plan_review",
            "objective_ledger",
            "messages",
        ):
            if key not in value:
                continue
            result[key] = (
                _message_list_progress(value[key])
                if key == "messages"
                else _progress_value(value[key], key=key)
            )
        return result
    for key in (
        "technical_task",
        "researcher_claim",
        "researcher_plan",
        "researcher_pending_plan",
        "conclusion_boundary",
        "required_outputs",
        "researcher_active_stage_id",
        "objective_ledger",
        "researcher_progress_decision",
        "department_task_docket",
        "plan",
        "messages",
    ):
        if key not in value:
            continue
        if key == "messages":
            result[key] = _message_list_progress(value[key])
        else:
            result[key] = _progress_value(value[key], key=key)
    return result


def _layer(namespace: Sequence[str], node: str) -> str:
    if namespace:
        return "R"
    return {
        "researcher": "R",
        "department": "D",
        "updater": "U",
        "planner": "P",
        "writer": "W",
    }.get(node, node)


def _reader_messages(value: Any) -> list[dict[str, Any]]:
    """Keep only the latest semantic action/result pair for a reader."""

    if not isinstance(value, list):
        return []
    actions: list[dict[str, Any]] = []
    for message in value[-2:]:
        if not isinstance(message, dict):
            continue
        if message.get("tool_calls"):
            tools: list[dict[str, Any]] = []
            for call in message["tool_calls"][:4]:
                if not isinstance(call, dict):
                    continue
                args = call.get("args") or {}
                action: dict[str, Any] = {"name": call.get("name")}
                if isinstance(args, dict):
                    for key in ("local_intent", "source_id", "temporary_tool_name"):
                        if key in args:
                            action[key] = _text_preview(args[key], 120)
                    request = args.get("request")
                    if isinstance(request, dict):
                        action["request_fields"] = list(request.keys())[:12]
                    elif request is not None:
                        action["request"] = _text_preview(request, 140)
                tools.append(action)
            actions.append({"kind": "model_action", "tools": tools})
        elif message.get("type") == "tool":
            item: dict[str, Any] = {
                "kind": "tool_result",
                "name": message.get("name"),
            }
            shape = message.get("content_shape")
            if isinstance(shape, dict):
                item["result"] = {
                    key: shape[key]
                    for key in ("type", "keys", "count", "sha256")
                    if key in shape
                }
            actions.append(item)
    return actions


def _reader_value(value: Any, *, key: str) -> Any:
    """Return the small, meaningful part of one state field."""

    if key in {
        "technical_task",
        "researcher_claim",
        "researcher_plan",
        "researcher_pending_plan",
        "technical_packet",
    }:
        if isinstance(value, dict):
            return {
                "keys": list(value.keys())[:20],
                "sha256": _digest(value),
            }
        return {"type": type(value).__name__, "sha256": _digest(value)}
    if key == "objective_ledger":
        if isinstance(value, dict):
            return {
                name: value[name]
                for name in ("count", "sha256")
                if name in value
            }
        return {"type": type(value).__name__, "count": len(value) if isinstance(value, list) else None}
    if key in {"researcher_progress_decision", "department_plan_review"}:
        if isinstance(value, dict):
            result = {}
            for name in ("decision", "status", "reason", "next_action"):
                if name in value:
                    result[name] = _text_preview(value[name], 220)
            return result
    if isinstance(value, str):
        return _text_preview(value, 180)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _progress_value(value, key=key)


def _reader_context(
    value: Any, *, include_fields: bool = True, include_messages: bool = True
) -> dict[str, Any]:
    """Project a state boundary without repeating the entire state."""

    if not isinstance(value, dict):
        return {"value": _reader_value(value, key="value")}
    fields = list(value.get("fields") or [])[:40]
    result: dict[str, Any] = {
        "fields": fields if include_fields else {"count": len(fields)},
    }
    for key in (
        "technical_task",
        "researcher_claim",
        "researcher_active_stage_id",
        "researcher_progress_decision",
        "department_plan_review",
        "objective_ledger",
    ):
        if key not in value:
            continue
        result[key] = _reader_value(value[key], key=key)
    if include_messages and "messages" in value:
        result["messages"] = _reader_messages(value["messages"])
    return result


def _reader_record(item: Mapping[str, Any]) -> dict[str, Any]:
    """Make one concise progress record from an internal task summary."""

    boundary = item.get("namespace") == "root" or item.get("node") in {
        "review_progress",
        "finalize",
    }
    record = {
        "sequence": item.get("sequence"),
        "layer": item.get("layer"),
        "namespace": item.get("namespace"),
        "node": item.get("node"),
        "status": item.get("status"),
        "duration_seconds": item.get("duration_seconds"),
    }
    if boundary:
        record["input"] = _reader_context(item.get("input"), include_messages=False)
        record["output"] = _reader_context(item.get("output"), include_messages=True)
    else:
        # Internal R steps remain visible in the timeline, but their repeated
        # state is reduced to field counts and the latest action/result.
        record["input"] = _reader_context(
            item.get("input"), include_fields=False, include_messages=False
        )
        record["output"] = _reader_context(
            item.get("output"), include_fields=False, include_messages=True
        )
    return record


def _episode_actions(value: Any) -> list[dict[str, Any]]:
    """Compress R actions while retaining repeated-tool visibility."""

    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for action in value:
        if not isinstance(action, dict):
            continue
        if action.get("kind") == "model_action":
            tools = []
            for tool in action.get("tools", []):
                if not isinstance(tool, dict):
                    continue
                compact = {"name": tool.get("name")}
                for key in ("local_intent", "source_id", "temporary_tool_name"):
                    if key in tool:
                        compact[key] = _text_preview(tool[key], 90)
                tools.append(compact)
            result.append({"kind": "model", "tools": tools})
        elif action.get("kind") == "tool_result":
            result.append({"kind": "result", "name": action.get("name")})
    return result


def _tiny_context(value: Any, field_ids: dict[str, int]) -> dict[str, Any]:
    """Encode a context boundary once, with shared field-name indexes."""

    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {
        "fields": [field_ids[name] for name in value.get("fields", []) if name in field_ids]
    }
    for key in (
        "technical_task",
        "researcher_claim",
        "researcher_active_stage_id",
        "researcher_progress_decision",
        "department_plan_review",
        "objective_ledger",
    ):
        if key in value:
            result[key] = _reader_value(value[key], key=key)
    return result


def _reader_document(
    progress: Sequence[Mapping[str, Any]],
    run: Mapping[str, Any],
    *,
    source_trace: str | None = None,
) -> dict[str, Any]:
    """Build the actual reader artifact.

    The raw task list is intentionally not copied here.  A short timeline
    shows research progress, root-layer records show cross-layer transfers,
    and R episodes show internal actions/decisions.  This is the useful view;
    the raw JSONL remains the lossless record.
    """

    records = [_reader_record(item) for item in progress]
    field_names = sorted(
        {
            field
            for record in records
            for side in (record.get("input"), record.get("output"))
            for field in (side.get("fields", []) if isinstance(side, dict) else [])
        }
    )
    field_ids = {name: index for index, name in enumerate(field_names)}
    timeline = [
        {
            "n": item.get("sequence"),
            "layer": item.get("layer"),
            "node": item.get("node"),
            "status": item.get("status"),
            "seconds": item.get("duration_seconds"),
        }
        for item in records
    ]
    transfers = [
        {
            "n": item.get("sequence"),
            "layer": item.get("layer"),
            "node": item.get("node"),
            "input": _tiny_context(item.get("input"), field_ids),
            "output": _tiny_context(item.get("output"), field_ids),
        }
        for item in records
        if item.get("namespace") == "root"
    ]
    episodes: list[dict[str, Any]] = []
    namespaces: list[str] = []
    for item in records:
        namespace = item.get("namespace")
        if namespace == "root" or namespace in namespaces:
            continue
        namespaces.append(namespace)
        episode_records = [record for record in records if record.get("namespace") == namespace]
        stages = []
        for record in episode_records:
            stage: dict[str, Any] = {
                "node": record.get("node"),
                "status": record.get("status"),
                "seconds": record.get("duration_seconds"),
            }
            output = record.get("output") or {}
            for key in ("researcher_active_stage_id", "researcher_progress_decision"):
                if key in output:
                    stage[key] = output[key]
            if output.get("messages"):
                stage["actions"] = _episode_actions(output["messages"])
            stages.append(stage)
        episodes.append(
            {
                "episode": len(episodes) + 1,
                "namespace": namespace,
                "stages": stages,
            }
        )
    slowest = sorted(
        (item for item in records if item.get("duration_seconds") is not None),
        key=lambda item: item["duration_seconds"],
        reverse=True,
    )[:10]
    document: dict[str, Any] = {
        "trace_contract": "research_progress_trace/v2",
        **({"source_trace": source_trace} if source_trace else {}),
        "run": dict(run),
        "field_catalog": field_names,
        "timeline": timeline,
        "layer_transfers": transfers,
        "research_episodes": episodes,
        "slowest_steps": [
            {
                "layer": item.get("layer"),
                "node": item.get("node"),
                "seconds": item.get("duration_seconds"),
                "status": item.get("status"),
            }
            for item in slowest
        ],
    }
    return document


def _overview_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Make the few-KB navigation file from the reader document."""

    episodes = []
    for episode in document.get("research_episodes", []):
        stages = []
        for stage in episode.get("stages", []):
            item = {
                "node": stage.get("node"),
                "seconds": stage.get("seconds"),
            }
            if stage.get("researcher_active_stage_id") is not None:
                item["stage"] = stage["researcher_active_stage_id"]
            decision = stage.get("researcher_progress_decision")
            if isinstance(decision, dict) and decision.get("decision"):
                item["decision"] = decision["decision"]
            actions = stage.get("actions") or []
            tool_names = [
                tool.get("name")
                for action in actions
                if isinstance(action, dict)
                for tool in action.get("tools", [])
                if isinstance(tool, dict) and tool.get("name")
            ]
            result_names = [
                action.get("name")
                for action in actions
                if isinstance(action, dict) and action.get("kind") == "result"
            ]
            if tool_names:
                item["tools"] = tool_names
            if result_names:
                item["results"] = result_names
            stages.append(item)
        episodes.append(
            {
                "episode": episode.get("episode"),
                "stages": stages,
            }
        )
    transfers = []
    for transfer in document.get("layer_transfers", []):
        transfers.append(
            {
                "n": transfer.get("n"),
                "layer": transfer.get("layer"),
                "node": transfer.get("node"),
                "seconds": next(
                    (
                        item.get("seconds")
                        for item in document.get("timeline", [])
                        if item.get("n") == transfer.get("n")
                    ),
                    None,
                ),
                "input_fields": (transfer.get("input") or {}).get("fields", []),
                "output_fields": (transfer.get("output") or {}).get("fields", []),
            }
        )
    run = document.get("run") or {}
    return {
        "trace_contract": "research_progress_overview/v1",
        "run": {
            key: run[key]
            for key in (
                "model",
                "thread_id",
                "question",
                "status",
                "started_at",
                "ended_at",
                "updated_at",
                "completed_steps",
                "open_steps",
            )
            if key in run
        },
        "field_catalog": document.get("field_catalog", []),
        "layer_transfers": transfers,
        "research_episodes": episodes,
        "slowest_steps": document.get("slowest_steps", [])[:5],
    }


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _compact(value: Any, *, key: str = "") -> Any:
    """Keep semantic structure while preventing state-sized trace records."""

    value = _jsonable(value)
    if isinstance(value, str):
        # Tool messages commonly carry a JSON document in a string.  Keeping
        # its structure makes the trace useful without retaining raw records.
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, (dict, list)):
            return _compact(decoded, key=key)
        if len(value) <= _MAX_TEXT:
            return value
        return {
            "_truncated": True,
            "length": len(value),
            "sha256": _digest(value),
            "head": value[:1_000],
            "tail": value[-500:],
        }
    if isinstance(value, dict):
        return {str(name): _compact(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, list):
        # Message order is itself part of the layer-to-layer conversation.
        # Retain every message, but compact each message's content instead of
        # dropping the middle of the conversation.
        if key == "messages":
            return [_compact(item, key=key) for item in value]
        if len(value) <= _MAX_LIST_ITEMS:
            return [_compact(item, key=key) for item in value]
        return {
            "_omitted_items": True,
            "count": len(value),
            "sha256": _digest(value),
            "head": [_compact(item, key=key) for item in value[:3]],
            "tail": [_compact(item, key=key) for item in value[-2:]],
        }
    return value


class LightweightTaskTrace:
    """Persist official ``tasks`` stream events and derive a human summary."""

    def __init__(
        self,
        path: Path,
        *,
        metadata: Mapping[str, Any],
        human_path: Path | None = None,
        progress_path: Path | None = None,
    ) -> None:
        self.path = path
        self.human_path = human_path or path.with_suffix(".md")
        self.progress_path = progress_path or path.with_name(
            f"{path.stem}_progress.json"
        )
        self.overview_path = path.with_name(f"{path.stem}_overview.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._sequence = 0
        self._starts: dict[tuple[str, str], int] = {}
        self._start_records: dict[tuple[str, str], dict[str, Any]] = {}
        self._summaries: list[dict[str, Any]] = []
        self._progress: list[dict[str, Any]] = []
        self._progress_metadata = _jsonable(dict(metadata))
        self._progress_started_at = _now()
        self._progress_status = "running"
        self._write(
            {
                "event": "run_start",
                "timestamp": _now(),
                **_jsonable(dict(metadata)),
            }
        )
        self._write_progress()

    def _write(self, event: Mapping[str, Any]) -> None:
        self._sequence += 1
        payload = {"sequence": self._sequence, **_compact(dict(event))}
        self._file.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        self._file.write("\n")
        self._file.flush()

    def on_part(self, part: Mapping[str, Any]) -> None:
        """Consume one v2 ``StreamPart`` from ``stream_mode='tasks'``."""

        if not isinstance(part, Mapping) or part.get("type") != "tasks":
            return
        namespace = _namespace(part.get("ns"))
        data = part.get("data")
        if not isinstance(data, Mapping):
            return
        task_id = str(data.get("id") or "")
        node = str(data.get("name") or "<unnamed>")
        namespace_label = _namespace_label(namespace)
        key = (namespace_label, task_id)
        timestamp = _now()
        monotonic_ns = time.perf_counter_ns()

        is_start = "input" in data and "result" not in data and "error" not in data
        if is_start:
            self._starts[key] = monotonic_ns
            detail = "full" if not namespace or node in {"review_progress", "finalize"} else "step"
            self._start_records[key] = {
                "timestamp": timestamp,
                "layer": _layer(namespace, node),
                "node": node,
                "namespace": namespace_label,
                "task_id": task_id,
                "detail": detail,
                "input": _context_progress(data.get("input"), detail=detail),
            }
            self._write(
                {
                    "event": "task_start",
                    "timestamp": timestamp,
                    "namespace": list(namespace),
                    "namespace_label": namespace_label,
                    "task_id": task_id,
                    "node": node,
                    "message": data.get("input"),
                }
            )
            return

        duration_seconds = None
        started = self._starts.pop(key, None)
        if started is not None:
            duration_seconds = round((monotonic_ns - started) / 1e9, 6)
        error = data.get("error")
        status = "failed" if error else "completed"
        if data.get("interrupts"):
            status = "interrupted"
        summary = {
            "namespace_label": namespace_label,
            "namespace": list(namespace),
            "task_id": task_id,
            "node": node,
            "status": status,
            "duration_seconds": duration_seconds,
            "error": error,
            "message": data.get("result"),
        }
        self._summaries.append(summary)
        started_record = self._start_records.pop(key, None)
        progress_record = {
            "sequence": self._sequence + 1,
            "layer": _layer(namespace, node),
            "namespace": namespace_label,
            "node": node,
            "task_id": task_id,
            "status": status,
            "started_at": started_record["timestamp"] if started_record else None,
            "ended_at": timestamp,
            "duration_seconds": duration_seconds,
            "input": started_record["input"] if started_record else {},
            "output": _context_progress(
                data.get("result"), detail=started_record.get("detail", "step")
            ),
        }
        if error:
            progress_record["error"] = _progress_value(error)
        self._progress.append(progress_record)
        self._write(
            {
                "event": "task_end",
                "timestamp": timestamp,
                **summary,
                "error": error,
            }
        )
        self._write_progress()

    def record_provider_cache_usage(self, usage: Mapping[str, Any]) -> None:
        """Persist the adapter's provider-neutral cache counters.

        The observer intentionally receives only ``cacheRead`` and
        ``cacheWrite`` (including its role scopes), never a provider's raw
        billing payload or request headers.
        """

        compact_usage = _compact(dict(usage))
        self._progress_metadata["provider_cache_usage"] = compact_usage
        self._write(
            {
                "event": "provider_cache_usage",
                "timestamp": _now(),
                "usage": compact_usage,
            }
        )
        self._write_progress()

    def finish(self, status: str, *, error: BaseException | None = None) -> None:
        self._progress_status = status
        end = {
            "event": "run_end",
            "timestamp": _now(),
            "status": status,
            "unfinished_task_count": len(self._starts),
            "failed_tasks": [
                item for item in self._summaries if item["status"] == "failed"
            ],
        }
        if error is not None:
            end["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        self._write(end)
        self._file.close()
        self._write_progress(error=error)
        self._write_human_summary(status, error)

    def _write_progress(self, *, error: BaseException | None = None) -> None:
        """Write a small, readable JSON snapshot and keep it current live."""

        completed = [item for item in self._progress if item["status"] == "completed"]
        run: dict[str, Any] = {
            **self._progress_metadata,
            "status": self._progress_status,
            "started_at": self._progress_started_at,
            "updated_at": _now(),
            "completed_steps": len(completed),
            "open_steps": len(self._starts),
        }
        if error is not None:
            run["error"] = {
                "type": type(error).__name__,
                "message": _text_preview(error, 500),
            }
        snapshot = _reader_document(self._progress, run)
        temporary = self.progress_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.progress_path)
        overview_temporary = self.overview_path.with_suffix(".json.tmp")
        overview_temporary.write_text(
            json.dumps(
                _overview_document(snapshot), ensure_ascii=False, indent=1
            ) + "\n",
            encoding="utf-8",
        )
        overview_temporary.replace(self.overview_path)

    def _write_human_summary(
        self, status: str, error: BaseException | None
    ) -> None:
        lines = [
            "# LangGraph task trace",
            "",
            f"- status: `{status}`",
            f"- task events: `{len(self._summaries)}`",
            "",
            "| namespace | node | status | duration (s) | error |",
            "|---|---|---:|---:|---|",
        ]
        for item in self._summaries:
            lines.append(
                "| {namespace_label} | {node} | {status} | {duration} | {error} |".format(
                    namespace_label=item["namespace_label"],
                    node=item["node"],
                    status=item["status"],
                    duration=item["duration_seconds"]
                    if item["duration_seconds"] is not None
                    else "-",
                    error=str(item["error"] or "").replace("|", "\\|"),
                )
            )
        if error is not None:
            lines.extend(["", "## Run error", "", f"`{type(error).__name__}: {error}`"])
        self.human_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_jsonl(source: Path, destination: Path | None = None) -> Path:
    """Convert an existing semantic JSONL trace to the small progress JSON."""

    destination = destination or source.with_name(f"{source.stem}_progress.json")
    starts: dict[tuple[str, str], dict[str, Any]] = {}
    progress: list[dict[str, Any]] = []
    run: dict[str, Any] = {}
    final_error: Any = None

    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            event_type = event.get("event")
            if event_type == "run_start":
                run = {
                    key: value
                    for key, value in event.items()
                    if key not in {"sequence", "event"}
                }
                run["started_at"] = event.get("timestamp")
            elif event_type == "task_start":
                key = (str(event.get("namespace_label") or "root"), str(event.get("task_id") or ""))
                namespace = event.get("namespace") or []
                starts[key] = {
                    "timestamp": event.get("timestamp"),
                    "layer": _layer(namespace, str(event.get("node") or "")),
                    "detail": (
                        "full"
                        if not namespace or str(event.get("node") or "") in {"review_progress", "finalize"}
                        else "step"
                    ),
                    "input": _context_progress(
                        event.get("message"),
                        detail=(
                            "full"
                            if not namespace or str(event.get("node") or "") in {"review_progress", "finalize"}
                            else "step"
                        ),
                    ),
                }
            elif event_type == "task_end":
                key = (str(event.get("namespace_label") or "root"), str(event.get("task_id") or ""))
                started = starts.pop(key, {})
                progress.append(
                    {
                        "sequence": event.get("sequence"),
                        "layer": started.get("layer") or _layer(
                            event.get("namespace") or [], str(event.get("node") or "")
                        ),
                        "namespace": event.get("namespace_label") or "root",
                        "node": event.get("node"),
                        "task_id": event.get("task_id"),
                        "status": event.get("status"),
                        "started_at": started.get("timestamp"),
                        "ended_at": event.get("timestamp"),
                        "duration_seconds": event.get("duration_seconds"),
                        "input": started.get("input", {}),
                        "output": _context_progress(
                            event.get("message"), detail=started.get("detail", "step")
                        ),
                    }
                )
            elif event_type == "run_end":
                run["status"] = event.get("status")
                run["ended_at"] = event.get("timestamp")
                final_error = event.get("error")

    run.update(
        {
            "completed_steps": sum(item.get("status") == "completed" for item in progress),
            "open_steps": len(starts),
        }
    )
    if final_error:
        run["error"] = _progress_value(final_error)
    document = _reader_document(progress, run, source_trace=str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    overview_path = destination.with_name(
        f"{destination.stem.replace('_progress', '')}_overview.json"
    )
    overview_path.write_text(
        json.dumps(_overview_document(document), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = ["LightweightTaskTrace", "compact_jsonl"]
