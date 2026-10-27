"""Optional model-backed deliberation with strict structural validation at the boundary."""

from __future__ import annotations

import json
from hashlib import sha256
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Any, Protocol

from jsonschema import Draft202012Validator, ValidationError

from .aiq_grounding import ReportClaim
from .contracts import EvidenceEntry, ResearchAction
from .deepseek_schemas import CLAIM_ENTAILMENT_SCHEMA, EVIDENCE_REVIEW_SCHEMA, FINAL_ANSWER_SCHEMA, GOAL_VERIFICATION_SCHEMA, NODE_CONCLUSION_SCHEMA, WRITER_SUBGRAPH_SCHEMA, planner_review_schema, planner_schema, research_action_schema, research_brief_schema, scope_resolution_schema, table_navigation_schema, tool_readiness_schema
from .tabular import CANONICAL_TABLE_FIELDS
from .resilience import HTTPStatusError, RetryPolicy, execute_with_retry


class JsonReasoner(Protocol):
    def complete(self, system: str, payload: dict[str, Any], *, schema: dict[str, Any] | None = None, tool_name: str = "submit_structured_result") -> dict[str, Any]: ...


class ReasoningUnavailable(RuntimeError):
    """Production research must not silently degrade into a keyword workflow."""


class StrictSchemaViolation(ReasoningUnavailable):
    """A provider result that does not satisfy the exact submitted schema."""


def _structured_complete(reasoner: Any, system: str, payload: dict[str, Any], *, schema: dict[str, Any], tool_name: str) -> dict[str, Any]:
    """Use server-side strict schema where supported; retain tiny test doubles.

    Real DeepSeek calls always take the first branch.  The compatibility branch
    exists solely for local deterministic test reasoners that implement the
    older two-argument protocol; runtime validation remains mandatory after it.
    """

    try:
        return reasoner.complete(system, payload, schema=schema, tool_name=tool_name)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        return reasoner.complete(system, payload)


class OpenAICompatibleJsonReasoner:
    """Small dependency-free client for a user-configured OpenAI-compatible endpoint."""

    def __init__(self, *, base_url: str, api_key: str, model: str, timeout_seconds: int = 90, retry_policy: RetryPolicy | None = None) -> None:
        self.base_url, self.api_key, self.model, self.timeout_seconds = base_url.rstrip("/"), api_key, model, timeout_seconds
        self.retry_policy = retry_policy or RetryPolicy(timeout_seconds=timeout_seconds)

    def complete(self, system: str, payload: dict[str, Any], *, schema: dict[str, Any] | None = None, tool_name: str = "submit_structured_result") -> dict[str, Any]:
        body = json.dumps({"model": self.model, "temperature": 0.1, "response_format": {"type": "json_object"}, "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}, method="POST")
        return execute_with_retry(
            lambda: self._complete_once(request), self.retry_policy, operation_name="openai_compatible_reasoner"
        ).value

    @staticmethod
    def _complete_once(request: urllib.request.Request) -> dict[str, Any]:
        body = request.data or b""
        request.add_header("Idempotency-Key", f"research-llm:{sha256(body).hexdigest()}")
        with urllib.request.urlopen(request) as response:  # noqa: S310 - endpoint is user-configured
            raw = json.loads(response.read().decode("utf-8"))
        return json.loads(raw["choices"][0]["message"]["content"])


class DeepSeekStrictToolReasoner:
    """DeepSeek's server-validated strict Function JSON Schema interface.

    The model selects a strict function call in thinking mode; this is not a
    prompt-only JSON convention and does not use Pydantic as a generation gate.
    """

    def __init__(self, *, api_key: str, model: str, base_url: str = "https://api.deepseek.com/beta", timeout_seconds: int = 180, max_tokens: int = 384_000, reasoning_effort: str = "max", invalid_argument_retries: int = 2, retry_policy: RetryPolicy | None = None) -> None:
        self.api_key = api_key
        self.model = model.split("[", 1)[0]
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        if reasoning_effort not in {"none", "low", "high", "max"}:
            raise ValueError("reasoning_effort must be one of none, low, high, max")
        self.reasoning_effort = reasoning_effort
        self.invalid_argument_retries = max(0, int(invalid_argument_retries))
        self.retry_policy = retry_policy or RetryPolicy(timeout_seconds=timeout_seconds)
        self.last_model_turn: dict[str, Any] | None = None

    def complete(self, system: str, payload: dict[str, Any], *, schema: dict[str, Any] | None = None, tool_name: str = "submit_structured_result") -> dict[str, Any]:
        if schema is None:
            raise ReasoningUnavailable("DeepSeek production calls require a strict JSON Schema")
        request_body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "enabled" if self.reasoning_effort != "none" else "disabled"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            "tools": [{"type": "function", "function": {"name": tool_name, "description": "Submit the required structured result.", "strict": True, "parameters": schema}}],
        }
        # The OpenAI-format API accepts low/high/max for reasoning effort.
        # ``none`` is represented only by thinking=disabled, not by an invalid
        # reasoning_effort value.
        if self.reasoning_effort != "none":
            request_body["reasoning_effort"] = self.reasoning_effort
        # Strict mode is a server contract, but a malformed tool argument is
        # still an external transport failure, never permission to infer or
        # repair scientific state locally.  Re-submit the *identical* request
        # a bounded number of times; only a fresh server-validated result may
        # cross this boundary.
        for attempt in range(self.invalid_argument_retries + 1):
            raw = self._post(request_body)
            message = raw.get("choices", [{}])[0].get("message", {})
            calls = message.get("tool_calls", [])
            finish_reason = raw.get("choices", [{}])[0].get("finish_reason", "unknown")
            self.last_model_turn = {
                "reasoning_content": message.get("reasoning_content", ""),
                "content": message.get("content", ""), "tool_calls": calls,
                "finish_reason": finish_reason, "attempt": attempt + 1,
            }
            if len(calls) != 1 or calls[0].get("function", {}).get("name") != tool_name:
                content_kind = "text" if message.get("content") else "empty"
                raise ReasoningUnavailable(
                    "DeepSeek strict endpoint did not select the required structured-result tool "
                    f"(finish_reason={finish_reason}, content={content_kind}, tool_call_count={len(calls)})."
                )
            try:
                arguments_object = json.loads(calls[0]["function"]["arguments"])
                Draft202012Validator(schema).validate(arguments_object)
                return arguments_object
            except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as error:
                arguments = calls[0].get("function", {}).get("arguments", "")
                diagnostic = {
                    "finish_reason": finish_reason,
                    "argument_length": len(arguments) if isinstance(arguments, str) else None,
                    "argument_sha256": sha256(arguments.encode("utf-8", errors="replace")).hexdigest() if isinstance(arguments, str) else None,
                    "parse_position": error.pos if isinstance(error, json.JSONDecodeError) else None,
                    "schema_path": list(error.absolute_schema_path) if isinstance(error, ValidationError) else [],
                    "instance_path": list(error.absolute_path) if isinstance(error, ValidationError) else [],
                    "schema_message": error.message[:500] if isinstance(error, ValidationError) else "",
                    "retry_attempt": attempt + 1,
                }
                self.last_model_turn["invalid_argument_diagnostic"] = diagnostic
                if attempt >= self.invalid_argument_retries:
                    raise StrictSchemaViolation(
                        "DeepSeek strict endpoint returned tool arguments that failed the locally revalidated schema after "
                        f"{attempt + 1} identical request(s): {diagnostic}"
                    ) from error

    def _post(self, request_body: dict[str, Any]) -> dict[str, Any]:
        return execute_with_retry(
            lambda: self._post_once(request_body), self.retry_policy, operation_name="deepseek_strict_reasoner"
        ).value

    def _post_once(self, request_body: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(request_body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}", "Idempotency-Key": f"research-llm:{sha256(body).hexdigest()}"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - user-configured provider credential
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:2000]
            raise HTTPStatusError(error.code, detail) from error
        return raw


class AnthropicCompatibleJsonReasoner:
    """Anthropic Messages client for DeepSeek's documented compatibility endpoint."""

    def __init__(self, *, base_url: str, auth_token: str, model: str, timeout_seconds: int = 180, reasoning_effort: str = "max", retry_policy: RetryPolicy | None = None) -> None:
        self.base_url, self.auth_token = base_url.rstrip("/"), auth_token
        # ``[1m]`` is a Claude Code selector suffix; DeepSeek's direct Messages
        # API documents ``deepseek-v4-pro`` as the model value.
        self.model = model.split("[", 1)[0]
        self.timeout_seconds = timeout_seconds
        self.retry_policy = retry_policy or RetryPolicy(timeout_seconds=timeout_seconds)
        self.reasoning_effort = reasoning_effort if reasoning_effort in {"none", "low", "high", "max"} else "max"

    def complete(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._request(system, payload, self.reasoning_effort)
        text = self._text(response)
        # V4-Pro may consume the available response on thinking tokens without
        # emitting a final text block.  A JSON-contract call cannot continue on
        # hidden reasoning alone, so retry once with thinking disabled.
        if not text and self.reasoning_effort != "none":
            response = self._request(system, payload, "none")
            text = self._text(response)
        if not text:
            content_types = [str(item.get("type")) for item in response.get("content", []) if isinstance(item, dict)]
            raise ReasoningUnavailable(f"Anthropic-compatible model returned no final text; stop_reason={response.get('stop_reason')!r}, content_types={content_types}")
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            # Unlike OpenAI's JSON response mode, the compatible Messages API
            # gives us text only.  Repair malformed syntax once, then retain
            # the normal caller-specific Pydantic gate as the authority.
            repair = self._request(
                "You are a JSON syntax repairer. Return only a valid JSON value that preserves the original response's intent and data. Do not add facts, tools, sources, claims, or explanations.",
                {"json_parse_error": str(error), "invalid_json": text},
                "none",
            )
            repaired_text = self._text(repair)
            try:
                return json.loads(repaired_text)
            except json.JSONDecodeError as repaired_error:
                raise ReasoningUnavailable("Anthropic-compatible model returned final text that is not valid JSON after one syntax-repair attempt") from repaired_error

    def _request(self, system: str, payload: dict[str, Any], effort: str) -> dict[str, Any]:
        body = json.dumps({
            "model": self.model, "max_tokens": 8192, "temperature": 0.1,
            "reasoning": {"effort": effort},
            "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}],
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/messages", data=body,
            headers={"Content-Type": "application/json", "x-api-key": self.auth_token, "anthropic-version": "2023-06-01", "Idempotency-Key": f"research-llm:{sha256(body).hexdigest()}"}, method="POST",
        )
        return execute_with_retry(
            lambda: self._request_once(request), self.retry_policy, operation_name="anthropic_compatible_reasoner"
        ).value

    @staticmethod
    def _request_once(request: urllib.request.Request) -> dict[str, Any]:
        with urllib.request.urlopen(request) as response:  # noqa: S310 - endpoint is user-configured
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _text(response: dict[str, Any]) -> str:
        return "".join(item.get("text", "") for item in response.get("content", []) if isinstance(item, dict) and item.get("type") == "text").strip()


class PlanAdvisor:
    def __init__(self, reasoner: JsonReasoner) -> None:
        self.reasoner = reasoner

    def propose(self, task: Any, scout: Any, skills: list[Any], *, evidence_vocabulary: list[str] | None = None, research_brief: Any | None = None) -> dict[str, Any]:
        scope_id = str(scout.task_scope.get("scope_id") or "")
        system = "You are the Scientific Planner Agent in an evidence-bound Deep Research system. You receive the complete Research Brief produced by Interpretation. Treat its object, spatial and temporal scope, core phenomenon or relation, assumptions and unresolved ambiguities, required knowledge gaps, and conclusion boundary as the problem definition. Do not reinterpret or narrow that Brief. Discover the unknowns that must be resolved to close the question and build the smallest falsifiable scientific decision DAG. Each component is a scientific claim, uncertainty boundary, alternative explanation, or identifiability question; depends_on expresses only scientific prerequisite relations. State observations, calculations, uncertainty information, or identifiability conditions that would resolve each unknown. These are open evidence requirements, not a closed list of registered tools. The Researcher later chooses freely among live registered sources and tools and may create any local execution shape needed. Scout is only a coarse catalogue supplement. Never write database fields, predicates, values, queries, tool ids, or action order. Never convert the Brief's knowledge gaps into download, cleaning, calculation, plotting, or export tasks. If evidence is unavailable, retain the unknown and represent the missing observability or identifiability as a scientific component or dependency. Call the supplied function exactly once."
        payload = {"user_request": task.question, "research_brief": asdict(research_brief) if research_brief is not None else None, "normalized_task": asdict(task), "scout_discovery": asdict(scout), "scientific_skills": [asdict(skill) for skill in skills], "capability_catalogue": [asdict(item) for item in getattr(scout, "sources", [])]}
        proposal = _structured_complete(self.reasoner, system, payload, schema=planner_schema(scope_ids=[scope_id], evidence_vocabulary=evidence_vocabulary), tool_name="submit_scientific_plan")
        # The strict output is a scientific decision graph. No execution lane
        # or registered evidence tag is reconstructed here.
        if "primary_component" in proposal:
            proposal["required_components"] = [proposal["primary_component"], *proposal.get("additional_components", [])]
        for component in proposal.get("required_components", []):
            # Strict DeepSeek output uses strings because its Beta schema does
            # not accept arbitrary maps.  Test/runtime adapters may already
            # provide decoded maps; both enter the same manual contract gate.
            if "constraints_json" in component:
                component["constraints"] = _object_from_json(component.pop("constraints_json"))
            # Compatibility with deterministic pre-strict test doubles.  The
            # result is overwritten with Scout's bundle below in either case.
            component.pop("scope_id", None)
            component.pop("scope_json", None)
            component["scope"] = scout.task_scope
        return proposal


class InterpretationAdvisor:
    """LLM adapter used only by the explicit Interpretation graph node."""

    def __init__(self, reasoner: JsonReasoner) -> None:
        self.reasoner = reasoner

    def interpret(self, question: str) -> dict[str, Any]:
        return _structured_complete(
            self.reasoner,
            "You are the Interpretation node of a scientific Deep Research system. Convert one ambiguous user question into a stable Research Brief. Define the research object, spatial and temporal bounds, the core phenomenon or relation, and the conclusion level that the evidence may reach. `must_answer` contains only the epistemic gaps needed to close the question, such as whether a credible spatiotemporal structure exists, whether major alternatives can be excluded, and what scope the conclusion applies to. Never put downloading, cleaning, statistics, plotting, table export, source names, tools, algorithms, component ids, DAG edges, evidence tags, thresholds, or execution order into the Brief. Preserve unresolved ambiguity explicitly; do not silently resolve it. Call the supplied function exactly once.",
            {"original_question": question},
            schema=research_brief_schema(),
            tool_name="submit_research_brief",
        )


class PlannerReviewAdvisor:
    """Top-level re-planning over a complete scientific-plan dossier.

    It is intentionally the only model role that decides whether an exhausted
    local route implies a changed observation world, a changed global route,
    or a formal gap.  It never writes a component conclusion.
    """

    def __init__(self, reasoner: JsonReasoner) -> None:
        self.reasoner = reasoner

    def review(self, *, question: str, plan: Any, component: Any, local_execution_graph: dict[str, Any], judge_decision: Any, scout: Any, component_conclusions: list[Any], component_evidence: list[Any], patch_rejection: dict[str, Any] | None = None) -> dict[str, Any]:
        response = _structured_complete(
            self.reasoner,
            "You are the top-level Scientific Planner reviewing a scientific-map event. The Research Brief is the immutable problem boundary; the Judge verdict is authoritative only about the current decision claim and does not choose a route. Choose exactly one route: repair_local, patch_graph, or formalize_gap. A map patch may add a falsifiable component, supersede an obsolete component, change prerequisite edges, reopen a component, or request a refreshed observation-world readout. Do not prescribe tools, queries, calculations, local action order, or a local directive. Use only the complete Research Brief, plan, component conclusions, local evidence packet, Scout, Judge record, and any prior patch rejection. Never create a conclusion, data, source, field, or capability. Call the supplied function exactly once.",
            {"original_user_question": question, "research_brief": asdict(plan.research_brief) if plan.research_brief else None, "scientific_plan": asdict(plan), "current_component": asdict(component), "local_execution_graph": local_execution_graph, "judge_decision": asdict(judge_decision), "isolated_component_conclusions": [asdict(item) for item in component_conclusions], "current_component_evidence_packet": [asdict(item) for item in component_evidence], "scout": asdict(scout), "prior_patch_rejection": patch_rejection},
            schema=planner_review_schema(
                component_ids=[item.component_id for item in plan.required_components],
                scope_ids=[str(scout.task_scope.get("scope_id", "default"))],
            ),
            tool_name="submit_planner_review",
        )
        if "decision" in response:
            response = response["decision"]
            if response.get("route") == "patch_graph":
                response["operations"] = [response.pop("primary_operation"), *response.pop("additional_operations", [])]
            else:
                response["operations"] = []
        for diagnosis in response.get("diagnoses", []):
            diagnosis["patch_scope"] = _object_from_json(diagnosis.pop("patch_scope_json"))
        return response


class ScopeResolutionAdvisor:
    """Maps open language to a source's live query contract, never to rows."""

    def __init__(self, reasoner: JsonReasoner) -> None:
        self.reasoner = reasoner

    def resolve(self, *, question: str, source_contract: dict[str, Any], task_scope: dict[str, Any], patch_scope: dict[str, Any]) -> dict[str, Any]:
        system = "You are the Planner's source-scope binder for one read-only observation source. Extract only direct observation-selection constraints: requested time, place, material, clock system, rock class or other dimensions that the live source contract can filter. Do not treat the scientific claim, requested calculation, uncertainty/bias test, mechanism, paleogeography, alternative hypothesis, or causal language as a database constraint. Put those phrases in research_terms_outside_query; they must never make a valid query partial. You may use only the supplied source_id and exact typed predicate alternatives from its query contract. For an age interval, use the provided numeric minimum_age_ma and maximum_age_ma fields; do not encode values as text. Do not invent geographic aliases, fields, categories, datasets, or records. If a requested observation-selection constraint cannot be expressed by the contract, list only that constraint in unresolved_terms and set status partial or scope_unresolvable. Use unconstrained only when the request genuinely supplies no source-relevant selection constraint."
        payload = {
            "question": question,
            "already_structured_task_scope": task_scope,
            "local_rescout_patch": patch_scope,
            "source_query_contract": source_contract,
        }
        return _structured_complete(
            self.reasoner,
            system,
            payload,
            schema=scope_resolution_schema(source_contract),
            tool_name="resolve_observation_scope",
        )

    def navigate_table(self, *, question: str, component: Any, source_catalogue: dict[str, Any], task_scope: dict[str, Any], local_scope: dict[str, Any], requested_table_id: str = "") -> dict[str, Any]:
        """Explore one unfamiliar table using only its current physical catalogue."""

        return _structured_complete(
            self.reasoner,
            "You are the data-navigation part of a scientific Researcher. Select one real table and bind only the physical columns needed by the current component's declared observations to the supplied canonical scientific fields. The original question, current component, and task scope define the requested observations; they do not authorize invented fields or proxy variables. Use only exact table_id and source_column enum values from the supplied live catalogue. If requested_table_id is non-empty, it is the physical table already declared by the current data operation and you must use exactly that table; it is not interchangeable with another table from the same source. Inspect column types, ranges, sample values, and row counts before binding. A binding says what a returned value physically represents; do not bind a merely correlated or convenient column. Predicates contain only direct record-selection conditions such as requested time, place, material, or clock system. Scientific hypotheses, calculations, robustness tests, and causal language belong in research_terms_outside_query. If the catalogue cannot support a required binding or selection, report it in unresolved_terms and return partial or scope_unresolvable. Do not guess aliases, units, rows, results, or conclusions. Keep the query read-only and bounded. Call the supplied function exactly once.",
            {
                "original_user_question": question,
                "current_component": asdict(component),
                "task_scope": task_scope,
                "local_scope_refinement": local_scope,
                "requested_table_id": requested_table_id,
                "source_catalogue": source_catalogue,
                "canonical_field_meanings": {
                    "age_ma": "event or analysis age in Ma", "age_error_ma": "uncertainty paired with age_ma in Ma",
                    "latitude": "latitude in decimal degrees", "longitude": "longitude in decimal degrees",
                    "position_uncertainty_km": "spatial uncertainty in kilometres", "event_interpretation": "source interpretation of the measured event",
                    "source_id": "record provenance", "sample_id": "sample identity", "paper_id": "publication identity",
                },
            },
            schema=table_navigation_schema(source_catalogue, canonical_fields=list(CANONICAL_TABLE_FIELDS), requested_table_id=requested_table_id or None),
            tool_name="submit_table_navigation",
        )


class ToolSelectionAdvisor:
    def __init__(self, reasoner: JsonReasoner) -> None:
        self.reasoner = reasoner

    def select(self, *, question: str, component: Any, candidates: list[Any], existing_tags: set[str], limit: int) -> list[str]:
        proposal = self.reasoner.complete(
            "You are a scientific research method selector. Select complementary registered tools that produce the required evidence. Do not invent tools, do not infer conclusions, and return JSON only as {tool_ids:[...]}. Prefer different statistical representations when appropriate.",
            {"question": question, "component": asdict(component), "existing_evidence_tags": sorted(existing_tags), "limit": limit, "candidates": [{"tool_id": tool.tool_id, "capability_group": tool.capability_group, "description": tool.description, "capabilities": tool.capabilities, "output_tags": tool.output_tags} for tool in candidates]},
        )
        allowed = {tool.tool_id for tool in candidates}
        return [tool_id for tool_id in proposal.get("tool_ids", []) if tool_id in allowed][:limit]


class ResearchActionAdvisor:
    """The model proposes only actions in the live registry; the executor remains authoritative."""

    def __init__(self, reasoner: JsonReasoner) -> None:
        self.reasoner = reasoner

    def propose(self, *, task: Any, local_graph: dict[str, Any], component: Any, scout: Any, skills: list[Any], tool_catalogue: list[dict[str, Any]], workspace_catalogue: list[dict[str, Any]] | None = None, matching_tool_ids: set[str], evidence: list[Any], component_evidence: list[Any], limit: int) -> list[ResearchAction]:
        response = _structured_complete(
            self.reasoner,
            "You are the scientific Researcher. The original user question remains authoritative. You receive a bounded scientific neighbourhood and a durable local execution graph for one current claim. The local graph is not a substitute for the global plan. You own method choice and local repair: you may replace a method, append prerequisite actions, or expand the local acyclic graph when prior evidence did not resolve the current claim. You may not repeat an equivalent action unless its declared upstream evidence, registered parameter input, data version, or registered tool-input state changed. Treat catalogue readiness and capability-coverage states as authoritative facts. Append new acyclic execution nodes only; action dependencies are declared with depends_on_action_ids and may point to already completed local nodes. `data_tool` is one read-only data-operation tool: use it for table discovery, selection, projection, union, explicit join or match, grouping, aggregation, binning, and materialization. It executes deterministic transformations internally and may return a common `decision_required` result when a relation, grouping, identity, unit, spatial tolerance, or temporal tolerance would change scientific meaning. In that case decide locally: specify the explicit relation intent and authorization, add a local probe/operation, reject the candidate, or state that the capability is unavailable. Do not let the tool choose a scientific boundary or silently enable fuzzy matching. For multi-table work, express the research relationship and matching policy in the request; do not create a special multi-table workflow or SQL. It is not a second Scout and must never contain SQL or prose field aliases. Use registered computations only after their required data artifact exists. Never invent data, tools, results, conclusions, global claims, or global dependencies. The catalogue is for initial selection only; complete capability cards are provided for selected computations before execution. If no legal local path remains, submit finish_component and state whether the method is absent or its inputs are unavailable. You may not alter the formal global plan. Call the supplied function exactly once. input_scope_json and parameters_json are JSON-object strings and must be {} when inapplicable. Return no more than the supplied action limit.",
            {"original_user_question": task.question, "local_research_graph": local_graph, "current_component": asdict(component), "observation_world": asdict(scout), "research_priors": [asdict(skill) for skill in skills], "tool_catalogue": [{**item, "initial_capability_match": item["tool_id"] in matching_tool_ids} for item in tool_catalogue], "data_tool_catalogue": list(workspace_catalogue or []), "ledger_evidence": [asdict(entry) for entry in evidence], "component_evidence": [asdict(entry) for entry in component_evidence], "action_limit": limit},
            schema=research_action_schema(
                tool_ids=[str(tool["tool_id"]) for tool in tool_catalogue],
                # ResearchActionAdvisor is called per frontier component, so
                # this enum is normally a singleton rather than a vague plan-
                # wide string field.
                component_ids=[component.component_id],
                source_ids=[profile.source_id for profile in scout.sources],
                workspace_catalogue=list(workspace_catalogue or []),
                tool_methods={str(tool["tool_id"]): list(tool.get("method_ids") or []) for tool in tool_catalogue},
                data_handle_evidence_ids=[
                    str(entry.evidence_id) for entry in evidence
                    if isinstance(getattr(entry, "findings", {}).get("dataset_handle"), dict)
                    and getattr(entry, "findings", {}).get("dataset_handle", {}).get("handle_contract") == "data_handle/v1"
                ],
                tool_requires_handle={str(tool["tool_id"]): bool(tool.get("requires_materialized_handle")) for tool in tool_catalogue},
                local_action_ids=[
                    str(node.get("action_id"))
                    for node in local_graph.get("execution_graph", {}).get("nodes", [])
                    if isinstance(node, dict) and node.get("action_id")
                ],
            ),
            tool_name="submit_research_actions",
        )
        # Production strict schema always contains a primary action.  Empty
        # actions are not a valid way to end a local graph; the model must use
        # its registered finish_component variant instead.  The old array is
        # accepted only for deterministic fixtures created before this schema.
        raw_actions = (
            [response["primary_action"], *response.get("additional_actions", [])]
            if "primary_action" in response else response.get("actions", [])
        )
        actions = []
        for raw in raw_actions[:limit]:
            try:
                raw["tool_id"] = raw["tool_id"] or None
                raw["source_id"] = raw["source_id"] or None
                raw["input_scope"] = _object_from_json(raw.pop("input_scope_json"))
                raw["parameters"] = _object_from_json(raw.pop("parameters_json"))
                raw["data_handle_evidence_id"] = str(raw.get("data_handle_evidence_id") or "")
                method = str(raw.pop("method", "") or "")
                if method:
                    raw["parameters"]["method"] = method
                actions.append(ResearchAction(**raw))
            except (KeyError, TypeError, ValueError) as error:
                raise StrictSchemaViolation("Researcher action failed the runtime ResearchAction contract") from error
        return actions

    def confirm_tool_actions(self, *, task: Any, local_graph: dict[str, Any], actions: list[ResearchAction], capability_cards: list[dict[str, Any]]) -> list[ResearchAction]:
        """仅向已选工具按需加载完整能力卡，再由模型复核是否执行。"""

        tool_actions = [action for action in actions if action.kind == "tool" and action.tool_id]
        if not tool_actions:
            return actions
        response = _structured_complete(
            self.reasoner,
            "You are the tool-call review stage for a scientific Researcher. The original question and local research graph remain authoritative. The following tools were selected from the catalogue and their complete capability cards are now available. For each action, determine whether its input objects, output objects, reference frame, and usage conditions can advance the current local task. Keep only actions that can advance it. You may reject every action and state the current gap. Do not propose new tools, data, conclusions, or changes to the formal plan. Call the supplied function exactly once.",
            {"original_user_question": task.question, "local_research_graph": local_graph, "candidate_actions": [asdict(action) for action in tool_actions], "selected_tool_capability_cards": capability_cards},
            schema=tool_readiness_schema(action_ids=[action.action_id for action in tool_actions]),
            tool_name="confirm_tool_actions",
        )
        accepted = set(response.get("accepted_action_ids", []))
        return [action for action in actions if action.kind != "tool" or action.action_id in accepted]


class EvidenceReviewAdvisor:
    """Independent R-layer Judge over one subproblem's evidence packet.

    It sees the component claim, its declared scope, and only the
    registered Ledger entries reaching that component.  It does not see the
    user question, the global Plan, the Researcher trajectory, or tools.  Its
    job is to decide whether the supplied calculation results support,
    contradict, or fail to establish this subproblem; it does not route or
    edit either graph.
    """

    def __init__(self, reasoner: JsonReasoner) -> None:
        self.reasoner = reasoner

    def review(self, packet: Any) -> dict[str, Any]:
        return _structured_complete(
            self.reasoner,
            "You are the isolated R-layer Judge for one scientific subproblem. You receive exactly the subproblem claim, its declared scope, and its registered Ledger calculation-result packet. Decide only whether the supplied scientific results support the subproblem, contradict it, or are unresolved. Judge the relation between result and claim, not whether evidence tags, result counts, or a particular method happened to appear. Calculation status, values, uncertainty, limitations, and provenance are inputs to your scientific judgment, not automatic pass/fail rules. Respect scope and uncertainty when they affect what the result means. If you report confidence, it must be your own assessment of this result-to-claim relation; the runtime must not derive it from status. Do not use outside knowledge, fill gaps, write a report, propose tools, or edit a Plan. Evidence IDs must be copied only from the supplied packet. If the result is numerically or formally valid but does not establish the claim, return unresolved and state the missing inference. Call the supplied function exactly once.",
            {"component_id": packet.component_id, "claim": packet.claim, "scope": packet.scope, "ledger_entries": [asdict(entry) for entry in packet.entries]},
            schema=EVIDENCE_REVIEW_SCHEMA, tool_name="submit_evidence_review",
        )


class IsolatedNodeConclusionAdvisor:
    """A writer restricted to one component and its local result packet."""

    def __init__(self, reasoner: JsonReasoner) -> None:
        self.reasoner = reasoner

    def conclude(self, packet: Any) -> dict[str, Any]:
        return _structured_complete(
            self.reasoner,
            "You are an isolated scientific node writer. You receive exactly one declared component, its deterministic assessment, and its local Ledger calculation results. You do not know the user question, other Plan nodes, Scout, tools, or research trajectory. State only what these results establish within the component scope. Do not use outside knowledge or create a mechanism explanation. Your status must equal the deterministic assessment outcome. Cite only supplied Ledger IDs. You must call the supplied function exactly once.",
            {"component": asdict(packet.component), "assessment": asdict(packet.assessment), "ledger_entries": [asdict(entry) for entry in packet.entries]},
            schema=NODE_CONCLUSION_SCHEMA, tool_name="submit_node_conclusion",
        )


class WriterSubgraphAdvisor:
    """Translate one conclusion-only local subgraph; never judge or route."""

    def __init__(self, reasoner: JsonReasoner) -> None:
        self.reasoner = reasoner

    def write_subgraph(self, packet: Any) -> dict[str, Any]:
        return _structured_complete(
            self.reasoner,
            "You are a scientific Writer compiler. Translate exactly one supplied local DAG subgraph into concise natural language. You are not a Judge, Planner, Researcher, or evidence reviewer: do not change statuses, decide sufficiency, add evidence, add mechanisms, choose tools, repair the graph, or infer beyond the supplied conclusions. Preserve the root node's status and claim boundary. Mention prerequisite relations only when they are present in the supplied subgraph. Return only text plus copied component and edge IDs from the packet. Call the supplied function exactly once.",
            {"root_component_id": packet.root_component_id, "component_ids": packet.component_ids, "node_ids": packet.node_ids, "edge_ids": packet.edge_ids, "conclusions": [asdict(item) for item in packet.conclusions]},
            schema=WRITER_SUBGRAPH_SCHEMA, tool_name="submit_writer_subgraph",
        )


class GoalPlanVerificationAdvisor:
    """Final Research Brief closure check over a conclusion-only DAG packet."""

    def __init__(self, reasoner: JsonReasoner) -> None:
        self.reasoner = reasoner

    def verify(self, *, question: str, plan: Any, synthesis: Any) -> dict[str, Any]:
        return _structured_complete(
            self.reasoner,
            "You are the final Research Brief closure verifier. You receive the complete Research Brief and a DAG synthesis of isolated node conclusions. You never receive raw data, tool outputs, Scout, or Researcher history. Decide only whether the scientific knowledge gaps in the Brief are closed, require a focused Planner repair, or have reached a formal evidence gap. Do not turn the Brief into an execution checklist. Do not create a scientific claim, evidence ID, tool, sensor, or new component. You must call the supplied function exactly once.",
            {"question": question, "research_brief": asdict(plan.research_brief) if plan.research_brief else None, "dag_synthesis": asdict(synthesis)},
            schema=GOAL_VERIFICATION_SCHEMA, tool_name="submit_goal_verification",
        )


class FinalAnswerSynthesisAdvisor:
    """Global Answer-node writer over a conclusion-only packet."""

    def __init__(self, reasoner: JsonReasoner) -> None:
        self.reasoner = reasoner

    def write_final_answer(self, packet: Any) -> dict[str, Any]:
        return _structured_complete(
            self.reasoner,
            "You are the final Answer-node Writer for a scientific Deep Research run. Compile the supplied local Writer sections in their DAG order into a coherent answer bounded by the Research Brief. You receive the original question, the complete Research Brief, the verified DAG synthesis, the verified Brief-closure status, formal evidence gaps, and conclusion-only local sections. You never receive raw data, Scout records, tools, Judge packets, or Researcher history. You are not a Judge and must not re-evaluate sufficiency or change the verified status. Preserve the conclusion level and branch/gap structure of the supplied sections. If status is evidence_gap, state exactly which knowledge gaps cannot be established and why; absence of evidence is not a negative scientific finding. Do not invent evidence, mechanisms, data sources, tools, components, or dependencies. You must call the supplied function exactly once.",
            {"question": packet.question, "research_brief": asdict(packet.research_brief) if packet.research_brief else None, "dag_synthesis": asdict(packet.synthesis), "goal_assessment": asdict(packet.goal_assessment), "evidence_gaps": [asdict(item) for item in packet.evidence_gaps], "local_sections": [asdict(item) for item in packet.local_sections]},
            schema=FINAL_ANSWER_SCHEMA, tool_name="submit_final_answer",
        )


class ClaimEntailmentAdvisor:
    """LLM verifier for a bounded claim/evidence comparison, never a report writer."""

    def __init__(self, reasoner: JsonReasoner) -> None:
        self.reasoner = reasoner

    def verify_claim(self, *, claim: ReportClaim, evidence: list[EvidenceEntry]) -> dict[str, Any]:
        return _structured_complete(
            self.reasoner,
            "You are a strict scientific entailment checker. Decide only whether supplied structured evidence entails the claim. Do not use outside knowledge, fill gaps, reinterpret unavailable data, or generate a better claim. You must call the supplied function exactly once. Mark not_entailed when evidence is insufficient or the claim exceeds its scope/version/uncertainty.",
            {"claim": asdict(claim) if hasattr(claim, "__dataclass_fields__") else claim.model_dump(mode="json"), "evidence": [asdict(entry) for entry in evidence]},
            schema=CLAIM_ENTAILMENT_SCHEMA, tool_name="submit_claim_entailment",
        )


def _object_from_json(raw: str) -> dict[str, Any]:
    """Decode an API-schema-constrained JSON-object string for runtime use."""
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise StrictSchemaViolation("Strict tool call contained a non-JSON object field") from error
    if not isinstance(value, dict):
        raise StrictSchemaViolation("Strict tool call object field did not decode to an object")
    return value


class ModelGuidedResearchPolicy:
    """Constrained model choice: it can select registry tools but cannot invent operations."""

    def __init__(self, advisor: ToolSelectionAdvisor, fallback: Any) -> None:
        self.advisor, self.fallback = advisor, fallback

    def choose_for_component(self, question: str, component: Any, candidates: list[Any], existing_tags: set[str], limit: int) -> list[Any]:
        try:
            selected_ids = self.advisor.select(question=question, component=component, candidates=candidates, existing_tags=existing_tags, limit=limit)
        except Exception as error:
            if self.fallback is None:
                raise ReasoningUnavailable("Research tool deliberation unavailable; configure a reasoning provider or use explicit test mode") from error
            return self.fallback.choose(candidates, existing_tags, limit)
        lookup = {tool.tool_id: tool for tool in candidates}
        return [lookup[tool_id] for tool_id in selected_ids] or self.fallback.choose(candidates, existing_tags, limit)
