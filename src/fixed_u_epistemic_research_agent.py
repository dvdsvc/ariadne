"""Fixed-U parallel explanation-assessment subset of the mixed graph.

I establishes the constitution.  U freezes and fans out one C project per
material competing explanation.  Each project has a D-frozen T table whose R
tasks run independently; D reduces them to one explanation assessment.  U
normalizes all returned assessments into one evidence matrix, and W adjudicates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Annotated, Any, Callable, Literal, Protocol, Sequence, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, Send, default_retry_on
from typing_extensions import TypedDict


JSON = dict[str, Any]
TechnicalTool = BaseTool | Callable[..., Any]
MAX_OUTPUT_TOKENS = 128 * 1024
MODEL_REQUEST_TIMEOUT_SECONDS = 120.0


class EpistemicPolicy(TypedDict):
    policy_id: str
    principle: str
    few_shot: dict[str, str]
    role_permissions: dict[str, str]


BASE_EPISTEMIC_POLICY: EpistemicPolicy = {
    "policy_id": "fixed-scientific-argument-tree-v1",
    "principle": (
        "研究以固定的竞争解释评估树组织：U一次性把竞争解释展开为并发C项目；"
        "D在每个C下冻结技术事项T，R执行T，D分别评估支持、反证和判别性；"
        "U把所有返回归一为共同证据矩阵，W才作最终裁决。"
        "任何项目评估解释，不为解释辩护；技术信息只保留会影响科学判断的事实。"
    ),
    "few_shot": {
        "task": "基于碎屑锆石U-Pb年龄数据，比较候选物源或古地理解释的证据强度。",
        "background_knowledge": (
            "区域构造演化、候选物源和已知年龄事件可以指导竞争解释；"
            "当前数据负责检验年龄分布、相似性、差异和稳健性。"
        ),
        "must_be_data_validated": (
            "候选解释是否与样本实际年龄结构相符、哪些观察具有判别力、"
            "替代解释是否仍可成立以及结论的适用边界。"
        ),
        "wrong": "R得到一个统计结果，因此U自动把某个古地理解释写成已证实。",
        "correct": "D保留样本范围、筛选、方法、参数、统计量和稳健性；U判断它对竞争解释是否具有判别力。",
    },
    "role_permissions": {
        "I": "固定研究对象、边界与输出，不建立C或T，不选择技术路径。",
        "U": "第一次把竞争解释展开为有限、相互独立评估的C项目；所有D返回后归一为证据矩阵；不得改写冻结C树或设计技术方案。",
        "P": "本版本不作为模型角色；LangGraph并发路由冻结C项目和T事项。",
        "D": "为一个C一次性建立固定T树，独立评估该解释的支持、反证和判别性；R返回后形成结构化技术结论；不得改写C树或作U级总体裁决。",
        "R": "仅执行一个D签发的T；自主选择工具和局部DAG，返回事实、计算、不确定性和可审查技术信息。",
        "W": "依据完整冻结C树、D技术结论和U认识结论写作，不增加新证据或改变裁决。",
    },
}


class InterpretationConstitution(TypedDict):
    research_object: str
    boundaries: list[str]
    requested_output: str


class InquiryConstitution(TypedDict):
    question: str
    research_object: str
    boundaries: list[str]
    requested_output: str
    epistemic_policy: EpistemicPolicy


class EpistemicItemDraft(TypedDict):
    """U writes semantics; code attaches provenance identities."""

    item_id: str
    question: str
    status: Literal["open", "resolved", "limited"]
    supported: list[str]
    challenged: list[str]
    uncertain: list[str]
    remaining_gap: list[str]


class EpistemicItem(EpistemicItemDraft):
    evidence_refs: list[str]


class OverallEpistemicState(TypedDict):
    current_answer: str
    claim_ceiling: str


class EpistemicMapDraft(TypedDict):
    revision: int
    items: list[EpistemicItemDraft]
    overall_state: OverallEpistemicState


class EpistemicMap(TypedDict):
    revision: int
    items: list[EpistemicItem]
    overall_state: OverallEpistemicState


class InterpretationOutput(TypedDict):
    constitution: InterpretationConstitution


class UpdaterOutput(TypedDict):
    epistemic_map: EpistemicMapDraft


class EpistemicSnapshot(TypedDict):
    constitution: InquiryConstitution
    decision: Literal["close", "continue", "restructure"]
    epistemic_map: EpistemicMap


class LocalEpistemicPackage(TypedDict):
    item: EpistemicItem
    constitution: InquiryConstitution


class PlannerSelection(TypedDict):
    selected_item_id: str


class RCapabilityEnvelope(TypedDict):
    summary: str


class DepartmentTaskDraft(TypedDict):
    """D writes task semantics; code attaches identity and target."""

    technical_objective: str
    scope: list[str]
    required_outputs: list[str]
    acceptance_criteria: list[str]
    coarse_stages: list[str]
    allowed_resources: list[str]
    escalation_conditions: list[str]


class TechnicalTask(DepartmentTaskDraft):
    task_id: str
    target_epistemic_item: str


class RTaskClaim(TypedDict):
    task_id: str


class RLocalStage(TypedDict):
    stage_id: str
    operation_objective: str
    depends_on: list[str]


class RPlanProposalDraft(TypedDict):
    reason: str
    stages: list[RLocalStage]


class RExecutionPlan(TypedDict):
    task_id: str
    revision: int
    stages: list[RLocalStage]


class RStageResult(TypedDict):
    """One completed or failed local-DAG stage, projected for later R turns.

    This is an execution artifact rather than a new scientific contract: raw
    messages remain in the stage subgraph checkpoint and trace, while later R
    turns receive only a dependency-scoped projection.
    """

    record_id: str
    stage_id: str
    status: Literal["completed", "failed"]
    tool_call_ids: list[str]
    observations: list[JSON]


class RPlanProposal(TypedDict):
    proposal_kind: Literal["initial", "patch"]
    reason: str
    candidate_plan: RExecutionPlan


class DPlanReview(TypedDict):
    decision: Literal["approve", "reject"]
    boundary_findings: list[str]


class RProgressDecision(TypedDict):
    decision: Literal["continue", "propose_patch", "submit"]
    reason: str


class DepartmentCoreEvidence(TypedDict):
    statement: str
    source: str
    evidence_refs: list[str]


class DepartmentReport(TypedDict):
    report_id: str
    item_id: str
    what_is_known: str
    core_evidence: list[DepartmentCoreEvidence]
    conclusion_boundary: str
    remaining_uncertainty: list[str]
    cross_item_implications: list[str]


class DepartmentResults(TypedDict):
    what_is_known: str
    core_evidence: list[str]
    conclusion_boundary: str
    remaining_uncertainty: list[str]
    cross_item_implications: list[str]


class TechnicalResults(TypedDict):
    results: list[str]


class TechnicalPacket(TypedDict):
    packet_id: str
    task_id: str
    item_id: str
    results: list[str]
    tool_call_ids: list[str]


class WriterOutput(TypedDict):
    answer: str


class CognitiveItem(TypedDict):
    c_id: str
    explanation_under_evaluation: str
    assessment_question: str
    scientific_role: str
    predicted_observations: list[str]
    counterevidence_signals: list[str]
    required_evidence: list[str]
    decision_target: str


class UPlanOutput(TypedDict):
    cognitive_items: list[CognitiveItem]


class TechnicalItemDraft(TypedDict):
    t_id: str
    question: str
    scientific_role: str
    assessment_dimension: Literal["support", "counterevidence", "discrimination", "robustness"]
    technical_task: DepartmentTaskDraft


class TechnicalItem(TechnicalItemDraft):
    task_id: str


class DepartmentPlanOutput(TypedDict):
    technical_items: list[TechnicalItemDraft]


class TechnicalBasis(TypedDict):
    data_scope: list[str]
    selection_conditions: list[str]
    sample_size: str
    grouping_criteria: list[str]
    methods: list[str]
    key_parameters: list[str]
    thresholds: list[str]
    core_statistics: list[str]
    uncertainty: list[str]
    robustness_checks: list[str]


class TechnicalConclusionDraft(TypedDict):
    t_id: str
    conclusion: str
    evidence_basis: list[str]
    technical_basis: TechnicalBasis
    uncertainty: list[str]
    scope: list[str]
    status: Literal["completed", "limited", "inconclusive"]


class TechnicalConclusion(TechnicalConclusionDraft):
    packet_refs: list[str]


class DepartmentConclusionOutput(TypedDict):
    c_id: str
    explanation_under_evaluation: str
    conclusion: str
    technical_conclusions: list[TechnicalConclusionDraft]
    supporting_evidence: list[str]
    counterevidence: list[str]
    discriminative_evidence: list[str]
    assessment_status: Literal["supported", "weakened", "mixed", "not_discriminated", "insufficient"]
    uncertainty: list[str]
    scope: list[str]


class DepartmentConclusion(DepartmentConclusionOutput):
    department_conclusion_id: str
    technical_conclusions: list[TechnicalConclusion]


class EvidenceMatrixRow(TypedDict):
    c_id: str
    explanation: str
    supporting_evidence: list[str]
    counterevidence: list[str]
    discriminative_evidence: list[str]
    non_discriminative_evidence: list[str]
    independent_evidence_chains: list[str]
    uncertainty: list[str]
    scope: list[str]
    status: Literal["supported", "weakened", "mixed", "still_viable", "not_discriminated", "insufficient"]


class EpistemicConclusionOutput(TypedDict):
    evidence_matrix: list[EvidenceMatrixRow]
    normalization_notes: list[str]
    cross_explanation_conflicts: list[str]
    unresolved_items: list[str]


ROLE_CONTRACTS: dict[str, type] = {
    "interpretation": InterpretationOutput,
    "u_plan": UPlanOutput,
    "department_plan": DepartmentPlanOutput,
    "researcher_claim": RTaskClaim,
    "researcher_plan": RPlanProposalDraft,
    "department_plan_review": DPlanReview,
    "researcher_progress": RProgressDecision,
    "technical_results": TechnicalResults,
    "department_conclusion": DepartmentConclusionOutput,
    "u_conclusion": EpistemicConclusionOutput,
    "writer": WriterOutput,
}


def api_schemas() -> dict[str, JSON]:
    generated: dict[str, JSON] = {}
    for name, contract in ROLE_CONTRACTS.items():
        tool_schema = convert_to_openai_tool(contract, strict=True)
        generated[name] = copy.deepcopy(tool_schema["function"]["parameters"])
    return generated


class ModelAdapter(Protocol):
    model: BaseChatModel
    bind_tools_kwargs: JSON

    def structured(
        self,
        schema: type,
        *,
        cache_scope: str = "unscoped",
    ) -> Runnable[Any, Any]: ...

    def invoke_tools(
        self,
        tools: Sequence[TechnicalTool],
        messages: Sequence[AnyMessage],
        *,
        cache_scope: str,
        tool_choice: str,
    ) -> AIMessage: ...

    def cache_usage(self) -> JSON: ...


@dataclass(frozen=True, slots=True)
class ProviderConfiguration:
    """One LangChain-facing configuration for a supported model provider.

    Provider transport, message conversion, and tool-call parsing stay in the
    maintained LangChain integration. This graph owns only provider selection
    and its public environment-variable contract.
    """

    name: str
    api_base: str
    api_key_environment: tuple[str, ...]
    model_environment: tuple[str, ...]
    strict_function_calling: bool
    # GLM and most OpenAI-compatible providers cache implicitly.  Only the
    # maintained OpenAI integration receives an affinity key; sending that
    # non-standard field to every compatible endpoint would create avoidable
    # request failures.
    supports_prompt_cache_key: bool = False
    # Pi's provider adapter normalizes cache reads to prior-request hits only.
    # Keep this explicit for the few gateways that report a request's writes
    # inside their "cached" total.
    cache_read_includes_write: bool = False


OPENAI_COMPATIBLE_PROVIDERS: dict[str, ProviderConfiguration] = {
    "glm": ProviderConfiguration(
        name="GLM",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        api_key_environment=("ZAI_API_KEY", "ZHIPUAI_API_KEY"),
        model_environment=("GLM_MODEL",),
        # Do not assume every OpenAI-compatible endpoint implements strict.
        strict_function_calling=False,
    ),
    "qwen": ProviderConfiguration(
        name="Qwen",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_environment=("DASHSCOPE_API_KEY",),
        model_environment=("QWEN_MODEL",),
        strict_function_calling=False,
    ),
    "mimo": ProviderConfiguration(
        name="MiMo",
        api_base="https://api.xiaomimimo.com/v1",
        api_key_environment=("MIMO_API_KEY",),
        model_environment=("MIMO_MODEL",),
        strict_function_calling=False,
    ),
    "openai": ProviderConfiguration(
        name="OpenAI",
        api_base="https://api.openai.com/v1",
        api_key_environment=("OPENAI_API_KEY",),
        model_environment=("OPENAI_MODEL",),
        strict_function_calling=True,
        supports_prompt_cache_key=True,
    ),
}

PROVIDER_ALIASES = {
    "chatglm": "glm",
    "gpt": "openai",
    "zhipu": "glm",
    "dashscope": "qwen",
    "xiaomi": "mimo",
}


def resolve_provider_name(provider: str) -> str:
    """Return the canonical configured provider name, without importing SDKs."""

    resolved = PROVIDER_ALIASES.get(provider.strip().lower(), provider.strip().lower())
    if resolved not in {"deepseek", *OPENAI_COMPATIBLE_PROVIDERS}:
        supported = ", ".join(("deepseek", *OPENAI_COMPATIBLE_PROVIDERS))
        raise ValueError(f"unsupported provider {provider!r}; choose one of {supported}")
    return resolved


def provider_api_base(provider: str) -> str:
    """Return the default API endpoint used by the selected provider."""

    resolved = resolve_provider_name(provider)
    if resolved == "deepseek":
        return "https://api.deepseek.com/v1"
    return OPENAI_COMPATIBLE_PROVIDERS[resolved].api_base


def _first_environment_value(names: Sequence[str]) -> str | None:
    return next((value for name in names if (value := os.getenv(name))), None)


def resolve_provider_api_key(
    provider: str,
    *,
    api_key: str | None = None,
) -> str:
    """Resolve a provider key without exposing it in traces or error messages."""

    if api_key:
        return api_key
    resolved = resolve_provider_name(provider)
    environment = (
        ("DEEPSEEK_API_KEY",)
        if resolved == "deepseek"
        else OPENAI_COMPATIBLE_PROVIDERS[resolved].api_key_environment
    )
    value = _first_environment_value(environment)
    if value:
        return value
    raise ValueError(f"set one of {', '.join(environment)} or pass api_key=...")


def resolve_provider_model(provider: str, model: str | None = None) -> str:
    if model:
        return model
    resolved = resolve_provider_name(provider)
    environment = (
        ("DEEPSEEK_MODEL",)
        if resolved == "deepseek"
        else OPENAI_COMPATIBLE_PROVIDERS[resolved].model_environment
    )
    value = _first_environment_value(environment)
    if value:
        return value
    raise ValueError(f"pass model=... or set one of {', '.join(environment)}")


class StructuredOutputFailure(ValueError):
    def __init__(
        self,
        *,
        category: Literal[
            "missing_tool_call",
            "invalid_tool_call",
            "missing_parsed_result",
            "parsing_error",
            "schema_contract",
        ],
        schema_name: str,
        raw_response: Any,
        parsing_error: BaseException | None = None,
    ) -> None:
        self.category = category
        self.schema_name = schema_name
        self.raw_response = raw_response
        self.parsing_error = parsing_error
        detail = (
            f"{type(parsing_error).__name__}: {parsing_error}"
            if parsing_error is not None
            else "provider response did not produce the required parsed structure"
        )
        super().__init__(f"structured-output {category} for {schema_name}: {detail}")

    def as_record(self) -> JSON:
        return {
            "layer": "structured_output_adapter",
            "category": self.category,
            "schema_name": self.schema_name,
            "parsing_error": (
                {
                    "type": type(self.parsing_error).__name__,
                    "message": str(self.parsing_error),
                }
                if self.parsing_error is not None
                else None
            ),
            "raw_response": self.raw_response,
        }


class InvalidToolCallFailure(ValueError):
    def __init__(self, raw_response: AIMessage) -> None:
        self.raw_response = raw_response
        super().__init__("provider returned invalid tool calls")


def _unwrap_structured_result(result: Any, schema: type) -> Any:
    if not isinstance(result, dict) or not {"raw", "parsed", "parsing_error"} <= result.keys():
        raise TypeError("with_structured_output(include_raw=True) returned an invalid envelope")
    raw = result["raw"]
    parsed = result["parsed"]
    parsing_error = result["parsing_error"]
    schema_name = getattr(schema, "__name__", str(schema))
    if parsing_error is not None:
        failure = StructuredOutputFailure(
            category="parsing_error",
            schema_name=schema_name,
            raw_response=raw,
            parsing_error=parsing_error,
        )
        raise failure from parsing_error
    if parsed is None:
        if isinstance(raw, AIMessage) and raw.invalid_tool_calls:
            category = "invalid_tool_call"
        elif isinstance(raw, AIMessage) and not raw.tool_calls:
            category = "missing_tool_call"
        else:
            category = "missing_parsed_result"
        raise StructuredOutputFailure(
            category=category,
            schema_name=schema_name,
            raw_response=raw,
        )
    return parsed


def _retry_model_boundary(exc: Exception) -> bool:
    if isinstance(exc, (StructuredOutputFailure, InvalidToolCallFailure)):
        return True
    try:
        import openai

        if isinstance(exc, openai.APIConnectionError):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code in {408, 409, 429} or exc.status_code >= 500
        if isinstance(exc, openai.OpenAIError):
            return False
    except ImportError:
        pass
    return default_retry_on(exc)


MODEL_RETRY_POLICY = RetryPolicy(
    initial_interval=2.0,
    backoff_factor=2.0,
    max_interval=8.0,
    max_attempts=4,
    jitter=False,
    retry_on=_retry_model_boundary,
)


def provider_max_output_tokens(provider: str) -> int:
    """Return a bounded response budget for one fixed-U model turn.

    Fixed-U exchanges compact frozen maps and conclusions rather than a whole
    long-form report in one response.  Keeping the response ceiling provider
    configurable prevents an OpenAI-compatible endpoint from reserving the
    global research-report budget for a small structured planning call.
    """

    configured = os.getenv("FIXED_U_RESEARCH_MAX_OUTPUT_TOKENS")
    if configured:
        try:
            value = int(configured)
        except ValueError as exc:
            raise ValueError(
                "FIXED_U_RESEARCH_MAX_OUTPUT_TOKENS must be a positive integer"
            ) from exc
        if value <= 0:
            raise ValueError("FIXED_U_RESEARCH_MAX_OUTPUT_TOKENS must be positive")
        return min(value, MAX_OUTPUT_TOKENS)
    # GLM Flash is used for compact structured role outputs in this runtime;
    # do not reserve the global 128k report budget per request.
    if resolve_provider_name(provider) == "glm":
        return 8_192
    return MAX_OUTPUT_TOKENS


def provider_request_timeout_seconds(provider: str) -> float:
    """Return the per-call timeout without changing the graph retry policy."""

    configured = os.getenv("FIXED_U_RESEARCH_REQUEST_TIMEOUT_SECONDS")
    if configured:
        try:
            value = float(configured)
        except ValueError as exc:
            raise ValueError(
                "FIXED_U_RESEARCH_REQUEST_TIMEOUT_SECONDS must be a positive number"
            ) from exc
        if value <= 0:
            raise ValueError(
                "FIXED_U_RESEARCH_REQUEST_TIMEOUT_SECONDS must be positive"
            )
        return value
    # GLM's structured calls can remain server-side longer than the former
    # DeepSeek-oriented 120-second ceiling.  This only bounds one request;
    # LangGraph still owns retrying and durable checkpoint recovery.
    if resolve_provider_name(provider) == "glm":
        return 600.0
    return MODEL_REQUEST_TIMEOUT_SECONDS


@dataclass(slots=True)
class LangChainChatAdapter:
    """Provider boundary with role-scoped prompt-cache activation and accounting.

    Graph roles do not learn a provider's usage field names.  They select a
    stable cache scope (I/U/D/R/W).  The adapter maps the scope to the
    provider's supported routing hint when one exists; for implicit-cache
    providers such as GLM it deliberately sends no invented parameter and
    relies on the stable message prefix prepared by the graph.  It records the
    provider response and exposes only Pi-style ``cacheRead`` and
    ``cacheWrite`` counters.  A scope is deliberately a *role* rather than a
    task ID: R workers share their stable system prompt and tool schema cache,
    while D or U do not share a cache affinity domain.
    """

    model: BaseChatModel
    structured_output_kwargs: JSON = field(default_factory=dict)
    bind_tools_kwargs: JSON = field(default_factory=dict)
    provider: str = "unknown"
    supports_prompt_cache_key: bool = False
    _cache_usage_by_scope: dict[str, JSON] = field(default_factory=dict, init=False, repr=False)
    _cache_usage_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    @staticmethod
    def _nonnegative_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    @classmethod
    def _first_usage_value(cls, sources: Sequence[Any], keys: Sequence[str]) -> int:
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key in keys:
                value = cls._nonnegative_int(source.get(key))
                if value is not None:
                    return value
        return 0

    @staticmethod
    def _cache_scope_name(cache_scope: str) -> str:
        normalized = "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in cache_scope.strip().lower()
        ).strip("-")
        return normalized or "unscoped"

    def _cache_request_kwargs(self, cache_scope: str) -> JSON:
        """Use an affinity key only where the provider contract supports it."""

        if not self.supports_prompt_cache_key:
            return {}
        model_name = str(
            getattr(self.model, "model_name", None)
            or getattr(self.model, "model", None)
            or "default"
        )
        scope = self._cache_scope_name(cache_scope)
        # This key has no user question, task id, or credentials.  It is only
        # a server-routing hint for the stable prefix owned by this role.
        return {"prompt_cache_key": f"fixed-u:{self.provider}:{model_name}:{scope}"}

    def _scoped_model(self, cache_scope: str) -> Runnable[Any, Any]:
        request_kwargs = self._cache_request_kwargs(cache_scope)
        return self.model.bind(**request_kwargs) if request_kwargs else self.model

    def _extract_cache_usage(self, response: Any) -> tuple[int, int]:
        """Normalize provider accounting to Pi's cacheRead/cacheWrite meaning.

        ``cacheRead`` means tokens reused from a *previous* request;
        ``cacheWrite`` means tokens added by this request.  LangChain's current
        OpenAI parser already exposes those as ``cache_read`` and
        ``cache_creation``.  The raw-response fallbacks keep the same public
        contract for official integrations that expose only provider fields.
        """

        usage = getattr(response, "usage_metadata", None)
        response_metadata = getattr(response, "response_metadata", None)
        if not isinstance(response_metadata, dict):
            response_metadata = {}
        token_usage = response_metadata.get("token_usage")
        if not isinstance(token_usage, dict):
            token_usage = response_metadata.get("usage")
        if not isinstance(token_usage, dict):
            token_usage = {}
        prompt_details = token_usage.get("prompt_tokens_details")
        if not isinstance(prompt_details, dict):
            prompt_details = token_usage.get("input_tokens_details")
        if not isinstance(prompt_details, dict):
            prompt_details = {}
        input_details = usage.get("input_token_details") if isinstance(usage, dict) else None
        if not isinstance(input_details, dict):
            input_details = {}

        cache_read = self._first_usage_value(
            (input_details, usage, prompt_details, token_usage),
            (
                "cacheRead",
                "cache_read",
                "priority_cache_read",
                "flex_cache_read",
                "cached_tokens",
                "prompt_cache_hit_tokens",
            ),
        )
        cache_write = self._first_usage_value(
            (input_details, usage, prompt_details, token_usage),
            (
                "cacheWrite",
                "cache_creation",
                "priority_cache_creation",
                "flex_cache_creation",
                "cache_write_tokens",
                "cache_write",
            ),
        )
        return cache_read, cache_write

    def record_response(self, response: Any, *, cache_scope: str) -> Any:
        """Attach normalized counters and retain them under one role scope."""

        cache_read, cache_write = self._extract_cache_usage(response)
        scope = self._cache_scope_name(cache_scope)
        with self._cache_usage_lock:
            counters = self._cache_usage_by_scope.setdefault(
                scope, {"cacheRead": 0, "cacheWrite": 0}
            )
            counters["cacheRead"] = int(counters["cacheRead"]) + cache_read
            counters["cacheWrite"] = int(counters["cacheWrite"]) + cache_write
        # The raw provider fields remain available to LangChain.  This small,
        # provider-neutral supplement is the only cache interface visible to
        # graph/trace consumers.
        if isinstance(response, AIMessage):
            usage = dict(response.usage_metadata or {})
            usage["cacheRead"] = cache_read
            usage["cacheWrite"] = cache_write
            response.usage_metadata = usage
        return response

    def cache_usage(self) -> JSON:
        """Return only normalized cache counters, grouped by independent role."""

        with self._cache_usage_lock:
            by_scope = {
                scope: {
                    "cacheRead": int(counters["cacheRead"]),
                    "cacheWrite": int(counters["cacheWrite"]),
                }
                for scope, counters in sorted(self._cache_usage_by_scope.items())
            }
        return {
            "cacheRead": sum(item["cacheRead"] for item in by_scope.values()),
            "cacheWrite": sum(item["cacheWrite"] for item in by_scope.values()),
            "byScope": by_scope,
        }

    def structured(
        self,
        schema: type,
        *,
        cache_scope: str = "unscoped",
    ) -> Runnable[Any, Any]:
        scoped_model = self._scoped_model(cache_scope)
        runnable = scoped_model.with_structured_output(
            schema,
            **{**self.structured_output_kwargs, "include_raw": True},
        )
        return runnable | RunnableLambda(
            lambda result: _unwrap_structured_result(
                self.record_response(result.get("raw") if isinstance(result, dict) else None, cache_scope=cache_scope),
                schema,
            )
        )

    def invoke_tools(
        self,
        tools: Sequence[TechnicalTool],
        messages: Sequence[AnyMessage],
        *,
        cache_scope: str,
        tool_choice: str,
    ) -> AIMessage:
        bind_kwargs = dict(self.bind_tools_kwargs)
        bind_kwargs.pop("tool_choice", None)
        response = self._scoped_model(cache_scope).bind_tools(
            list(tools), tool_choice=tool_choice, **bind_kwargs
        ).invoke(list(messages))
        if not isinstance(response, AIMessage):
            raise TypeError("R tool-bound model must return AIMessage")
        return cast(AIMessage, self.record_response(response, cache_scope=cache_scope))


def make_deepseek_adapter(
    *,
    api_key: str | None = None,
    model: str | None = None,
    api_base: str = "https://api.deepseek.com/v1",
) -> LangChainChatAdapter:
    from langchain_deepseek import ChatDeepSeek

    resolved_model = model or os.getenv("DEEPSEEK_MODEL")
    if not resolved_model:
        raise ValueError("pass model=... or set DEEPSEEK_MODEL")
    chat_model = ChatDeepSeek(
        model=resolved_model,
        api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
        api_base=api_base,
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=MODEL_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return LangChainChatAdapter(
        chat_model,
        structured_output_kwargs={"method": "function_calling", "strict": True},
        bind_tools_kwargs={"strict": True},
        provider="deepseek",
    )


def make_openai_compatible_adapter(
    provider: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    api_base: str | None = None,
    strict_function_calling: bool | None = None,
) -> LangChainChatAdapter:
    """Wrap an OpenAI-compatible provider with LangChain's ChatOpenAI.

    This deliberately does not implement a provider REST client. GLM, Qwen,
    and MiMo publish Chat Completions-compatible endpoints; ChatOpenAI owns the
    OpenAI protocol, message serialization, structured-output parsing, and
    standard ``AIMessage.tool_calls`` conversion used by the graph.
    """

    resolved = resolve_provider_name(provider)
    if resolved == "deepseek":
        raise ValueError("use make_deepseek_adapter() for provider='deepseek'")
    configuration = OPENAI_COMPATIBLE_PROVIDERS[resolved]
    from langchain_openai import ChatOpenAI

    strict = (
        configuration.strict_function_calling
        if strict_function_calling is None
        else strict_function_calling
    )
    chat_model = ChatOpenAI(
        model=resolve_provider_model(resolved, model),
        api_key=resolve_provider_api_key(resolved, api_key=api_key),
        base_url=api_base or configuration.api_base,
        temperature=0,
        max_tokens=provider_max_output_tokens(resolved),
        timeout=provider_request_timeout_seconds(resolved),
        max_retries=0,
    )
    return LangChainChatAdapter(
        chat_model,
        structured_output_kwargs={"method": "function_calling", "strict": strict},
        bind_tools_kwargs={"strict": strict},
        provider=resolved,
        supports_prompt_cache_key=configuration.supports_prompt_cache_key,
        cache_read_includes_write=configuration.cache_read_includes_write,
    )


def make_model_adapter(
    provider: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    api_base: str | None = None,
    strict_function_calling: bool | None = None,
) -> LangChainChatAdapter:
    """Create the selected LangChain adapter without changing graph code."""

    resolved = resolve_provider_name(provider)
    if resolved == "deepseek":
        if strict_function_calling not in {None, True}:
            raise ValueError("DeepSeek adapter requires strict function calling")
        return make_deepseek_adapter(
            api_key=resolve_provider_api_key(resolved, api_key=api_key),
            model=resolve_provider_model(resolved, model),
            api_base=api_base or provider_api_base(resolved),
        )
    return make_openai_compatible_adapter(
        resolved,
        api_key=api_key,
        model=model,
        api_base=api_base,
        strict_function_calling=strict_function_calling,
    )


def _merge_records(left: list[JSON], right: list[JSON]) -> list[JSON]:
    merged: list[JSON] = []
    positions: dict[str, int] = {}
    for record in [*left, *right]:
        record_id = next(
            (
                str(record.get(key))
                for key in (
                    "record_id",
                    "packet_id",
                    "report_id",
                    "department_conclusion_id",
                    "archive_id",
                )
                if record.get(key)
            ),
            "",
        )
        if not record_id:
            canonical = json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            record_id = f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
        if record_id in positions:
            merged[positions[record_id]] = record
        else:
            positions[record_id] = len(merged)
            merged.append(record)
    return merged


class DeepResearchState(TypedDict, total=False):
    question: str
    constitution: InquiryConstitution
    u_snapshot: EpistemicSnapshot
    selected_item_package: LocalEpistemicPackage | None
    technical_task: TechnicalTask | None
    researcher_claim: RTaskClaim | None
    researcher_plan: RExecutionPlan | None
    researcher_pending_plan: RPlanProposal | None
    department_plan_review: DPlanReview | None
    researcher_progress_decision: RProgressDecision | None
    researcher_completed_stage_ids: list[str]
    researcher_active_stage_id: str | None
    researcher_stage_results: Annotated[list[RStageResult], _merge_records]
    messages: Annotated[list[AnyMessage], add_messages]
    objective_ledger: Annotated[list[JSON], _merge_records]
    evidence_ledger: Annotated[list[TechnicalPacket], _merge_records]
    report_ledger: Annotated[list[DepartmentReport], _merge_records]
    archive_index: Annotated[list[JSON], _merge_records]
    technical_packet: TechnicalPacket | None
    department_report: DepartmentReport | None
    final_answer: str


class ResearcherGraphState(TypedDict, total=False):
    constitution: InquiryConstitution
    technical_task: TechnicalTask
    researcher_claim: RTaskClaim | None
    researcher_plan: RExecutionPlan | None
    researcher_pending_plan: RPlanProposal | None
    department_plan_review: DPlanReview | None
    researcher_progress_decision: RProgressDecision | None
    researcher_completed_stage_ids: list[str]
    researcher_active_stage_id: str | None
    researcher_stage_results: Annotated[list[RStageResult], _merge_records]
    messages: Annotated[list[AnyMessage], add_messages]
    objective_ledger: Annotated[list[JSON], _merge_records]
    evidence_ledger: Annotated[list[TechnicalPacket], _merge_records]
    report_ledger: Annotated[list[DepartmentReport], _merge_records]
    archive_index: Annotated[list[JSON], _merge_records]
    technical_packet: TechnicalPacket | None


class ResearchInput(TypedDict):
    question: str


class ResearchOutput(TypedDict):
    final_answer: str


class InterpretationInput(TypedDict):
    question: str


class PlannerInput(TypedDict):
    constitution: InquiryConstitution
    u_snapshot: EpistemicSnapshot


class PlannerRouteInput(TypedDict, total=False):
    selected_item_package: LocalEpistemicPackage | None


class DepartmentInput(TypedDict, total=False):
    constitution: InquiryConstitution
    selected_item_package: LocalEpistemicPackage
    technical_task: TechnicalTask | None
    technical_packet: TechnicalPacket | None
    evidence_ledger: list[TechnicalPacket]


class DepartmentRouteInput(TypedDict, total=False):
    technical_task: TechnicalTask | None
    department_report: DepartmentReport | None


class RegisteredTableDescription(TypedDict):
    table_id: str
    description: str


class RegisteredDataSourceDescription(TypedDict):
    source_id: str
    name: str
    description: str
    tables: list[RegisteredTableDescription]


class UpdaterInput(TypedDict, total=False):
    constitution: InquiryConstitution
    u_snapshot: EpistemicSnapshot
    department_report: DepartmentReport


class WriterInput(TypedDict):
    constitution: InquiryConstitution
    u_snapshot: EpistemicSnapshot


class RuntimeContext(TypedDict):
    model: ModelAdapter



class ArgumentTreeInput(TypedDict):
    question: str


class ArgumentTreeOutput(TypedDict):
    final_answer: str


class ArgumentTreeState(TypedDict, total=False):
    question: str
    constitution: InquiryConstitution
    cognitive_items: list[CognitiveItem]
    active_cognitive_item: CognitiveItem | None
    technical_items: list[TechnicalItem]
    technical_task: TechnicalTask | None
    researcher_claim: RTaskClaim | None
    researcher_plan: RExecutionPlan | None
    researcher_pending_plan: RPlanProposal | None
    department_plan_review: DPlanReview | None
    researcher_progress_decision: RProgressDecision | None
    researcher_completed_stage_ids: list[str]
    researcher_active_stage_id: str | None
    researcher_stage_results: Annotated[list[RStageResult], _merge_records]
    messages: Annotated[list[AnyMessage], add_messages]
    objective_ledger: Annotated[list[JSON], _merge_records]
    evidence_ledger: Annotated[list[TechnicalPacket], _merge_records]
    department_conclusions: Annotated[list[DepartmentConclusion], _merge_records]
    archive_index: Annotated[list[JSON], _merge_records]
    technical_packet: TechnicalPacket | None
    epistemic_conclusion: EpistemicConclusionOutput | None
    matrix_ready: bool
    final_answer: str


class CognitiveProjectInput(TypedDict):
    constitution: InquiryConstitution
    cognitive_item: CognitiveItem


class CognitiveProjectOutput(TypedDict, total=False):
    evidence_ledger: list[TechnicalPacket]
    department_conclusions: list[DepartmentConclusion]
    archive_index: list[JSON]


class CognitiveProjectState(TypedDict, total=False):
    constitution: InquiryConstitution
    cognitive_item: CognitiveItem
    technical_items: list[TechnicalItem]
    technical_task: TechnicalTask | None
    researcher_claim: RTaskClaim | None
    researcher_plan: RExecutionPlan | None
    researcher_pending_plan: RPlanProposal | None
    department_plan_review: DPlanReview | None
    researcher_progress_decision: RProgressDecision | None
    researcher_completed_stage_ids: list[str]
    researcher_active_stage_id: str | None
    researcher_stage_results: Annotated[list[RStageResult], _merge_records]
    messages: Annotated[list[AnyMessage], add_messages]
    objective_ledger: Annotated[list[JSON], _merge_records]
    evidence_ledger: Annotated[list[TechnicalPacket], _merge_records]
    department_conclusions: Annotated[list[DepartmentConclusion], _merge_records]
    archive_index: Annotated[list[JSON], _merge_records]
    technical_packet: TechnicalPacket | None


ORGANIZATION_BOUNDARY_FEWSHOT = """
Fixed hierarchy example — I fixes the research object and boundary. U first expands candidate explanations into a finite C1/C2/C3 set of independent empirical assessment projects. A C asks how one explanation should be assessed, what observation would support it, what would count against it, and what evidence would distinguish it; it never says to prove that explanation. C projects run concurrently. For one C, D first creates a finite T1/T2/T3 technical tree; T items may run concurrently and collectively assess support, counterevidence, discrimination, or robustness. R chooses tools, queries, algorithms, parameters, and its local execution DAG inside each T. D returns a per-explanation assessment. Only when every C returns does U normalize them into one common evidence matrix. W writes the final adjudication.
Counterexamples: U says "run KDE with bandwidth 20"; D says "collect evidence proving explanation A"; R adds an unrelated lithology project; W treats a failed tool retry as evidence. Each crosses a fixed authority boundary.
"""


INTERPRETATION_PROMPT = """
You are boundary-registration layer I. Record the central research object, explicit user boundaries, and requested output. Resolve references only enough to make those entries unambiguous. Return the constitution fields.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

UPDATER_BOOTSTRAP_PROMPT = """
You are the U layer of a fixed scientific argument tree. Your first and only planning action is to identify the finite set of material competing explanations for the total question, then expand them into parallel C projects. Each C independently evaluates one explanation; it must not be framed as proving or defending it. Do not design technical plans, choose data, tools, algorithms, parameters, or execution steps. The C tree freezes after you return it.

For each C, state the explanation under evaluation, the local assessment question, its independent scientific role, predicted observations that would support it, counterevidence signals that would weaken it, required empirical evidence, and decision target. Avoid duplicate explanations and omit candidates that cannot materially alter the final scientific judgment.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

DEPARTMENT_TASK_PROMPT = """
You are D beneath one frozen U explanation-assessment project C. First and only once, decompose C into a finite fixed technical tree T. Each T must have a distinct empirical role: support, counterevidence, discrimination, or robustness. The collective task is to assess the explanation's empirical support, counterevidence, and discriminative value — never to prove it. Define each T's question, scientific role, assessment dimension, and bounded task fields. Do not revise C, make an overall U-level judgment, or prescribe R's exact tools, algorithms, parameters, or execution order. The T tree freezes after you return it.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

RESEARCHER_CLAIM_PROMPT = """
You are technical execution layer R. Accept the presented D-issued task by returning its task_id exactly; this binds the episode to that task.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

RESEARCHER_PLAN_PROMPT = """
You are technical execution layer R. Propose a finite local execution DAG for the claimed task. Propose a replacement only when observations require a topology change. Each stage is one coherent operation objective and may use one call or a small parallel tool batch. Dependencies express local data flow. Stay within the task boundary and explain why the candidate is needed; D must approve it before execution.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

D_PLAN_REVIEW_PROMPT = """
You are D's technical control function. Approve an R-authored DAG that stays inside the issued task; reject one that changes or expands the task. Return only the decision and boundary findings. Do not create, repair, reorder, or parameterize R's DAG.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

RESEARCHER_EXECUTE_PROMPT = """
You are technical execution layer R. Execute the current ready stage of the approved DAG with one registered tool call or a small parallel batch. Use only the supplied dependency-stage results and remain inside the D-issued task. The full R execution transcript is an audit record, not your working context; do not assume facts beyond this stage packet and its DAG dependencies.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

RESEARCHER_PROGRESS_PROMPT = """
You are technical execution layer R. Observe the latest completed DAG wave through its stage artifacts. Continue when the remaining approved stages still fit; propose a patch only when DAG topology must change; submit when results suffice for D or an escalation condition is reached. Tool, query, algorithm, parameter, ordering, or retry changes inside an existing stage are not DAG changes. A patch cannot execute before D approval.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

RESEARCHER_FINALIZE_PROMPT = """
You are technical execution layer R. Return only the obtained values, observations, calculations, and execution failures that address the task's required outputs. Do not decide U's proposition status or copy tool-call identifiers; code attaches provenance.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

DEPARTMENT_REPORT_PROMPT = """
You are D after all T packets beneath one frozen explanation-assessment project C have returned. Produce a structured Department Conclusion: one technical conclusion for every fixed T, then assess the explanation's supporting evidence, counterevidence, and genuinely discriminative evidence. Report an assessment status; do not convert a result into advocacy for the explanation. Preserve only technical facts that can affect scientific judgment: data scope, selection conditions, sample size, grouping criteria, method, key parameters, thresholds, core statistics, uncertainty, and robustness checks. Exclude retries, failed calls, Python variables, query order, and unrelated logs. Do not update C, decide the overall scientific interpretation, or invent evidence identifiers.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

U_CONCLUSION_PROMPT = """
You are U after every frozen C has a Department Conclusion. Normalize every
returned explanation assessment into one common evidence matrix, not a revised
plan or final verdict. Each matrix row must retain support, counterevidence,
discriminative evidence, non-discriminative evidence, independent evidence
chains, uncertainty, scope, and an assessment status. Normalize differences in
data scope, method, and evidential independence explicitly rather than treating
all positive results as commensurable. Record cross-explanation conflicts and
unresolved items. Leave W an accurate common evidence space; do not decide the
final overall claim ceiling, create C or T items, alter frozen trees, add
evidence, or prescribe work.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

WRITER_PROMPT = """
You are W. Make the final scientific adjudication only from U's normalized evidence matrix. Weigh support, counterevidence, discriminative power, independence, conflicts, unresolved issues, and scope; write the overall claim ceiling here. Do not generate evidence, modify a matrix row, or elevate a technical parameter into a scientific claim by yourself.
""" + ORGANIZATION_BOUNDARY_FEWSHOT


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _cache_ordered_context(
    stable_prefix: Mapping[str, Any],
    dynamic_suffix: Mapping[str, Any],
) -> JSON:
    """Return one semantic packet with an intentionally stable JSON prefix.

    This is not a second state or a cache node.  It preserves the exact data
    contract, but serializes task/role-constant material before per-stage
    artifacts.  GLM's documented implicit context cache can then reuse the
    common prefix across independent R DAG workers; providers with an explicit
    affinity key receive the same packet through ``LangChainChatAdapter``.
    """

    overlap = set(stable_prefix).intersection(dynamic_suffix)
    if overlap:
        raise ValueError(
            "cache packet cannot place one field in both stable and dynamic "
            f"sections: {', '.join(sorted(overlap))}"
        )
    return {
        **copy.deepcopy(dict(stable_prefix)),
        **copy.deepcopy(dict(dynamic_suffix)),
    }


def _policy_projection(
    policy: EpistemicPolicy,
    role: str,
) -> JSON:
    expected = json.dumps(
        BASE_EPISTEMIC_POLICY,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    actual = json.dumps(
        policy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if actual != expected:
        raise ValueError(
            "constitution epistemic_policy drifted from BASE_EPISTEMIC_POLICY"
        )
    permission = policy["role_permissions"].get(role)
    if permission is None:
        raise ValueError(f"no epistemic policy projection for role {role!r}")
    return {
        "policy_id": policy["policy_id"],
        "principle": policy["principle"],
        "few_shot": copy.deepcopy(policy["few_shot"]),
        "role": role,
        "role_permission": permission,
    }


def _constitution_policy_projection(
    constitution: InquiryConstitution | None,
    role: str,
) -> JSON:
    policy = BASE_EPISTEMIC_POLICY if constitution is None else constitution.get(
        "epistemic_policy"
    )
    if not isinstance(policy, dict):
        raise ValueError("constitution has no epistemic_policy")
    return _policy_projection(cast(EpistemicPolicy, policy), role)


def _prompt_with_epistemic_policy(
    prompt: str,
    constitution: InquiryConstitution | None,
    role: str,
) -> str:
    projection = _constitution_policy_projection(constitution, role)
    return (
        f"{prompt}\n\n"
        "常驻成文法则（只读；由代码维护，模型不得修改）：\n"
        f"{_json(projection)}"
    )


def _constitution_digest(constitution: InquiryConstitution) -> str:
    canonical = json.dumps(
        constitution,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _project_constitution_reference(
    container: JSON,
    constitution: InquiryConstitution,
    *,
    owner: str,
) -> JSON:
    embedded = container.get("constitution")
    if not isinstance(embedded, dict):
        raise ValueError(f"{owner} has no independently stored constitution")
    expected = _constitution_digest(constitution)
    actual = _constitution_digest(cast(InquiryConstitution, embedded))
    if actual != expected:
        raise ValueError(
            f"{owner} constitution drifted from the canonical task constitution: "
            f"{actual} != {expected}"
        )
    projected = copy.deepcopy(container)
    projected.pop("constitution", None)
    projected["constitution_ref"] = expected
    return projected


def _structured(
    adapter: ModelAdapter,
    schema: type,
    prompt: str,
    context: Any,
    *,
    policy_role: str | None = None,
) -> JSON:
    if policy_role is not None:
        constitution = (
            context.get("constitution")
            if isinstance(context, dict)
            else None
        )
        prompt = _prompt_with_epistemic_policy(
            prompt,
            cast(InquiryConstitution, constitution)
            if isinstance(constitution, dict)
            else None,
            policy_role,
        )
    # Every semantic role receives a stable, independent cache scope.  The
    # adapter decides whether the selected provider has a request-level cache
    # affinity mechanism; graph code never carries provider-specific fields.
    result = adapter.structured(
        schema,
        cache_scope=policy_role or "unscoped",
    ).invoke(
        [SystemMessage(content=prompt), HumanMessage(content=_json(context))]
    )
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    if not isinstance(result, dict):
        raise TypeError("structured model adapter must return a mapping")
    _validate_structured_schema(result, schema)
    return cast(JSON, result)


def _validate_structured_schema(result: JSON, schema: type) -> None:
    """Validate required fields at the common structured-output boundary."""

    parameters = convert_to_openai_tool(schema, strict=True)["function"]["parameters"]

    def walk(value: Any, shape: Any, path: str) -> None:
        if not isinstance(shape, dict):
            return
        if shape.get("type") == "object":
            if not isinstance(value, dict):
                return
            properties = shape.get("properties", {})
            required = shape.get("required", [])
            missing = [key for key in required if key not in value]
            if missing:
                missing_path = ", ".join(f"{path}.{key}" for key in missing)
                detail = f"missing required field(s): {missing_path}"
                raise StructuredOutputFailure(
                    category="schema_contract",
                    schema_name=getattr(schema, "__name__", str(schema)),
                    raw_response=result,
                    parsing_error=ValueError(detail),
                )
            for key, property_shape in properties.items():
                if key in value:
                    walk(value[key], property_shape, f"{path}.{key}")
        elif shape.get("type") == "array" and isinstance(value, list):
            item_shape = shape.get("items")
            for index, item in enumerate(value):
                walk(item, item_shape, f"{path}[{index}]")

    walk(result, parameters, getattr(schema, "__name__", str(schema)))


def _materialize_local_epistemic_package(
    snapshot: EpistemicSnapshot,
    item: EpistemicItem,
) -> LocalEpistemicPackage:
    return {"item": copy.deepcopy(item), "constitution": snapshot["constitution"]}


def _r_capability_envelope(
    technical_tools: Sequence[TechnicalTool],
) -> RCapabilityEnvelope:
    has_data = False
    has_computation = False
    has_temporary_python = False
    for technical_tool in technical_tools:
        metadata = getattr(technical_tool, "metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        tool_name = str(convert_to_openai_tool(technical_tool)["function"].get("name") or "")
        if str(metadata.get("native_tool_id") or "") == "data.tool" or tool_name == "data__tool":
            has_data = True
        elif (
            str(metadata.get("native_tool_id") or "") == "sandbox.python"
            or tool_name == "sandbox__python"
        ):
            has_temporary_python = True
        elif str(metadata.get("capability_group") or "") != "execution_archive":
            has_computation = True
    capabilities: list[str] = []
    if has_data:
        capabilities.append("read registered tables and perform read-only SQL-style data operations")
    if has_computation:
        capabilities.append("perform registered mathematical or statistical computations")
    if has_temporary_python:
        capabilities.append(
            "create auditable run-scoped Python operations for missing atomic capabilities"
        )
    return {
        "summary": (
            "R can " + " and ".join(capabilities) + "."
            if capabilities
            else "R has no registered capability for new technical work."
        )
    }


def _tool_catalogue(technical_tools: Sequence[TechnicalTool]) -> list[JSON]:
    return [
        {
            "name": str(convert_to_openai_tool(item)["function"].get("name") or ""),
            "description": str(
                convert_to_openai_tool(item)["function"].get("description") or ""
            ),
        }
        for item in technical_tools
    ]


def _clean_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _materialize_epistemic_snapshot(
    *,
    constitution: InquiryConstitution,
    update: JSON,
    prior: EpistemicSnapshot | None,
    report: DepartmentReport | None,
) -> EpistemicSnapshot:
    draft_map = cast(JSON, update["epistemic_map"])
    prior_refs = {
        item["item_id"]: list(item.get("evidence_refs", []))
        for item in (prior["epistemic_map"]["items"] if prior else [])
    }
    report_refs = list(
        dict.fromkeys(
            reference
            for evidence in (report.get("core_evidence", []) if report else [])
            for reference in evidence.get("evidence_refs", [])
        )
    )
    materialized_items: list[EpistemicItem] = []
    for raw_item in draft_map["items"]:
        item = cast(JSON, raw_item)
        item_id = str(item["item_id"])
        evidence_refs = list(prior_refs.get(item_id, []))
        if report is not None and item_id == report["item_id"]:
            evidence_refs = list(dict.fromkeys([*evidence_refs, *report_refs]))
        materialized_items.append(
            cast(EpistemicItem, {**copy.deepcopy(item), "evidence_refs": evidence_refs})
        )
    if report is not None and report["item_id"] not in {
        item["item_id"] for item in materialized_items
    }:
        raise ValueError("U must retain the reported epistemic item identity")
    if any(item["status"] == "open" for item in materialized_items):
        decision: Literal["close", "continue", "restructure"] = "continue"
    else:
        prior_item_ids = {
            item["item_id"] for item in prior["epistemic_map"]["items"]
        } if prior else set()
        introduced_new_proposition = prior is not None and any(
            item["item_id"] not in prior_item_ids for item in materialized_items
        )
        decision = "restructure" if introduced_new_proposition else "close"
    return {
        "constitution": constitution if prior is None else prior["constitution"],
        "decision": decision,
        "epistemic_map": {
            "revision": int(draft_map["revision"]),
            "items": materialized_items,
            "overall_state": copy.deepcopy(draft_map["overall_state"]),
        },
    }


def _validate_epistemic_snapshot(
    snapshot: EpistemicSnapshot,
    old_revision: int | None,
) -> None:
    revision = int(snapshot["epistemic_map"]["revision"])
    if old_revision is None and revision != 0:
        raise ValueError("U bootstrap must publish revision 0")
    if old_revision is not None and revision <= old_revision:
        raise ValueError("U must advance the epistemic map revision")
    items = snapshot["epistemic_map"]["items"]
    ids = [str(item["item_id"]) for item in items]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("U must publish non-empty epistemic items with unique item_id values")


def _validate_execution_plan(plan: RExecutionPlan, task: TechnicalTask) -> None:
    if plan["task_id"] != task["task_id"]:
        raise ValueError("R execution plan does not belong to the claimed task_id")
    stages = plan["stages"]
    if not stages:
        raise ValueError("R execution plan must contain at least one stage")
    ids = [str(stage["stage_id"]) for stage in stages]
    if any(not value.strip() for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("R plan stage_id values must be non-empty and unique")
    known = set(ids)
    dependencies = {
        str(stage["stage_id"]): {str(value) for value in stage["depends_on"]}
        for stage in stages
    }
    if any(stage_id in deps or not deps <= known for stage_id, deps in dependencies.items()):
        raise ValueError("R plan dependencies must reference other stages in the same DAG")
    completed: set[str] = set()
    while len(completed) < len(stages):
        ready = {
            stage_id
            for stage_id, deps in dependencies.items()
            if stage_id not in completed and deps <= completed
        }
        if not ready:
            raise ValueError("R execution plan must be acyclic")
        completed.update(ready)


def _validate_plan_proposal(
    proposal: RPlanProposal,
    task: TechnicalTask,
    authorized_plan: RExecutionPlan | None,
    completed_stage_ids: Sequence[str],
) -> None:
    candidate = proposal["candidate_plan"]
    _validate_execution_plan(candidate, task)
    expected_kind = "initial" if authorized_plan is None else "patch"
    if proposal["proposal_kind"] != expected_kind:
        raise ValueError(f"R must submit an {expected_kind} plan proposal")
    expected_revision = 0 if authorized_plan is None else authorized_plan["revision"] + 1
    if candidate["revision"] != expected_revision:
        raise ValueError("R candidate plan revision is inconsistent with the authorized DAG")
    if authorized_plan is None:
        return
    previous = {stage["stage_id"]: stage for stage in authorized_plan["stages"]}
    candidate_by_id = {stage["stage_id"]: stage for stage in candidate["stages"]}
    for stage_id in completed_stage_ids:
        if stage_id not in previous or candidate_by_id.get(stage_id) != previous[stage_id]:
            raise ValueError("R cannot remove or rewrite a completed DAG stage")


def _next_ready_stage(
    plan: RExecutionPlan,
    completed_stage_ids: Sequence[str],
) -> RLocalStage | None:
    ready = _ready_stages(plan, completed_stage_ids)
    return ready[0] if ready else None


def _ready_stages(
    plan: RExecutionPlan,
    completed_stage_ids: Sequence[str],
) -> list[RLocalStage]:
    """Return every dependency-ready stage, preserving DAG order only for display."""

    completed = set(completed_stage_ids)
    return [
        copy.deepcopy(stage)
        for stage in plan["stages"]
        if stage["stage_id"] not in completed
        and set(stage["depends_on"]) <= completed
    ]


def _dag_stage_closure(
    plan: RExecutionPlan,
    stage_ids: Sequence[str],
) -> set[str]:
    stages = {str(stage["stage_id"]): stage for stage in plan["stages"]}
    closure: set[str] = set()
    pending = [str(stage_id) for stage_id in stage_ids if stage_id in stages]
    while pending:
        stage_id = pending.pop()
        if stage_id in closure:
            continue
        closure.add(stage_id)
        pending.extend(str(item) for item in stages[stage_id]["depends_on"])
    return closure


def _terminal_stage_ids(plan: RExecutionPlan) -> set[str]:
    depended_on = {
        str(dependency)
        for stage in plan["stages"]
        for dependency in stage["depends_on"]
    }
    return {
        str(stage["stage_id"])
        for stage in plan["stages"]
        if str(stage["stage_id"]) not in depended_on
    }


def _bounded_observation(value: Any, *, maximum_characters: int = 12_000) -> Any:
    """Keep decision-relevant tool output without replaying raw execution history."""

    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value
    else:
        decoded = copy.deepcopy(value)
    if isinstance(decoded, dict):
        preferred = {
            key: decoded[key]
            for key in (
                "summary",
                "findings",
                "calculation_status",
                "uncertainty",
                "limitations",
                "warnings",
                "data_operation",
            )
            if key in decoded
        }
        if preferred:
            decoded = preferred
    encoded = json.dumps(decoded, ensure_ascii=False, default=str, sort_keys=True)
    if len(encoded) <= maximum_characters:
        return decoded
    return {
        "truncated": True,
        "original_characters": len(encoded),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "preview": encoded[:maximum_characters],
    }


def _stage_result_from_tool_messages(
    *,
    task_id: str,
    plan_revision: int,
    stage_id: str,
    messages: Sequence[ToolMessage],
) -> RStageResult:
    observations = [
        {
            "tool": str(message.name or ""),
            "tool_call_id": str(message.tool_call_id or ""),
            "status": message.status,
            "result": _bounded_observation(message.content),
        }
        for message in messages
    ]
    return {
        "record_id": f"r-stage:{task_id}:r{plan_revision}:{stage_id}",
        "stage_id": stage_id,
        "status": "failed" if any(message.status == "error" for message in messages) else "completed",
        "tool_call_ids": [
            str(message.tool_call_id or "")
            for message in messages
            if str(message.tool_call_id or "")
        ],
        "observations": observations,
    }


def _legacy_stage_results(state: ResearcherGraphState) -> list[RStageResult]:
    """One-way bridge for checkpoints produced by the removed serial R path."""

    plan = state.get("researcher_plan")
    task = state.get("technical_task")
    if plan is None or task is None:
        return []
    stage_sequence = list(state.get("researcher_completed_stage_ids", []))
    active = state.get("researcher_active_stage_id")
    if active and active not in stage_sequence:
        stage_sequence.append(active)
    tool_messages = {
        str(message.tool_call_id or ""): message
        for message in state.get("messages", [])
        if isinstance(message, ToolMessage)
    }
    results: list[RStageResult] = []
    stage_index = 0
    for message in state.get("messages", []):
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        if stage_index >= len(stage_sequence):
            break
        call_ids = [str(call.get("id") or "") for call in message.tool_calls]
        returned = [tool_messages[call_id] for call_id in call_ids if call_id in tool_messages]
        if not returned:
            continue
        stage_id = stage_sequence[stage_index]
        results.append(
            _stage_result_from_tool_messages(
                task_id=task["task_id"],
                plan_revision=plan["revision"],
                stage_id=stage_id,
                messages=returned,
            )
        )
        if set(call_ids) <= {str(item.tool_call_id or "") for item in returned}:
            stage_index += 1
    return results


def _stage_results(state: ResearcherGraphState) -> list[RStageResult]:
    existing = state.get("researcher_stage_results", [])
    return list(existing) if existing else _legacy_stage_results(state)


def _completed_stage_ids_from_results(results: Sequence[RStageResult]) -> list[str]:
    return [item["stage_id"] for item in results if item["status"] == "completed"]


def _stage_result_projection(results: Sequence[RStageResult]) -> list[JSON]:
    return [
        {
            "stage_id": item["stage_id"],
            "status": item["status"],
            "observations": copy.deepcopy(item["observations"]),
        }
        for item in results
    ]


def _dependency_stage_results(
    plan: RExecutionPlan,
    stage: RLocalStage,
    results: Sequence[RStageResult],
) -> list[RStageResult]:
    closure = _dag_stage_closure(plan, stage["depends_on"])
    by_id = {item["stage_id"]: item for item in results}
    return [by_id[stage_id] for stage_id in sorted(closure) if stage_id in by_id]


def _latest_tool_messages(messages: Sequence[AnyMessage]) -> list[ToolMessage]:
    call_ids: set[str] = set()
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.tool_calls:
            call_ids = {
                str(call.get("id") or "")
                for call in message.tool_calls
                if call.get("id")
            }
            break
    return [
        message
        for message in messages
        if isinstance(message, ToolMessage)
        and str(message.tool_call_id or "") in call_ids
    ]


class RStageWorkerInput(TypedDict):
    constitution: InquiryConstitution
    technical_task: TechnicalTask
    researcher_claim: RTaskClaim
    execution_plan: RExecutionPlan
    current_stage: RLocalStage
    dependency_stage_results: list[RStageResult]


class RStageWorkerState(RStageWorkerInput, total=False):
    messages: Annotated[list[AnyMessage], add_messages]


class RStageWorkerOutput(TypedDict):
    researcher_stage_results: list[RStageResult]


def _build_researcher_graph(
    technical_tools: Sequence[TechnicalTool],
    registered_data_sources: Sequence[RegisteredDataSourceDescription],
):
    """Build R's task claim -> finite DAG -> one call per stage subgraph."""

    tools = list(technical_tools)
    if not tools:
        raise ValueError("R requires at least one registered technical tool")
    tool_names = {
        str(convert_to_openai_tool(item)["function"]["name"])
        for item in tools
    }
    tool_catalogue = _tool_catalogue(tools)
    sources = copy.deepcopy(list(registered_data_sources))

    def execute_parallel_stage_node(
        state: RStageWorkerState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        """Execute exactly one DAG stage in an isolated worker subgraph."""

        adapter = runtime.context["model"]
        request_messages = [
            SystemMessage(
                content=_prompt_with_epistemic_policy(
                    RESEARCHER_EXECUTE_PROMPT,
                    state["constitution"],
                    "R",
                )
            ),
            HumanMessage(
                content=_json(
                    _cache_ordered_context(
                        {
                            "technical_task": state["technical_task"],
                            "task_claim": state["researcher_claim"],
                            "execution_plan": state["execution_plan"],
                            # Both are fixed for every worker in this R DAG.
                            # They must precede per-stage output so GLM's
                            # implicit prefix cache can reuse these cards.
                            "registered_data_sources": sources,
                        },
                        {
                            "current_stage": state["current_stage"],
                            "dag_dependency_stage_results": _stage_result_projection(
                                state["dependency_stage_results"]
                            ),
                        },
                    )
                )
            ),
        ]
        # Keep the provider/cache boundary in the adapter.  The compatibility
        # fallback lets older scripted test adapters remain valid without
        # teaching graph code any provider cache semantics.
        invoke_tools = getattr(adapter, "invoke_tools", None)
        if callable(invoke_tools):
            response = invoke_tools(
                tools,
                request_messages,
                cache_scope="R",
                tool_choice="required",
            )
        else:
            model = getattr(adapter, "model", None)
            if not isinstance(model, BaseChatModel):
                raise TypeError("R tool execution requires ModelAdapter.model")
            bind_kwargs = dict(getattr(adapter, "bind_tools_kwargs", {}))
            bind_kwargs.pop("tool_choice", None)
            response = model.bind_tools(tools, tool_choice="required", **bind_kwargs).invoke(
                request_messages
            )
        if not isinstance(response, AIMessage):
            raise TypeError("R tool-bound model must return AIMessage")
        if response.invalid_tool_calls:
            raise InvalidToolCallFailure(response)
        if not response.tool_calls:
            raise ValueError("R must emit at least one tool call for an executable DAG stage")
        for call in response.tool_calls:
            tool_name = str(call.get("name") or "")
            if tool_name not in tool_names:
                raise ValueError(f"R selected an unregistered tool: {tool_name}")
        return {"messages": [response]}

    def package_parallel_stage_node(state: RStageWorkerState) -> RStageWorkerOutput:
        tool_messages = _latest_tool_messages(state.get("messages", []))
        if not tool_messages:
            raise ValueError("R stage worker returned no ToolMessage")
        return {
            "researcher_stage_results": [
                _stage_result_from_tool_messages(
                    task_id=state["technical_task"]["task_id"],
                    plan_revision=state["execution_plan"]["revision"],
                    stage_id=state["current_stage"]["stage_id"],
                    messages=tool_messages,
                )
            ]
        }

    stage_worker_builder = StateGraph(
        RStageWorkerState,
        context_schema=RuntimeContext,
        input_schema=RStageWorkerInput,
        output_schema=RStageWorkerOutput,
    )
    stage_worker_builder.add_node(
        "execute", execute_parallel_stage_node, retry_policy=MODEL_RETRY_POLICY
    )
    stage_worker_builder.add_node("tools", ToolNode(tools, messages_key="messages"))
    stage_worker_builder.add_node("package", package_parallel_stage_node)
    stage_worker_builder.add_edge(START, "execute")
    stage_worker_builder.add_edge("execute", "tools")
    stage_worker_builder.add_edge("tools", "package")
    stage_worker_builder.add_edge("package", END)
    stage_worker = stage_worker_builder.compile()

    def claim_node(
        state: ResearcherGraphState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        task = state["technical_task"]
        claim = cast(
            RTaskClaim,
            _structured(
                runtime.context["model"],
                RTaskClaim,
                RESEARCHER_CLAIM_PROMPT,
                {"constitution": state["constitution"], "technical_task": task},
                policy_role="R",
            ),
        )
        if claim["task_id"] != task["task_id"]:
            raise ValueError("R must claim exactly the task_id issued by D")
        return {
            "researcher_claim": claim,
            "researcher_plan": None,
            "researcher_pending_plan": None,
            "department_plan_review": None,
            "researcher_progress_decision": None,
            "researcher_completed_stage_ids": [],
            "researcher_active_stage_id": None,
            "researcher_stage_results": [],
            "technical_packet": None,
        }

    def plan_node(
        state: ResearcherGraphState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        task = state["technical_task"]
        claim = state["researcher_claim"]
        if claim is None or claim["task_id"] != task["task_id"]:
            raise ValueError("R cannot plan before claiming the D-issued task_id")
        authorized_plan = state.get("researcher_plan")
        stage_results = _stage_results(state)
        completed = _completed_stage_ids_from_results(stage_results)
        draft = cast(
            RPlanProposalDraft,
            _structured(
                runtime.context["model"],
                RPlanProposalDraft,
                RESEARCHER_PLAN_PROMPT,
                _cache_ordered_context(
                    {
                        "constitution": state["constitution"],
                        "technical_task": task,
                        "task_claim": claim,
                        "registered_capabilities": tool_catalogue,
                        "registered_data_sources": sources,
                    },
                    {
                        "proposal_kind": (
                            "initial" if authorized_plan is None else "patch"
                        ),
                        "current_authorized_dag": authorized_plan,
                        "completed_stage_ids": completed,
                        "latest_d_review": state.get("department_plan_review"),
                        "completed_stage_artifacts": _stage_result_projection(stage_results),
                        "prior_packets_for_same_item": [
                            packet
                            for packet in state.get("evidence_ledger", [])
                            if packet["item_id"] == task["target_epistemic_item"]
                            and packet["task_id"] != task["task_id"]
                        ],
                    },
                ),
                policy_role="R",
            ),
        )
        candidate: RExecutionPlan = {
            "task_id": task["task_id"],
            "revision": 0 if authorized_plan is None else authorized_plan["revision"] + 1,
            "stages": copy.deepcopy(draft["stages"]),
        }
        proposal: RPlanProposal = {
            "proposal_kind": "initial" if authorized_plan is None else "patch",
            "reason": str(draft["reason"]).strip(),
            "candidate_plan": candidate,
        }
        if not proposal["reason"]:
            raise ValueError("R plan proposal requires a reason")
        _validate_plan_proposal(
            proposal,
            task,
            authorized_plan,
            completed,
        )
        return {
            "researcher_pending_plan": proposal,
            "department_plan_review": None,
            "researcher_progress_decision": None,
        }

    def review_plan_node(
        state: ResearcherGraphState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        proposal = state.get("researcher_pending_plan")
        if proposal is None:
            raise ValueError("D cannot review a missing R plan proposal")
        review = cast(
            DPlanReview,
            _structured(
                runtime.context["model"],
                DPlanReview,
                D_PLAN_REVIEW_PROMPT,
                {
                    "constitution": state["constitution"],
                    "technical_task": state["technical_task"],
                    "current_authorized_dag": state.get("researcher_plan"),
                    "completed_stage_ids": state.get(
                        "researcher_completed_stage_ids", []
                    ),
                    "r_plan_proposal": proposal,
                },
                policy_role="D",
            ),
        )
        update: dict[str, Any] = {"department_plan_review": review}
        if review["decision"] == "approve":
            _validate_plan_proposal(
                proposal,
                state["technical_task"],
                state.get("researcher_plan"),
                state.get("researcher_completed_stage_ids", []),
            )
            update.update(
                {
                    "researcher_plan": copy.deepcopy(proposal["candidate_plan"]),
                    "researcher_pending_plan": None,
                    "researcher_progress_decision": None,
                }
            )
        return update

    def dispatch_ready_stages(state: ResearcherGraphState) -> list[Send]:
        """Fan out one isolated worker for every dependency-ready DAG stage."""

        plan = state.get("researcher_plan")
        claim = state.get("researcher_claim")
        if plan is None or claim is None:
            raise ValueError("R cannot dispatch before D approves a claimed DAG")
        results = _stage_results(state)
        completed = _completed_stage_ids_from_results(results)
        ready = _ready_stages(plan, completed)
        if not ready:
            raise ValueError("R has unfinished stages but none are dependency-ready")
        return [
            Send(
                "stage_worker",
                {
                    "constitution": state["constitution"],
                    "technical_task": state["technical_task"],
                    "researcher_claim": claim,
                    "execution_plan": plan,
                    "current_stage": stage,
                    "dependency_stage_results": _dependency_stage_results(
                        plan, stage, results
                    ),
                    "messages": [],
                },
            )
            for stage in ready
        ]

    def route_after_plan_review(
        state: ResearcherGraphState,
    ) -> Any:
        review = state["department_plan_review"]
        if review["decision"] == "approve":
            plan = state["researcher_plan"]
            completed = set(_completed_stage_ids_from_results(_stage_results(state)))
            if all(stage["stage_id"] in completed for stage in plan["stages"]):
                return "finalize"
            return dispatch_ready_stages(state)
        proposal = state["researcher_pending_plan"]
        return (
            "plan_local_dag"
            if proposal["proposal_kind"] == "initial"
            else "review_progress"
        )

    def execute_stage_node(
        state: ResearcherGraphState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        """Compatibility landing pad for a checkpoint on the retired serial edge.

        New runs never route here.  It performs at most the one already
        checkpointed stage, without exposing rolling messages, then joins the
        new wave-based dispatcher through ``legacy_stage_bridge``.
        """

        plan = state.get("researcher_plan")
        if plan is None:
            raise ValueError("R has no execution plan")
        results = _stage_results(state)
        stage = _next_ready_stage(plan, _completed_stage_ids_from_results(results))
        if stage is None:
            raise ValueError("R has no ready stage to execute")
        claim = state.get("researcher_claim")
        if claim is None:
            raise ValueError("R cannot execute an unclaimed task")
        update = execute_parallel_stage_node(
            {
                "constitution": state["constitution"],
                "technical_task": state["technical_task"],
                "researcher_claim": claim,
                "execution_plan": plan,
                "current_stage": stage,
                "dependency_stage_results": _dependency_stage_results(
                    plan, stage, results
                ),
                "messages": [],
            },
            runtime,
        )
        return {
            "researcher_active_stage_id": stage["stage_id"],
            "messages": update["messages"],
        }

    def completed_stage_ids(state: ResearcherGraphState) -> list[str]:
        completed = set(_completed_stage_ids_from_results(_stage_results(state)))
        plan = state.get("researcher_plan")
        if plan is None:
            return []
        return [
            stage["stage_id"] for stage in plan["stages"] if stage["stage_id"] in completed
        ]

    def legacy_stage_bridge_node(state: ResearcherGraphState) -> dict[str, Any]:
        """Materialize a serial-checkpoint stage as a normal stage artifact."""

        latest_tool_messages = _latest_tool_messages(state.get("messages", []))
        if not latest_tool_messages:
            raise ValueError("legacy ToolNode returned no ToolMessage")
        plan = state["researcher_plan"]
        active = state.get("researcher_active_stage_id")
        if not active:
            raise ValueError("legacy ToolNode has no active stage identity")
        artifact = _stage_result_from_tool_messages(
            task_id=state["technical_task"]["task_id"],
            plan_revision=plan["revision"],
            stage_id=active,
            messages=latest_tool_messages,
        )
        legacy = _legacy_stage_results(state)
        # The current active call may not yet be inferable from an interrupted
        # checkpoint, so retain the explicit artifact as the last-write winner.
        return {
            "researcher_stage_results": [*legacy, artifact],
            "researcher_active_stage_id": None,
        }

    def review_progress_node(
        state: ResearcherGraphState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        plan = state.get("researcher_plan")
        if plan is None:
            raise ValueError("R cannot review progress without an authorized DAG")
        results = _stage_results(state)
        completed = _completed_stage_ids_from_results(results)
        already_reviewed = set(state.get("researcher_completed_stage_ids", []))
        latest = [
            item
            for item in results
            if item["stage_id"] not in already_reviewed
        ]
        if all(stage["stage_id"] in set(completed) for stage in plan["stages"]):
            return {
                "researcher_progress_decision": {
                    "decision": "submit",
                    "reason": "所有冻结的本地 DAG stage 均已完成；交由 R 技术结果汇总。",
                },
                "researcher_pending_plan": None,
                "researcher_completed_stage_ids": completed,
                "researcher_active_stage_id": None,
                "researcher_stage_results": results,
            }
        decision = cast(
            RProgressDecision,
            _structured(
                runtime.context["model"],
                RProgressDecision,
                RESEARCHER_PROGRESS_PROMPT,
                _cache_ordered_context(
                    {
                        "constitution": state["constitution"],
                        "technical_task": state["technical_task"],
                        "authorized_dag": plan,
                    },
                    {
                        "completed_stage_ids": completed,
                        "latest_stage_artifacts": _stage_result_projection(latest),
                        "latest_d_review": state.get("department_plan_review"),
                    },
                ),
                policy_role="R",
            ),
        )
        return {
            "researcher_progress_decision": decision,
            "researcher_pending_plan": None,
            "researcher_completed_stage_ids": completed,
            "researcher_active_stage_id": None,
            "researcher_stage_results": results,
        }

    def route_after_progress(
        state: ResearcherGraphState,
    ) -> Any:
        decision = state["researcher_progress_decision"]["decision"]
        if decision == "continue":
            return dispatch_ready_stages(state)
        if decision == "propose_patch":
            return "plan_local_dag"
        return "finalize"

    def finalize_node(
        state: ResearcherGraphState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        task = state["technical_task"]
        claim = state.get("researcher_claim")
        if claim is None or claim["task_id"] != task["task_id"]:
            raise ValueError("R cannot finalize an unclaimed task")
        stage_results = _stage_results(state)
        if not stage_results:
            raise ValueError("R must execute at least one registered technical tool")
        structured = _structured(
            runtime.context["model"],
            TechnicalResults,
            RESEARCHER_FINALIZE_PROMPT,
            _cache_ordered_context(
                {
                    "constitution": state["constitution"],
                    "technical_task": task,
                    "task_claim": claim,
                    "execution_plan": state["researcher_plan"],
                },
                {
                    "terminal_stage_artifacts": _stage_result_projection(
                        [
                            item
                            for item in stage_results
                            if item["stage_id"] in _terminal_stage_ids(
                                state["researcher_plan"]
                            )
                        ]
                    ),
                },
            ),
            policy_role="R",
        )
        results = _clean_strings(structured.get("results", []))
        if not results:
            raise ValueError("R must return at least one obtained technical result")
        tool_call_ids = list(
            dict.fromkeys(
                tool_call_id
                for stage_result in stage_results
                if stage_result["status"] == "completed"
                for tool_call_id in stage_result["tool_call_ids"]
                if tool_call_id
            )
        )
        packet_digest = hashlib.sha256(
            _json(
                {
                    "task_id": task["task_id"],
                    "results": results,
                    "tool_call_ids": tool_call_ids,
                }
            ).encode("utf-8")
        ).hexdigest()[:16]
        packet: TechnicalPacket = {
            "packet_id": f"packet-{packet_digest}",
            "task_id": task["task_id"],
            "item_id": task["target_epistemic_item"],
            "results": results,
            "tool_call_ids": tool_call_ids,
        }
        return {
            "technical_packet": packet,
            "researcher_completed_stage_ids": completed_stage_ids(state),
            "researcher_active_stage_id": None,
            "evidence_ledger": [packet],
            "archive_index": [
                {
                    "archive_id": f"packet:{packet['packet_id']}",
                    "entry_kind": "technical_packet",
                    "item_id": packet["item_id"],
                    "task_id": packet["task_id"],
                    "reference_id": packet["packet_id"],
                }
            ],
        }

    builder = StateGraph(ResearcherGraphState, context_schema=RuntimeContext)
    builder.add_node("claim_task", claim_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_node("plan_local_dag", plan_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_node("department_plan_review", review_plan_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_node("stage_worker", stage_worker)
    # Retained solely as a checkpoint landing pad for runs created before the
    # parallel worker graph.  No new route enters this serial compatibility edge.
    builder.add_node("execute_stage", execute_stage_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_node("tools", ToolNode(tools, messages_key="messages"))
    builder.add_node("legacy_stage_bridge", legacy_stage_bridge_node)
    builder.add_node("review_progress", review_progress_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_node("finalize", finalize_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_edge(START, "claim_task")
    builder.add_edge("claim_task", "plan_local_dag")
    builder.add_edge("plan_local_dag", "department_plan_review")
    builder.add_conditional_edges(
        "department_plan_review",
        route_after_plan_review,
        {
            "finalize": "finalize",
            "plan_local_dag": "plan_local_dag",
            "review_progress": "review_progress",
            "stage_worker": "stage_worker",
        },
    )
    builder.add_edge("stage_worker", "review_progress")
    builder.add_edge("execute_stage", "tools")
    builder.add_edge("tools", "legacy_stage_bridge")
    builder.add_edge("legacy_stage_bridge", "review_progress")
    builder.add_conditional_edges(
        "review_progress",
        route_after_progress,
        {
            "plan_local_dag": "plan_local_dag",
            "finalize": "finalize",
            "stage_worker": "stage_worker",
        },
    )
    builder.add_edge("finalize", END)
    return builder.compile()


def _legacy_fixed_u_build_graph(
    technical_tools: Sequence[TechnicalTool],
    *,
    checkpointer: BaseCheckpointSaver,
    registered_data_sources: Sequence[RegisteredDataSourceDescription] = (),
):
    """Compile the outer epistemic graph around D's action space and R's DAG."""

    capability_envelope = _r_capability_envelope(technical_tools)
    researcher_graph = _build_researcher_graph(
        technical_tools,
        registered_data_sources,
    )

    def interpretation_node(
        state: InterpretationInput,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        output = _structured(
            runtime.context["model"],
            InterpretationOutput,
            INTERPRETATION_PROMPT,
            {"question": state["question"]},
            policy_role="I",
        )
        extracted = output["constitution"]
        constitution: InquiryConstitution = {
            "question": state["question"],
            "research_object": str(extracted["research_object"]),
            "boundaries": _clean_strings(extracted["boundaries"]),
            "requested_output": str(extracted["requested_output"]),
            "epistemic_policy": copy.deepcopy(BASE_EPISTEMIC_POLICY),
        }
        return {
            "constitution": constitution,
            "u_snapshot": None,
            "selected_item_package": None,
            "technical_task": None,
            "researcher_claim": None,
            "researcher_plan": None,
            "researcher_pending_plan": None,
            "department_plan_review": None,
            "researcher_progress_decision": None,
            "researcher_completed_stage_ids": [],
            "researcher_active_stage_id": None,
            "messages": [],
            "objective_ledger": [],
            "evidence_ledger": [],
            "report_ledger": [],
            "archive_index": [],
            "technical_packet": None,
            "department_report": None,
        }

    def updater_node(
        state: UpdaterInput,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        prior = state.get("u_snapshot")
        if prior is not None:
            raise ValueError("fixed-U graph never revises the initial U map")
        context = {"mode": "fixed-bootstrap", "constitution": state["constitution"]}
        update = _structured(
            runtime.context["model"],
            UpdaterOutput,
            UPDATER_BOOTSTRAP_PROMPT,
            context,
            policy_role="U",
        )
        output = _materialize_epistemic_snapshot(
            constitution=state["constitution"],
            update=update,
            prior=None,
            report=None,
        )
        _validate_epistemic_snapshot(output, None)
        return {
            "u_snapshot": output,
            "selected_item_package": None,
            "technical_task": None,
            "researcher_claim": None,
            "researcher_plan": None,
            "researcher_pending_plan": None,
            "department_plan_review": None,
            "researcher_progress_decision": None,
            "researcher_completed_stage_ids": [],
            "researcher_active_stage_id": None,
            "messages": [],
            "technical_packet": None,
            "department_report": None,
        }

    def fixed_selector_node(
        state: DeepResearchState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        """Contract the old P role into deterministic frozen-map traversal."""

        del runtime
        snapshot = state["u_snapshot"]
        completed_item_ids = {
            report["item_id"] for report in state.get("report_ledger", [])
        }
        selected = next(
            (
                item
                for item in snapshot["epistemic_map"]["items"]
                if item["status"] == "open" and item["item_id"] not in completed_item_ids
            ),
            None,
        )
        return {
            "selected_item_package": (
                _materialize_local_epistemic_package(snapshot, selected)
                if selected is not None
                else None
            ),
            "technical_task": None,
            "technical_packet": None,
            "department_report": None,
        }

    def planner_node(
        state: PlannerInput,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        snapshot = state["u_snapshot"]
        selection = _structured(
            runtime.context["model"],
            PlannerSelection,
            PLANNER_PROMPT,
            {
                "constitution": state["constitution"],
                "u_snapshot": _project_constitution_reference(
                    cast(JSON, snapshot),
                    state["constitution"],
                    owner="U snapshot",
                ),
            },
            policy_role="P",
        )
        selected_id = str(selection["selected_item_id"])
        selected = next(
            (
                item
                for item in snapshot["epistemic_map"]["items"]
                if item["item_id"] == selected_id and item["status"] == "open"
            ),
            None,
        )
        if selected is None:
            raise ValueError("P must select one open item_id from U's epistemic map")
        return {
            "selected_item_package": _materialize_local_epistemic_package(
                snapshot, selected
            ),
            "technical_task": None,
            "technical_packet": None,
            "department_report": None,
        }

    def department_node(
        state: DepartmentInput,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        package = state["selected_item_package"]
        item_id = str(package["item"]["item_id"])
        task = state.get("technical_task")
        packet = state.get("technical_packet")
        if packet is None:
            if task is not None:
                raise ValueError("D cannot replace an issued task before R returns")
            draft = _structured(
                runtime.context["model"],
                DepartmentTaskDraft,
                DEPARTMENT_TASK_PROMPT,
                {
                    "constitution": state["constitution"],
                    "selected_epistemic_item": _project_constitution_reference(
                        cast(JSON, package),
                        state["constitution"],
                        owner="P-selected local epistemic package",
                    ),
                    "r_capability_envelope": capability_envelope,
                },
                policy_role="D",
            )
            semantic_task: DepartmentTaskDraft = {
                "technical_objective": str(draft["technical_objective"]).strip(),
                "scope": _clean_strings(draft["scope"]),
                "required_outputs": _clean_strings(draft["required_outputs"]),
                "acceptance_criteria": _clean_strings(draft["acceptance_criteria"]),
                "coarse_stages": _clean_strings(draft["coarse_stages"]),
                "allowed_resources": _clean_strings(draft["allowed_resources"]),
                "escalation_conditions": _clean_strings(draft["escalation_conditions"]),
            }
            required_nonempty = (
                semantic_task["technical_objective"],
                semantic_task["scope"],
                semantic_task["required_outputs"],
                semantic_task["acceptance_criteria"],
                semantic_task["coarse_stages"],
            )
            if not all(required_nonempty):
                raise ValueError("D task draft is missing an executable boundary")
            prior_item_packets = [
                record
                for record in state.get("evidence_ledger", [])
                if record["item_id"] == item_id
            ]
            task_digest = hashlib.sha256(
                _json(
                    {
                        "item_id": item_id,
                        "episode_ordinal": len(prior_item_packets) + 1,
                        "task": semantic_task,
                    }
                ).encode("utf-8")
            ).hexdigest()[:16]
            task: TechnicalTask = {
                "task_id": f"task-{task_digest}",
                "target_epistemic_item": item_id,
                **semantic_task,
            }
            return {
                "technical_task": task,
                "technical_packet": None,
                "department_report": None,
                "researcher_claim": None,
                "researcher_plan": None,
                "researcher_pending_plan": None,
                "department_plan_review": None,
                "researcher_progress_decision": None,
                "researcher_completed_stage_ids": [],
                "researcher_active_stage_id": None,
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    HumanMessage(
                        id=f"technical-task:{task['task_id']}",
                        content=_json({"technical_task": task}),
                    ),
                ],
            }

        if task is None:
            raise ValueError("D cannot report a TechnicalPacket without its issued task")
        if packet["task_id"] != task["task_id"] or packet["item_id"] != item_id:
            raise ValueError("R TechnicalPacket is outside D's signed task")
        if not any(
            record["packet_id"] == packet["packet_id"]
            for record in state.get("evidence_ledger", [])
        ):
            raise ValueError("R TechnicalPacket is absent from the evidence ledger")
        results = _structured(
            runtime.context["model"],
            DepartmentResults,
            DEPARTMENT_REPORT_PROMPT,
            {
                "constitution": state["constitution"],
                "selected_epistemic_item": _project_constitution_reference(
                    cast(JSON, package),
                    state["constitution"],
                    owner="P-selected local epistemic package",
                ),
                "technical_task": task,
                "technical_packet": packet,
            },
            policy_role="D",
        )
        evidence_statements = _clean_strings(results.get("core_evidence", []))
        if not evidence_statements:
            raise ValueError("D report requires at least one core evidence statement")
        packet_refs = [packet["packet_id"]]
        report_content = {
            "what_is_known": str(results.get("what_is_known") or "").strip(),
            "core_evidence": [
                {
                    "statement": statement,
                    "source": "R TechnicalPackets",
                    "evidence_refs": packet_refs,
                }
                for statement in evidence_statements
            ],
            "conclusion_boundary": str(
                results.get("conclusion_boundary") or ""
            ).strip(),
            "remaining_uncertainty": _clean_strings(
                results.get("remaining_uncertainty", [])
            ),
            "cross_item_implications": _clean_strings(
                results.get("cross_item_implications", [])
            ),
        }
        if not report_content["what_is_known"] or not report_content["conclusion_boundary"]:
            raise ValueError("D report requires current understanding and its boundary")
        report_digest = hashlib.sha256(
            _json(
                {
                    "item_id": item_id,
                    "packet_ids": packet_refs,
                    "report": report_content,
                }
            ).encode("utf-8")
        ).hexdigest()[:16]
        report = cast(
            DepartmentReport,
            {
                "report_id": f"report-{report_digest}",
                "item_id": item_id,
                **report_content,
            },
        )
        return {
            "department_report": report,
            "report_ledger": [report],
            "archive_index": [
                {
                    "archive_id": f"report:{report['report_id']}",
                    "entry_kind": "department_report",
                    "item_id": item_id,
                    "task_id": task["task_id"],
                    "reference_id": report["report_id"],
                }
            ],
        }

    def writer_node(
        state: WriterInput,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        output = _structured(
            runtime.context["model"],
            WriterOutput,
            WRITER_PROMPT,
            {
                "constitution": state["constitution"],
                "u_snapshot": _project_constitution_reference(
                    cast(JSON, state["u_snapshot"]),
                    state["constitution"],
                    owner="publication U snapshot",
                ),
                "department_reports": state.get("report_ledger", []),
            },
            policy_role="W",
        )
        return {"final_answer": output["answer"]}

    def route_after_fixed_selector(
        state: PlannerRouteInput,
    ) -> Literal["department", "writer"]:
        return "department" if state.get("selected_item_package") is not None else "writer"

    def route_after_department(
        state: DepartmentRouteInput,
    ) -> Literal["researcher", "fixed_selector"]:
        return "fixed_selector" if state.get("department_report") is not None else "researcher"

    builder = StateGraph(
        DeepResearchState,
        context_schema=RuntimeContext,
        input_schema=ResearchInput,
        output_schema=ResearchOutput,
    )
    builder.add_node(
        "interpretation",
        interpretation_node,
        input_schema=InterpretationInput,
        retry_policy=MODEL_RETRY_POLICY,
    )
    builder.add_node(
        "updater", updater_node, input_schema=UpdaterInput, retry_policy=MODEL_RETRY_POLICY
    )
    builder.add_node("fixed_selector", fixed_selector_node)
    builder.add_node(
        "department",
        department_node,
        input_schema=DepartmentInput,
        retry_policy=MODEL_RETRY_POLICY,
    )
    builder.add_node("researcher", researcher_graph)
    builder.add_node(
        "writer", writer_node, input_schema=WriterInput, retry_policy=MODEL_RETRY_POLICY
    )
    builder.add_edge(START, "interpretation")
    builder.add_edge("interpretation", "updater")
    builder.add_edge("updater", "fixed_selector")
    builder.add_conditional_edges(
        "fixed_selector",
        route_after_fixed_selector,
        {"department": "department", "writer": "writer"},
    )
    builder.add_conditional_edges(
        "department",
        route_after_department,
        {"researcher": "researcher", "fixed_selector": "fixed_selector"},
    )
    builder.add_edge("researcher", "department")
    builder.add_edge("writer", END)
    return builder.compile(checkpointer=checkpointer)


def _require_text(value: Any, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _validate_cognitive_items(items: Sequence[CognitiveItem]) -> None:
    identifiers = [str(item.get("c_id") or "").strip() for item in items]
    if not identifiers or any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("U must return a finite non-empty C tree with unique c_id values")
    explanations: list[str] = []
    for item in items:
        explanation = _require_text(
            item.get("explanation_under_evaluation"),
            "C.explanation_under_evaluation",
        )
        explanations.append(explanation.casefold())
        _require_text(item.get("assessment_question"), "C.assessment_question")
        _require_text(item.get("scientific_role"), "C.scientific_role")
        _require_text(item.get("decision_target"), "C.decision_target")
        if not _clean_strings(item.get("predicted_observations")):
            raise ValueError("C.predicted_observations must be non-empty")
        if not _clean_strings(item.get("counterevidence_signals")):
            raise ValueError("C.counterevidence_signals must be non-empty")
        if not _clean_strings(item.get("required_evidence")):
            raise ValueError("C.required_evidence must be non-empty")
    if len(explanations) != len(set(explanations)):
        raise ValueError("U must assign each material explanation to exactly one C project")


def _materialize_technical_items(
    cognitive_item: CognitiveItem,
    output: DepartmentPlanOutput,
) -> list[TechnicalItem]:
    raw_items = output["technical_items"]
    identifiers = [str(item.get("t_id") or "").strip() for item in raw_items]
    if not identifiers or any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("D must return a finite non-empty T tree with unique t_id values")
    materialized: list[TechnicalItem] = []
    for item in raw_items:
        task = item["technical_task"]
        dimension = str(item.get("assessment_dimension") or "").strip()
        if dimension not in {"support", "counterevidence", "discrimination", "robustness"}:
            raise ValueError("T.assessment_dimension must assess support, counterevidence, discrimination, or robustness")
        required = (
            _require_text(item.get("question"), "T.question"),
            _require_text(item.get("scientific_role"), "T.scientific_role"),
            dimension,
            _require_text(task.get("technical_objective"), "T.technical_objective"),
            _clean_strings(task.get("scope")),
            _clean_strings(task.get("required_outputs")),
            _clean_strings(task.get("acceptance_criteria")),
            _clean_strings(task.get("coarse_stages")),
        )
        if not all(required):
            raise ValueError("each T requires a bounded executable technical task")
        digest = hashlib.sha256(
            _json(
                {
                    "c_id": cognitive_item["c_id"],
                    "t_id": item["t_id"],
                    "technical_task": task,
                }
            ).encode("utf-8")
        ).hexdigest()[:16]
        materialized.append(
            {
                "t_id": str(item["t_id"]),
                "question": str(item["question"]).strip(),
                "scientific_role": str(item["scientific_role"]).strip(),
                "assessment_dimension": dimension,
                "technical_task": copy.deepcopy(task),
                "task_id": f"task-{digest}",
            }
        )
    return materialized


def _task_from_technical_item(
    cognitive_item: CognitiveItem,
    technical_item: TechnicalItem,
) -> TechnicalTask:
    return {
        "task_id": technical_item["task_id"],
        "target_epistemic_item": cognitive_item["c_id"],
        **copy.deepcopy(technical_item["technical_task"]),
    }


def _serial_fixed_u_build_graph(
    technical_tools: Sequence[TechnicalTool],
    *,
    checkpointer: BaseCheckpointSaver,
    registered_data_sources: Sequence[RegisteredDataSourceDescription] = (),
):
    """Compile the fixed C -> T scientific argument tree.

    The only non-role node is ``select_technical_task``: it deterministically
    traverses D's already-frozen T tree and makes no scientific decision.
    """

    capability_envelope = _r_capability_envelope(technical_tools)
    researcher_graph = _build_researcher_graph(technical_tools, registered_data_sources)

    def interpretation_node(
        state: ArgumentTreeInput,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        output = _structured(
            runtime.context["model"],
            InterpretationOutput,
            INTERPRETATION_PROMPT,
            {"question": state["question"]},
            policy_role="I",
        )
        extracted = output["constitution"]
        constitution: InquiryConstitution = {
            "question": state["question"],
            "research_object": _require_text(extracted["research_object"], "I.research_object"),
            "boundaries": _clean_strings(extracted["boundaries"]),
            "requested_output": _require_text(extracted["requested_output"], "I.requested_output"),
            "epistemic_policy": copy.deepcopy(BASE_EPISTEMIC_POLICY),
        }
        if not constitution["boundaries"]:
            raise ValueError("I.boundaries must be non-empty")
        return {
            "constitution": constitution,
            "cognitive_items": [],
            "active_cognitive_item": None,
            "technical_items": [],
            "technical_task": None,
            "researcher_claim": None,
            "researcher_plan": None,
            "researcher_pending_plan": None,
            "department_plan_review": None,
            "researcher_progress_decision": None,
            "researcher_completed_stage_ids": [],
            "researcher_active_stage_id": None,
            "messages": [],
            "objective_ledger": [],
            "evidence_ledger": [],
            "department_conclusions": [],
            "archive_index": [],
            "technical_packet": None,
            "epistemic_conclusion": None,
        }

    def u_plan_node(
        state: ArgumentTreeState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        output = cast(
            UPlanOutput,
            _structured(
                runtime.context["model"],
                UPlanOutput,
                UPDATER_BOOTSTRAP_PROMPT,
                {"constitution": state["constitution"]},
                policy_role="U",
            ),
        )
        items = copy.deepcopy(output["cognitive_items"])
        _validate_cognitive_items(items)
        return {"cognitive_items": items}

    def department_plan_node(
        state: ArgumentTreeState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        completed_c_ids = {item["c_id"] for item in state.get("department_conclusions", [])}
        cognitive_item = next(
            (item for item in state["cognitive_items"] if item["c_id"] not in completed_c_ids),
            None,
        )
        if cognitive_item is None:
            return {"active_cognitive_item": None, "technical_items": [], "technical_task": None}
        output = cast(
            DepartmentPlanOutput,
            _structured(
                runtime.context["model"],
                DepartmentPlanOutput,
                DEPARTMENT_TASK_PROMPT,
                {
                    "constitution": state["constitution"],
                    "cognitive_item": cognitive_item,
                    "r_capability_envelope": capability_envelope,
                },
                policy_role="D",
            ),
        )
        technical_items = _materialize_technical_items(cognitive_item, output)
        return {
            "active_cognitive_item": copy.deepcopy(cognitive_item),
            "technical_items": technical_items,
            "technical_task": None,
            "technical_packet": None,
        }

    def select_technical_task_node(
        state: ArgumentTreeState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        del runtime
        cognitive_item = state.get("active_cognitive_item")
        if cognitive_item is None:
            raise ValueError("cannot traverse T tree without an active C item")
        completed_task_ids = {
            packet["task_id"]
            for packet in state.get("evidence_ledger", [])
            if packet["item_id"] == cognitive_item["c_id"]
        }
        next_item = next(
            (item for item in state["technical_items"] if item["task_id"] not in completed_task_ids),
            None,
        )
        if next_item is None:
            return {"technical_task": None}
        task = _task_from_technical_item(cognitive_item, next_item)
        return {
            "technical_task": task,
            "technical_packet": None,
            "researcher_claim": None,
            "researcher_plan": None,
            "researcher_pending_plan": None,
            "department_plan_review": None,
            "researcher_progress_decision": None,
            "researcher_completed_stage_ids": [],
            "researcher_active_stage_id": None,
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                HumanMessage(
                    id=f"technical-task:{task['task_id']}",
                    content=_json({"technical_task": task}),
                ),
            ],
        }

    def department_conclusion_node(
        state: ArgumentTreeState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        cognitive_item = state.get("active_cognitive_item")
        if cognitive_item is None:
            raise ValueError("D cannot conclude without an active C item")
        packets_by_task = {
            packet["task_id"]: packet
            for packet in state.get("evidence_ledger", [])
            if packet["item_id"] == cognitive_item["c_id"]
        }
        expected_task_ids = {item["task_id"] for item in state["technical_items"]}
        if set(packets_by_task) != expected_task_ids:
            raise ValueError("D cannot conclude before every frozen T item returns")
        output = cast(
            DepartmentConclusionOutput,
            _structured(
                runtime.context["model"],
                DepartmentConclusionOutput,
                DEPARTMENT_REPORT_PROMPT,
                {
                    "constitution": state["constitution"],
                    "cognitive_item": cognitive_item,
                    "technical_items": state["technical_items"],
                    "technical_packets": [packets_by_task[item["task_id"]] for item in state["technical_items"]],
                },
                policy_role="D",
            ),
        )
        if output["c_id"] != cognitive_item["c_id"]:
            raise ValueError("D conclusion must retain the active c_id")
        by_t_id = {entry["t_id"]: entry for entry in output["technical_conclusions"]}
        expected_t_ids = [item["t_id"] for item in state["technical_items"]]
        if set(by_t_id) != set(expected_t_ids) or len(by_t_id) != len(expected_t_ids):
            raise ValueError("D conclusion must cover every frozen T item exactly once")
        materialized: list[TechnicalConclusion] = []
        for item in state["technical_items"]:
            entry = by_t_id[item["t_id"]]
            materialized.append(
                {
                    **copy.deepcopy(entry),
                    "packet_refs": [packets_by_task[item["task_id"]]["packet_id"]],
                }
            )
        payload = {
            "c_id": cognitive_item["c_id"],
            "conclusion": output["conclusion"],
            "technical_conclusions": materialized,
        }
        digest = hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()[:16]
        conclusion: DepartmentConclusion = {
            "department_conclusion_id": f"department-conclusion-{digest}",
            "c_id": cognitive_item["c_id"],
            "conclusion": _require_text(output["conclusion"], "D.conclusion"),
            "technical_conclusions": materialized,
            "uncertainty": _clean_strings(output["uncertainty"]),
            "scope": _clean_strings(output["scope"]),
        }
        return {
            "department_conclusions": [conclusion],
            "archive_index": [
                {
                    "archive_id": f"department-conclusion:{conclusion['department_conclusion_id']}",
                    "entry_kind": "department_conclusion",
                    "item_id": cognitive_item["c_id"],
                    "reference_id": conclusion["department_conclusion_id"],
                }
            ],
            "technical_task": None,
            "technical_packet": None,
        }

    def u_conclusion_node(
        state: ArgumentTreeState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        reports = state.get("department_conclusions", [])
        expected_c_ids = [item["c_id"] for item in state["cognitive_items"]]
        if {report["c_id"] for report in reports} != set(expected_c_ids):
            raise ValueError("U cannot conclude before every frozen C has a Department Conclusion")
        output = cast(
            EpistemicConclusionOutput,
            _structured(
                runtime.context["model"],
                EpistemicConclusionOutput,
                U_CONCLUSION_PROMPT,
                {
                    "constitution": state["constitution"],
                    "frozen_cognitive_items": state["cognitive_items"],
                    "department_conclusions": reports,
                },
                policy_role="U",
            ),
        )
        actual_c_ids = [item["c_id"] for item in output["cognitive_conclusions"]]
        if set(actual_c_ids) != set(expected_c_ids) or len(actual_c_ids) != len(expected_c_ids):
            raise ValueError("U conclusion must cover every frozen C item exactly once")
        return {"epistemic_conclusion": output}

    def writer_node(
        state: ArgumentTreeState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        conclusion = state.get("epistemic_conclusion")
        if conclusion is None:
            raise ValueError("W requires the completed structured U Epistemic Conclusion")
        output = cast(
            WriterOutput,
            _structured(
                runtime.context["model"],
                WriterOutput,
                WRITER_PROMPT,
                {
                    "constitution": state["constitution"],
                    "frozen_cognitive_items": state["cognitive_items"],
                    "department_conclusions": state.get("department_conclusions", []),
                    "epistemic_conclusion": conclusion,
                },
                policy_role="W",
            ),
        )
        return {"final_answer": _require_text(output["answer"], "W.answer")}

    def route_after_department_plan(
        state: ArgumentTreeState,
    ) -> Literal["select_technical_task", "u_conclusion"]:
        return "select_technical_task" if state.get("active_cognitive_item") is not None else "u_conclusion"

    def route_after_technical_selection(
        state: ArgumentTreeState,
    ) -> Literal["researcher", "department_conclusion"]:
        return "researcher" if state.get("technical_task") is not None else "department_conclusion"

    builder = StateGraph(
        ArgumentTreeState,
        context_schema=RuntimeContext,
        input_schema=ArgumentTreeInput,
        output_schema=ArgumentTreeOutput,
    )
    builder.add_node("interpretation", interpretation_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_node("u_plan", u_plan_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_node("department_plan", department_plan_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_node("select_technical_task", select_technical_task_node)
    builder.add_node("researcher", researcher_graph)
    builder.add_node("department_conclusion", department_conclusion_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_node("u_conclusion", u_conclusion_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_node("writer", writer_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_edge(START, "interpretation")
    builder.add_edge("interpretation", "u_plan")
    builder.add_edge("u_plan", "department_plan")
    builder.add_conditional_edges(
        "department_plan",
        route_after_department_plan,
        {"select_technical_task": "select_technical_task", "u_conclusion": "u_conclusion"},
    )
    builder.add_conditional_edges(
        "select_technical_task",
        route_after_technical_selection,
        {"researcher": "researcher", "department_conclusion": "department_conclusion"},
    )
    builder.add_edge("researcher", "select_technical_task")
    builder.add_edge("department_conclusion", "department_plan")
    builder.add_edge("u_conclusion", "writer")
    builder.add_edge("writer", END)
    return builder.compile(checkpointer=checkpointer)


def _build_cognitive_project_graph(
    technical_tools: Sequence[TechnicalTool],
    registered_data_sources: Sequence[RegisteredDataSourceDescription],
):
    """Run one frozen explanation-assessment C project.

    D freezes its T table once.  LangGraph then fans out the independent T
    tasks to R.  The D conclusion node is a reducer gate: early returning R
    branches are ignored until every fixed T packet is present.
    """

    capability_envelope = _r_capability_envelope(technical_tools)
    researcher_graph = _build_researcher_graph(technical_tools, registered_data_sources)

    def department_plan_node(
        state: CognitiveProjectState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        cognitive_item = state["cognitive_item"]
        output = cast(
            DepartmentPlanOutput,
            _structured(
                runtime.context["model"],
                DepartmentPlanOutput,
                DEPARTMENT_TASK_PROMPT,
                {
                    "constitution": state["constitution"],
                    "cognitive_item": cognitive_item,
                    "r_capability_envelope": capability_envelope,
                },
                policy_role="D",
            ),
        )
        return {
            "technical_items": _materialize_technical_items(cognitive_item, output),
            "evidence_ledger": [],
            "department_conclusions": [],
            "archive_index": [],
        }

    def dispatch_technical_tasks(
        state: CognitiveProjectState,
    ) -> list[Send]:
        cognitive_item = state["cognitive_item"]
        return [
            Send(
                "researcher",
                {
                    "constitution": state["constitution"],
                    "technical_task": _task_from_technical_item(cognitive_item, item),
                    "researcher_claim": None,
                    "researcher_plan": None,
                    "researcher_pending_plan": None,
                    "department_plan_review": None,
                    "researcher_progress_decision": None,
                    "researcher_completed_stage_ids": [],
                    "researcher_active_stage_id": None,
                    "messages": [
                        HumanMessage(
                            id=f"technical-task:{item['task_id']}",
                            content=_json(
                                {
                                    "technical_task": _task_from_technical_item(
                                        cognitive_item, item
                                    )
                                }
                            ),
                        )
                    ],
                    "objective_ledger": [],
                    "evidence_ledger": [],
                    "report_ledger": [],
                    "archive_index": [],
                    "technical_packet": None,
                },
            )
            for item in state["technical_items"]
        ]

    def department_conclusion_node(
        state: CognitiveProjectState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        cognitive_item = state["cognitive_item"]
        if any(
            report["c_id"] == cognitive_item["c_id"]
            for report in state.get("department_conclusions", [])
        ):
            return {}
        packets_by_task = {
            packet["task_id"]: packet
            for packet in state.get("evidence_ledger", [])
            if packet["item_id"] == cognitive_item["c_id"]
        }
        expected_task_ids = {item["task_id"] for item in state["technical_items"]}
        if set(packets_by_task) != expected_task_ids:
            return {}
        output = cast(
            DepartmentConclusionOutput,
            _structured(
                runtime.context["model"],
                DepartmentConclusionOutput,
                DEPARTMENT_REPORT_PROMPT,
                {
                    "constitution": state["constitution"],
                    "cognitive_item": cognitive_item,
                    "technical_items": state["technical_items"],
                    "technical_packets": [
                        packets_by_task[item["task_id"]]
                        for item in state["technical_items"]
                    ],
                },
                policy_role="D",
            ),
        )
        if output["c_id"] != cognitive_item["c_id"]:
            raise ValueError("D conclusion must retain the active c_id")
        if output["explanation_under_evaluation"] != cognitive_item["explanation_under_evaluation"]:
            raise ValueError("D conclusion must assess the C explanation, not substitute another")
        by_t_id = {entry["t_id"]: entry for entry in output["technical_conclusions"]}
        expected_t_ids = [item["t_id"] for item in state["technical_items"]]
        if set(by_t_id) != set(expected_t_ids) or len(by_t_id) != len(expected_t_ids):
            raise ValueError("D conclusion must cover every frozen T item exactly once")
        materialized: list[TechnicalConclusion] = []
        for item in state["technical_items"]:
            materialized.append(
                {
                    **copy.deepcopy(by_t_id[item["t_id"]]),
                    "packet_refs": [packets_by_task[item["task_id"]]["packet_id"]],
                }
            )
        content = {
            "c_id": cognitive_item["c_id"],
            "explanation_under_evaluation": cognitive_item["explanation_under_evaluation"],
            "conclusion": _require_text(output["conclusion"], "D.conclusion"),
            "technical_conclusions": materialized,
            "supporting_evidence": _clean_strings(output["supporting_evidence"]),
            "counterevidence": _clean_strings(output["counterevidence"]),
            "discriminative_evidence": _clean_strings(output["discriminative_evidence"]),
            "assessment_status": output["assessment_status"],
            "uncertainty": _clean_strings(output["uncertainty"]),
            "scope": _clean_strings(output["scope"]),
        }
        digest = hashlib.sha256(_json(content).encode("utf-8")).hexdigest()[:16]
        conclusion: DepartmentConclusion = {
            "department_conclusion_id": f"department-conclusion-{digest}",
            **content,
        }
        return {
            "department_conclusions": [conclusion],
            "archive_index": [
                {
                    "archive_id": f"department-conclusion:{conclusion['department_conclusion_id']}",
                    "entry_kind": "department_conclusion",
                    "item_id": cognitive_item["c_id"],
                    "reference_id": conclusion["department_conclusion_id"],
                }
            ],
        }

    project = StateGraph(
        CognitiveProjectState,
        context_schema=RuntimeContext,
        input_schema=CognitiveProjectInput,
        output_schema=CognitiveProjectOutput,
    )
    project.add_node("department_plan", department_plan_node, retry_policy=MODEL_RETRY_POLICY)
    project.add_node("researcher", researcher_graph)
    project.add_node("department_conclusion", department_conclusion_node, retry_policy=MODEL_RETRY_POLICY)
    project.add_edge(START, "department_plan")
    project.add_conditional_edges("department_plan", dispatch_technical_tasks)
    project.add_edge("researcher", "department_conclusion")
    project.add_edge("department_conclusion", END)
    return project.compile()


def build_graph(
    technical_tools: Sequence[TechnicalTool],
    *,
    checkpointer: BaseCheckpointSaver,
    registered_data_sources: Sequence[RegisteredDataSourceDescription] = (),
):
    """Compile U's parallel explanation-assessment argument tree.

    U fixes and fans out C projects.  Each project independently fans out D's
    fixed T tasks to R.  D and U are reducer gates, not scheduling agents:
    they write only after their complete fixed child table has returned.
    """

    cognitive_project_graph = _build_cognitive_project_graph(
        technical_tools, registered_data_sources
    )

    def interpretation_node(
        state: ArgumentTreeInput,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        output = _structured(
            runtime.context["model"],
            InterpretationOutput,
            INTERPRETATION_PROMPT,
            {"question": state["question"]},
            policy_role="I",
        )
        extracted = output["constitution"]
        constitution: InquiryConstitution = {
            "question": state["question"],
            "research_object": _require_text(extracted["research_object"], "I.research_object"),
            "boundaries": _clean_strings(extracted["boundaries"]),
            "requested_output": _require_text(extracted["requested_output"], "I.requested_output"),
            "epistemic_policy": copy.deepcopy(BASE_EPISTEMIC_POLICY),
        }
        if not constitution["boundaries"]:
            raise ValueError("I.boundaries must be non-empty")
        return {
            "constitution": constitution,
            "cognitive_items": [],
            "evidence_ledger": [],
            "department_conclusions": [],
            "archive_index": [],
            "epistemic_conclusion": None,
            "matrix_ready": False,
        }

    def u_plan_node(
        state: ArgumentTreeState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        output = cast(
            UPlanOutput,
            _structured(
                runtime.context["model"],
                UPlanOutput,
                UPDATER_BOOTSTRAP_PROMPT,
                {"constitution": state["constitution"]},
                policy_role="U",
            ),
        )
        items = copy.deepcopy(output["cognitive_items"])
        _validate_cognitive_items(items)
        return {"cognitive_items": items}

    def dispatch_cognitive_projects(state: ArgumentTreeState) -> list[Send]:
        return [
            Send(
                "cognitive_project",
                {"constitution": state["constitution"], "cognitive_item": item},
            )
            for item in state["cognitive_items"]
        ]

    def u_conclusion_node(
        state: ArgumentTreeState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        expected_items = state["cognitive_items"]
        reports = state.get("department_conclusions", [])
        expected_c_ids = {item["c_id"] for item in expected_items}
        if {report["c_id"] for report in reports} != expected_c_ids:
            return {"matrix_ready": False}
        output = cast(
            EpistemicConclusionOutput,
            _structured(
                runtime.context["model"],
                EpistemicConclusionOutput,
                U_CONCLUSION_PROMPT,
                {
                    "constitution": state["constitution"],
                    "frozen_cognitive_projects": expected_items,
                    "department_conclusions": reports,
                },
                policy_role="U",
            ),
        )
        rows = output["evidence_matrix"]
        by_c_id = {row["c_id"]: row for row in rows}
        if set(by_c_id) != expected_c_ids or len(by_c_id) != len(expected_c_ids):
            raise ValueError("U evidence matrix must contain every frozen C project exactly once")
        expected_explanations = {
            item["c_id"]: item["explanation_under_evaluation"] for item in expected_items
        }
        for c_id, row in by_c_id.items():
            if row["explanation"] != expected_explanations[c_id]:
                raise ValueError("U evidence matrix must preserve each C explanation identity")
        return {"epistemic_conclusion": output, "matrix_ready": True}

    def route_after_u_conclusion(
        state: ArgumentTreeState,
    ) -> Literal["writer", "__end__"]:
        return "writer" if state.get("matrix_ready") else END

    def writer_node(
        state: ArgumentTreeState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        conclusion = state.get("epistemic_conclusion")
        if conclusion is None:
            raise ValueError("W requires U's complete normalized evidence matrix")
        output = cast(
            WriterOutput,
            _structured(
                runtime.context["model"],
                WriterOutput,
                WRITER_PROMPT,
                {
                    "constitution": state["constitution"],
                    "normalized_evidence_matrix": conclusion,
                },
                policy_role="W",
            ),
        )
        return {"final_answer": _require_text(output["answer"], "W.answer")}

    builder = StateGraph(
        ArgumentTreeState,
        context_schema=RuntimeContext,
        input_schema=ArgumentTreeInput,
        output_schema=ArgumentTreeOutput,
    )
    builder.add_node("interpretation", interpretation_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_node("u_plan", u_plan_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_node("cognitive_project", cognitive_project_graph)
    builder.add_node("u_conclusion", u_conclusion_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_node("writer", writer_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_edge(START, "interpretation")
    builder.add_edge("interpretation", "u_plan")
    builder.add_conditional_edges("u_plan", dispatch_cognitive_projects)
    builder.add_edge("cognitive_project", "u_conclusion")
    builder.add_conditional_edges(
        "u_conclusion",
        route_after_u_conclusion,
        {"writer": "writer", END: END},
    )
    builder.add_edge("writer", END)
    return builder.compile(checkpointer=checkpointer)


__all__ = [
    "ArgumentTreeInput",
    "ArgumentTreeOutput",
    "ArgumentTreeState",
    "DepartmentInput",
    "DepartmentConclusion",
    "DepartmentConclusionOutput",
    "DepartmentPlanOutput",
    "DepartmentReport",
    "DepartmentResults",
    "DepartmentTaskDraft",
    "DeepResearchState",
    "BASE_EPISTEMIC_POLICY",
    "EpistemicPolicy",
    "EpistemicConclusionOutput",
    "EpistemicItem",
    "EpistemicMap",
    "EpistemicSnapshot",
    "InquiryConstitution",
    "InvalidToolCallFailure",
    "LangChainChatAdapter",
    "LocalEpistemicPackage",
    "MAX_OUTPUT_TOKENS",
    "MODEL_REQUEST_TIMEOUT_SECONDS",
    "MODEL_RETRY_POLICY",
    "ModelAdapter",
    "OPENAI_COMPATIBLE_PROVIDERS",
    "ProviderConfiguration",
    "RCapabilityEnvelope",
    "RExecutionPlan",
    "RStageResult",
    "RPlanProposal",
    "RPlanProposalDraft",
    "RProgressDecision",
    "DPlanReview",
    "RLocalStage",
    "RTaskClaim",
    "RegisteredDataSourceDescription",
    "RegisteredTableDescription",
    "StructuredOutputFailure",
    "TechnicalPacket",
    "TechnicalBasis",
    "TechnicalConclusion",
    "TechnicalConclusionDraft",
    "TechnicalItem",
    "TechnicalItemDraft",
    "TechnicalTask",
    "CognitiveItem",
    "CognitiveProjectInput",
    "CognitiveProjectOutput",
    "CognitiveProjectState",
    "EvidenceMatrixRow",
    "UPlanOutput",
    "api_schemas",
    "build_graph",
    "make_deepseek_adapter",
    "make_model_adapter",
    "make_openai_compatible_adapter",
    "provider_api_base",
    "resolve_provider_api_key",
    "resolve_provider_name",
    "resolve_provider_model",
]
