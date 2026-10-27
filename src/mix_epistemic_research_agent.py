"""Mixed epistemic-research graph with asymmetric autonomy.

I -> U -> P -> D <-> R -> U -> ... -> W. U revises proposition space;
D issues one task per U/P cycle and reviews R-authored plans; R proposes and executes
its finite local DAG. Code owns identities and provenance; LangGraph owns routing.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
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
from langgraph.types import RetryPolicy, default_retry_on
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
    "policy_id": "evidence-boundary-v1",
    "principle": (
        "指定数据源限制的是本任务的新增实证证据，不限制使用可靠的领域常识"
        "理解数据、提出假设、推导机制或引用已知事件。关于当前数据实际呈现的"
        "模式、差异以及对假设的支持程度，必须由指定数据验证。"
    ),
    "few_shot": {
        "task": "根据一组河流流量与降水记录，评估某次已知大型火山喷发是否影响了区域水循环。",
        "background_knowledge": (
            "该火山喷发的大致年代；火山气溶胶通常会影响辐射和气候；"
            "降水与径流之间的一般水文关系。"
        ),
        "must_be_data_validated": (
            "喷发前后降水和流量是否实际发生异常、异常持续多久、"
            "变化是否符合相关机制的预期。"
        ),
        "wrong": "因为数据表没有“火山喷发”字段，所以无法研究喷发影响。",
        "correct": "喷发年代可作为背景知识，当前数据负责检验喷发前后的实际响应。",
    },
    "role_permissions": {
        "I": "固定该法则并将其挂入任务 constitution；不得把法则改写成任务边界。",
        "U": "区分背景知识、数据发现和解释性推断；只用数据证据更新数据事实及支持程度。",
        "P": "按认识增益选择问题；不得把背景知识误拆成必须从指定数据重新证明的命题。",
        "D": "把需要数据验证的部分转成技术任务；不得因缺少同名字段而否定背景知识或机制假设。",
        "R": "用指定数据检验当前模式和响应；可使用背景知识指导查询与假设，但不得伪装成数据发现。",
        "W": "明确区分领域背景、数据结果和由数据支持的结论，不扩大数据证据边界。",
    },
}


class InterpretationConstitution(TypedDict):
    research_object: str
    boundaries: list[str]
    requested_output: str
    epistemic_policy: EpistemicPolicy


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


ROLE_CONTRACTS: dict[str, type] = {
    "interpretation": InterpretationOutput,
    "updater": UpdaterOutput,
    "planner": PlannerSelection,
    "department_task": DepartmentTaskDraft,
    "researcher_claim": RTaskClaim,
    "researcher_plan": RPlanProposalDraft,
    "department_plan_review": DPlanReview,
    "researcher_progress": RProgressDecision,
    "technical_results": TechnicalResults,
    "department_report": DepartmentResults,
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

    def structured(self, schema: type) -> Runnable[Any, Any]: ...


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


@dataclass(slots=True)
class LangChainChatAdapter:
    model: BaseChatModel
    structured_output_kwargs: JSON = field(default_factory=dict)
    bind_tools_kwargs: JSON = field(default_factory=dict)

    def structured(self, schema: type) -> Runnable[Any, Any]:
        runnable = self.model.with_structured_output(
            schema,
            **{**self.structured_output_kwargs, "include_raw": True},
        )
        return runnable | RunnableLambda(
            lambda result: _unwrap_structured_result(result, schema)
        )


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
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=MODEL_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )
    return LangChainChatAdapter(
        chat_model,
        structured_output_kwargs={"method": "function_calling", "strict": strict},
        bind_tools_kwargs={"strict": strict},
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
                for key in ("record_id", "packet_id", "report_id", "archive_id")
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


ORGANIZATION_BOUNDARY_FEWSHOT = """
Shared boundary example — question: "Do the northern and southern zircon age structures differ stably, and how far could that support different tectono-magmatic histories?"
I fixes the user's object, geographic scope, and requested answer; it does not choose tables, age fields, or methods. U records the few genuine propositions: whether a stable difference exists, which competing explanations remain, and the present claim ceiling; it does not write data or analysis steps. P copies one open U item_id. D translates that item into a bounded technical evidence task with required outputs and acceptance conditions; it does not prescribe SQL, algorithms, or parameters. R claims that task_id, designs and executes its local DAG, and may change tools, queries, parameters, ordering, or uncompleted stages inside the task. D reports the technical result and its boundary. U alone updates proposition status and interpretation. Writer publishes only U's final state.
Boundary counterexamples: U says "query a field and run KDE"; D says "the result proves separate magmatic events"; R adds a lithology investigation because it looks interesting. Each statement uses authority owned by another layer.
"""


INTERPRETATION_PROMPT = """
You are boundary-registration layer I. Record the central research object, explicit user boundaries, and requested output. Resolve references only enough to make those entries unambiguous. Return the constitution fields.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

UPDATER_BOOTSTRAP_PROMPT = """
You are epistemic update layer U. From the constitution, create revision 0 of the epistemic_map. Use the fewest items that preserve distinct epistemic propositions; do not split a question into data preparation, method selection, and validation phases. Each item contains a genuine question, proposition, competing explanation, or material uncertainty, represented by supported, challenged, uncertain, remaining_gap, and status. current_answer says only what can presently be answered; claim_ceiling says the strongest defensible claim. Code derives routing from the map.
Return the complete epistemic_map, including the required overall_state object with both current_answer and claim_ceiling, even when no technical evidence has yet been collected. Do not omit overall_state or either of its fields.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

UPDATER_PROMPT = """
You are epistemic update layer U. Revise the previous epistemic_map from the latest DepartmentReport. Preserve item_id and question when only knowledge changes; create a new item_id only for a genuinely new proposition. Update supported, challenged, uncertain, remaining_gap, status, current_answer, and claim_ceiling. current_answer must never prescribe future technical work. Return the complete map with an increased revision; code derives routing.
The returned map must always include the required overall_state object with both current_answer and claim_ceiling. Preserve or revise those two fields on every update; never omit overall_state or either of its fields.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

PLANNER_PROMPT = """
You are selection layer P. Select the one open U item with the highest expected epistemic gain relative to cost. Copy its item_id exactly and return only selected_item_id.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

DEPARTMENT_TASK_PROMPT = """
You are technical organization layer D. Translate the selected U item into one executable, bounded task for R. Define technical_objective, scope, required_outputs, acceptance_criteria, coarse_stages, allowed resource classes, and escalation_conditions. Coarse stages describe the work's shape; R chooses concrete tools, queries, algorithms, parameters, and execution order. Return one task draft.
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
You are technical execution layer R. Execute the current ready stage of the approved DAG with one registered tool call or a small parallel batch. Use dependency results and remain inside the D-issued task. If the immediately preceding call in this stage was rejected as a tool-input error, correct it from the returned message; a successful call ends this repair loop.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

RESEARCHER_PROGRESS_PROMPT = """
You are technical execution layer R. Observe the latest completed stage. Continue when the remaining approved stages still fit; propose a patch only when DAG topology must change; submit when results suffice for D or an escalation condition is reached. Tool, query, algorithm, parameter, ordering, or retry changes inside an existing stage are not DAG changes. A patch cannot execute before D approval.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

RESEARCHER_FINALIZE_PROMPT = """
You are technical execution layer R. Return only the obtained values, observations, calculations, and execution failures that address the task's required outputs. Do not decide U's proposition status or copy tool-call identifiers; code attaches provenance.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

DEPARTMENT_REPORT_PROMPT = """
You are technical organization layer D. Organize the TechnicalPacket returned for the issued task into what is technically known, concise core evidence, the conclusion boundary, remaining uncertainty, and cross-item implications. Do not update the epistemic map, make the scientific interpretation, invent results, or copy evidence identifiers. Return DepartmentResults.
""" + ORGANIZATION_BOUNDARY_FEWSHOT

WRITER_PROMPT = """
You are publication layer Writer. Present U's closed epistemic_map as the strongest supported answer, its scope, material uncertainty, and claim ceiling. Do not introduce new analysis or strengthen U's conclusions.
""" + ORGANIZATION_BOUNDARY_FEWSHOT


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


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
    result = adapter.structured(schema).invoke(
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
    completed = set(completed_stage_ids)
    for stage in plan["stages"]:
        if stage["stage_id"] not in completed and set(stage["depends_on"]) <= completed:
            return stage
    return None


def _project_researcher_messages(
    messages: Sequence[AnyMessage],
    transient_tool_names: set[str],
) -> list[AnyMessage]:
    """Keep durable scientific-tool history; omit dependency-scoped reads."""

    projected: list[AnyMessage] = []
    transient_call_ids = {
        str(call.get("id") or "")
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if str(call.get("name") or "") in transient_tool_names
    }
    for message in messages:
        if isinstance(message, HumanMessage):
            projected.append(HumanMessage(content=copy.deepcopy(message.content), id=message.id))
        elif isinstance(message, AIMessage) and message.tool_calls:
            durable_calls = [
                copy.deepcopy(call)
                for call in message.tool_calls
                if str(call.get("name") or "") not in transient_tool_names
            ]
            if durable_calls:
                projected.append(AIMessage(content="", tool_calls=durable_calls))
        elif (
            isinstance(message, ToolMessage)
            and str(message.tool_call_id or "") not in transient_call_ids
            and str(message.name or "") not in transient_tool_names
        ):
            projected.append(
                ToolMessage(
                    content=copy.deepcopy(message.content),
                    name=message.name,
                    tool_call_id=message.tool_call_id,
                    status=message.status,
                    artifact=copy.deepcopy(message.artifact),
                    additional_kwargs=copy.deepcopy(message.additional_kwargs),
                )
            )
    return projected


def _tool_observations(
    messages: Sequence[AnyMessage],
    *,
    excluded_tool_names: set[str] | None = None,
) -> list[JSON]:
    excluded = excluded_tool_names or set()
    return [
        {
            "tool": str(message.name or ""),
            "tool_call_id": str(message.tool_call_id or ""),
            "status": message.status,
            "result": copy.deepcopy(message.content),
        }
        for message in messages
        if isinstance(message, ToolMessage)
        and str(message.name or "") not in excluded
    ]


def _stage_tool_groups(
    messages: Sequence[AnyMessage],
    completed_stage_ids: Sequence[str],
    active_stage_id: str | None,
) -> list[JSON]:
    """Derive each action group's DAG stage from the existing ordered trace."""

    stage_sequence = list(completed_stage_ids)
    if active_stage_id and active_stage_id not in stage_sequence:
        stage_sequence.append(active_stage_id)
    tool_messages = {
        str(message.tool_call_id or ""): message
        for message in messages
        if isinstance(message, ToolMessage)
    }
    groups: list[JSON] = []
    stage_index = 0
    for message in messages:
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        stage_id = (
            stage_sequence[stage_index]
            if stage_index < len(stage_sequence)
            else ""
        )
        calls = [copy.deepcopy(call) for call in message.tool_calls]
        results = [
            tool_messages[str(call.get("id") or "")]
            for call in calls
            if str(call.get("id") or "") in tool_messages
        ]
        groups.append(
            {
                "stage_id": stage_id,
                "calls": calls,
                "results": results,
            }
        )
        call_ids = {str(call.get("id") or "") for call in calls}
        returned_ids = {str(result.tool_call_id or "") for result in results}
        if (
            call_ids
            and call_ids <= returned_ids
            and all(result.status != "error" for result in results)
        ):
            stage_index += 1
    return groups


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


def _transient_stage_observations(
    messages: Sequence[AnyMessage],
    *,
    completed_stage_ids: Sequence[str],
    active_stage_id: str | None,
    included_stage_ids: set[str],
    transient_tool_names: set[str],
) -> list[JSON]:
    observations: list[JSON] = []
    for group in _stage_tool_groups(
        messages,
        completed_stage_ids,
        active_stage_id,
    ):
        if group["stage_id"] not in included_stage_ids:
            continue
        results_by_id = {
            str(result.tool_call_id or ""): result
            for result in group["results"]
        }
        for call in group["calls"]:
            if str(call.get("name") or "") not in transient_tool_names:
                continue
            call_id = str(call.get("id") or "")
            result = results_by_id.get(call_id)
            if result is None:
                continue
            observations.append(
                {
                    "stage_id": group["stage_id"],
                    "tool": str(call.get("name") or ""),
                    "tool_call_id": call_id,
                    "arguments": copy.deepcopy(call.get("args") or {}),
                    "status": result.status,
                    "result": copy.deepcopy(result.content),
                }
            )
    return observations


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
    transient_tool_names = {
        str(convert_to_openai_tool(item)["function"]["name"])
        for item in tools
        if (
            str((getattr(item, "metadata", None) or {}).get("native_tool_id") or "")
            in {"data.tool", "archive.read"}
            or str(convert_to_openai_tool(item)["function"]["name"])
            in {"data__tool", "archive__read"}
        )
    }
    tool_catalogue = _tool_catalogue(tools)
    sources = copy.deepcopy(list(registered_data_sources))

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
        completed = state.get("researcher_completed_stage_ids", [])
        patch_focus = [completed[-1]] if authorized_plan is not None and completed else []
        patch_stage_scope = (
            _dag_stage_closure(authorized_plan, patch_focus)
            if authorized_plan is not None
            else set()
        )
        draft = cast(
            RPlanProposalDraft,
            _structured(
                runtime.context["model"],
                RPlanProposalDraft,
                RESEARCHER_PLAN_PROMPT,
                {
                    "constitution": state["constitution"],
                    "technical_task": task,
                    "task_claim": claim,
                    "proposal_kind": (
                        "initial" if authorized_plan is None else "patch"
                    ),
                    "current_authorized_dag": authorized_plan,
                    "completed_stage_ids": completed,
                    "latest_d_review": state.get("department_plan_review"),
                    "scientific_tool_history": _tool_observations(
                        state.get("messages", []),
                        excluded_tool_names=transient_tool_names,
                    ),
                    "dag_dependency_observations": _transient_stage_observations(
                        state.get("messages", []),
                        completed_stage_ids=completed,
                        active_stage_id=state.get("researcher_active_stage_id"),
                        included_stage_ids=patch_stage_scope,
                        transient_tool_names=transient_tool_names,
                    ),
                    "registered_capabilities": tool_catalogue,
                    "registered_data_sources": sources,
                    "prior_packets_for_same_item": [
                        packet
                        for packet in state.get("evidence_ledger", [])
                        if packet["item_id"] == task["target_epistemic_item"]
                        and packet["task_id"] != task["task_id"]
                    ],
                },
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
            state.get("researcher_completed_stage_ids", []),
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

    def route_after_plan_review(
        state: ResearcherGraphState,
    ) -> Literal["execute_stage", "finalize", "plan_local_dag", "review_progress"]:
        review = state["department_plan_review"]
        if review["decision"] == "approve":
            plan = state["researcher_plan"]
            completed = set(state.get("researcher_completed_stage_ids", []))
            return (
                "finalize"
                if all(stage["stage_id"] in completed for stage in plan["stages"])
                else "execute_stage"
            )
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
        plan = state.get("researcher_plan")
        if plan is None:
            raise ValueError("R has no execution plan")
        stage = _next_ready_stage(plan, state.get("researcher_completed_stage_ids", []))
        if stage is None:
            raise ValueError("R has no ready stage to execute")
        adapter = runtime.context["model"]
        model = getattr(adapter, "model", None)
        if not isinstance(model, BaseChatModel):
            raise TypeError("R tool execution requires ModelAdapter.model")
        bind_kwargs = dict(getattr(adapter, "bind_tools_kwargs", {}))
        bind_kwargs.pop("tool_choice", None)
        bound = model.bind_tools(tools, tool_choice="required", **bind_kwargs)
        response = bound.invoke(
            [
                SystemMessage(
                    content=_prompt_with_epistemic_policy(
                        RESEARCHER_EXECUTE_PROMPT,
                        state["constitution"],
                        "R",
                    )
                ),
                *_project_researcher_messages(
                    state.get("messages", []),
                    transient_tool_names,
                ),
                HumanMessage(
                    content=_json(
                        {
                            "technical_task": state["technical_task"],
                            "task_claim": state["researcher_claim"],
                            "execution_plan": plan,
                            "completed_stage_ids": state.get(
                                "researcher_completed_stage_ids", []
                            ),
                            "current_stage": stage,
                            "dag_dependency_observations": _transient_stage_observations(
                                state.get("messages", []),
                                completed_stage_ids=state.get(
                                    "researcher_completed_stage_ids", []
                                ),
                                active_stage_id=state.get("researcher_active_stage_id"),
                                included_stage_ids=(
                                    _dag_stage_closure(plan, stage["depends_on"])
                                    | {stage["stage_id"]}
                                ),
                                transient_tool_names=transient_tool_names,
                            ),
                            "registered_data_sources": sources,
                        }
                    )
                ),
            ]
        )
        if not isinstance(response, AIMessage):
            raise TypeError("R tool-bound model must return AIMessage")
        if response.invalid_tool_calls:
            raise InvalidToolCallFailure(response)
        calls = response.tool_calls
        if not calls:
            raise ValueError("R must emit at least one tool call for an executable DAG stage")
        for call in calls:
            tool_name = str(call.get("name") or "")
            if tool_name not in tool_names:
                raise ValueError(f"R selected an unregistered tool: {tool_name}")
        return {
            "researcher_active_stage_id": stage["stage_id"],
            "messages": [response],
        }

    def completed_stage_ids(state: ResearcherGraphState) -> list[str]:
        active = state.get("researcher_active_stage_id")
        if not active:
            return list(state.get("researcher_completed_stage_ids", []))
        latest_tool_messages = _latest_tool_messages(state.get("messages", []))
        if not latest_tool_messages:
            raise ValueError("R cannot advance a stage without a tool observation")
        completed = list(state.get("researcher_completed_stage_ids", []))
        if any(message.status == "error" for message in latest_tool_messages):
            return completed
        if active not in completed:
            completed.append(active)
        return completed

    def route_after_tools(
        state: ResearcherGraphState,
    ) -> Literal["execute_stage", "review_progress", "finalize"]:
        latest_tool_messages = _latest_tool_messages(state.get("messages", []))
        if not latest_tool_messages:
            raise ValueError("ToolNode returned no ToolMessage for the current R action")
        if any(message.status == "error" for message in latest_tool_messages):
            return "execute_stage"
        plan = state["researcher_plan"]
        completed = set(completed_stage_ids(state))
        return (
            "finalize"
            if all(stage["stage_id"] in completed for stage in plan["stages"])
            else "review_progress"
        )

    def review_progress_node(
        state: ResearcherGraphState,
        runtime: Runtime[RuntimeContext],
    ) -> dict[str, Any]:
        plan = state.get("researcher_plan")
        if plan is None:
            raise ValueError("R cannot review progress without an authorized DAG")
        completed = completed_stage_ids(state)
        active_stage_id = state.get("researcher_active_stage_id")
        decision = cast(
            RProgressDecision,
            _structured(
                runtime.context["model"],
                RProgressDecision,
                RESEARCHER_PROGRESS_PROMPT,
                {
                    "constitution": state["constitution"],
                    "technical_task": state["technical_task"],
                    "authorized_dag": plan,
                    "completed_stage_ids": completed,
                    "scientific_tool_history": _tool_observations(
                        state.get("messages", []),
                        excluded_tool_names=transient_tool_names,
                    ),
                    "latest_stage_observations": _transient_stage_observations(
                        state.get("messages", []),
                        completed_stage_ids=state.get(
                            "researcher_completed_stage_ids", []
                        ),
                        active_stage_id=active_stage_id,
                        included_stage_ids={active_stage_id} if active_stage_id else set(),
                        transient_tool_names=transient_tool_names,
                    ),
                    "latest_d_review": state.get("department_plan_review"),
                },
                policy_role="R",
            ),
        )
        return {
            "researcher_progress_decision": decision,
            "researcher_pending_plan": None,
            "researcher_completed_stage_ids": completed,
            "researcher_active_stage_id": None,
        }

    def route_after_progress(
        state: ResearcherGraphState,
    ) -> Literal["execute_stage", "plan_local_dag", "finalize"]:
        decision = state["researcher_progress_decision"]["decision"]
        if decision == "continue":
            return "execute_stage"
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
        observations = _tool_observations(state.get("messages", []))
        if not observations:
            raise ValueError("R must execute at least one registered technical tool")
        structured = _structured(
            runtime.context["model"],
            TechnicalResults,
            RESEARCHER_FINALIZE_PROMPT,
            {
                "constitution": state["constitution"],
                "technical_task": task,
                "task_claim": claim,
                "execution_plan": state["researcher_plan"],
                "scientific_tool_history": _tool_observations(
                    state.get("messages", []),
                    excluded_tool_names=transient_tool_names,
                ),
                "terminal_stage_observations": _transient_stage_observations(
                    state.get("messages", []),
                    completed_stage_ids=state.get(
                        "researcher_completed_stage_ids", []
                    ),
                    active_stage_id=state.get("researcher_active_stage_id"),
                    included_stage_ids=_terminal_stage_ids(state["researcher_plan"]),
                    transient_tool_names=transient_tool_names,
                ),
            },
            policy_role="R",
        )
        results = _clean_strings(structured.get("results", []))
        if not results:
            raise ValueError("R must return at least one obtained technical result")
        tool_call_ids = list(
            dict.fromkeys(
                item["tool_call_id"]
                for item in observations
                if item["tool_call_id"] and item["status"] != "error"
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
    builder.add_node("execute_stage", execute_stage_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_node("tools", ToolNode(tools, messages_key="messages"))
    builder.add_node("review_progress", review_progress_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_node("finalize", finalize_node, retry_policy=MODEL_RETRY_POLICY)
    builder.add_edge(START, "claim_task")
    builder.add_edge("claim_task", "plan_local_dag")
    builder.add_edge("plan_local_dag", "department_plan_review")
    builder.add_conditional_edges(
        "department_plan_review",
        route_after_plan_review,
        {
            "execute_stage": "execute_stage",
            "finalize": "finalize",
            "plan_local_dag": "plan_local_dag",
            "review_progress": "review_progress",
        },
    )
    builder.add_edge("execute_stage", "tools")
    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "execute_stage": "execute_stage",
            "review_progress": "review_progress",
            "finalize": "finalize",
        },
    )
    builder.add_conditional_edges(
        "review_progress",
        route_after_progress,
        {
            "execute_stage": "execute_stage",
            "plan_local_dag": "plan_local_dag",
            "finalize": "finalize",
        },
    )
    builder.add_edge("finalize", END)
    return builder.compile()


def build_graph(
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
        report = state.get("department_report")
        if prior is None:
            context = {"mode": "bootstrap", "constitution": state["constitution"]}
            prompt = UPDATER_BOOTSTRAP_PROMPT
            old_revision = None
        else:
            if report is None:
                raise ValueError("U revision requires a DepartmentReport")
            context = {
                "mode": "revise",
                "constitution": state["constitution"],
                "current_u_snapshot": _project_constitution_reference(
                    cast(JSON, prior),
                    state["constitution"],
                    owner="current U snapshot",
                ),
                "department_report": report,
            }
            prompt = UPDATER_PROMPT
            old_revision = int(prior["epistemic_map"]["revision"])
        update = _structured(
            runtime.context["model"],
            UpdaterOutput,
            prompt,
            context,
            policy_role="U",
        )
        output = _materialize_epistemic_snapshot(
            constitution=state["constitution"],
            update=update,
            prior=prior,
            report=report,
        )
        _validate_epistemic_snapshot(output, old_revision)
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
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)] if prior else [],
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
            },
            policy_role="W",
        )
        return {"final_answer": output["answer"]}

    def route_after_updater(state: PlannerInput) -> Literal["planner", "writer"]:
        has_open_item = any(
            item["status"] == "open"
            for item in state["u_snapshot"]["epistemic_map"]["items"]
        )
        return "planner" if has_open_item else "writer"

    def route_after_planner(state: PlannerRouteInput) -> Literal["department"]:
        if state.get("selected_item_package") is None:
            raise ValueError("P did not select an epistemic item")
        return "department"

    def route_after_department(
        state: DepartmentRouteInput,
    ) -> Literal["researcher", "updater"]:
        return "updater" if state.get("department_report") is not None else "researcher"

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
    builder.add_node(
        "planner", planner_node, input_schema=PlannerInput, retry_policy=MODEL_RETRY_POLICY
    )
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
    builder.add_conditional_edges(
        "updater", route_after_updater, {"planner": "planner", "writer": "writer"}
    )
    builder.add_conditional_edges(
        "planner", route_after_planner, {"department": "department"}
    )
    builder.add_conditional_edges(
        "department",
        route_after_department,
        {"researcher": "researcher", "updater": "updater"},
    )
    builder.add_edge("researcher", "department")
    builder.add_edge("writer", END)
    return builder.compile(checkpointer=checkpointer)


__all__ = [
    "DepartmentInput",
    "DepartmentReport",
    "DepartmentResults",
    "DepartmentTaskDraft",
    "DeepResearchState",
    "BASE_EPISTEMIC_POLICY",
    "EpistemicPolicy",
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
    "TechnicalTask",
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
