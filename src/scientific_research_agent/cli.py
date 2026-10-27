from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from .contracts import to_dict
from .external import register_manifest
from .planner import ScientificPlanner
from .persistence import RunStore
from .reasoning import AnthropicCompatibleJsonReasoner, ClaimEntailmentAdvisor, DeepSeekStrictToolReasoner, EvidenceReviewAdvisor, FinalAnswerSynthesisAdvisor, GoalPlanVerificationAdvisor, InterpretationAdvisor, IsolatedNodeConclusionAdvisor, ModelGuidedResearchPolicy, OpenAICompatibleJsonReasoner, PlanAdvisor, PlannerReviewAdvisor, ResearchActionAdvisor, ScopeResolutionAdvisor, ToolSelectionAdvisor, WriterSubgraphAdvisor
from .question import QuestionInterpreter
from .research import CapabilityResearchPolicy
from .registry import ExternalDataRegistry, SourceRegistry
from .data_adapters import DynamicTableObservationSource, GlobalZirconSQLiteSource, JsonObservationSource, TotalTreeXlsxSource, build_global_zircon_cache
from .workflow import ScientificResearchAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the scientific deep research agent")
    parser.add_argument("--data", required=True, help="Existing JSON or total_tree.xlsx observation database")
    parser.add_argument("--data-kind", choices=("tabular", "json", "total-tree", "global-zircon"), default="tabular")
    parser.add_argument("--zircon-cache", help="Reusable SQLite index for --data-kind global-zircon")
    parser.add_argument("--zircon-age-channel", choices=("206Pb/238U", "207Pb/235U", "207Pb/206Pb", "208Pb/232Th"), help="Explicit native age channel projected as age_ma for global-zircon runs")
    parser.add_argument("--external-manifest", help="Versioned local external-data manifest; no web crawling occurs")
    parser.add_argument("--model", help="Reasoning model used for planning and research actions")
    parser.add_argument("--provider", choices=("openai", "anthropic", "deepseek"), default="openai", help="Model API contract")
    parser.add_argument("--base-url", help="Provider base URL; DeepSeek defaults to its strict Beta endpoint")
    parser.add_argument("--api-key-env", help="Environment variable holding the model API key")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "high", "max"), default="max", help="DeepSeek thinking switch and effort")
    parser.add_argument("--max-model-tokens", type=int, default=384_000, help="Maximum tokens per DeepSeek structured model call")
    parser.add_argument("--question", help="Question for a new research run")
    parser.add_argument("--resume-run-id", help="Resume a durable run from --state-db")
    parser.add_argument("--state-db", default="research_runs.sqlite", help="Durable SQLite run/ledger store")
    parser.add_argument("--max-outer-rounds", type=int, default=8, help="Maximum Scout/Plan/Research/Judge cycles for this run")
    parser.add_argument("--max-inner-steps", type=int, default=12, help="Maximum frontier executions inside one research cycle")
    parser.add_argument("--test-rules", action="store_true", help="Explicit offline contract-test mode only; not for scientific research")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if not args.question and not args.resume_run_id:
        parser.error("one of --question or --resume-run-id is required")
    sources = SourceRegistry()
    if args.data_kind == "tabular":
        sources.register(DynamicTableObservationSource.from_path(args.data))
    elif args.data_kind == "total-tree":
        sources.register(TotalTreeXlsxSource(args.data))
    elif args.data_kind == "global-zircon":
        if not args.zircon_age_channel:
            parser.error("--data-kind global-zircon requires an explicit --zircon-age-channel")
        cache_path = Path(args.zircon_cache) if args.zircon_cache else Path(args.state_db).resolve().parent / "global_zircon_v2_1.sqlite"
        if not cache_path.exists():
            build_global_zircon_cache(args.data, cache_path, progress=lambda message: print(message, file=sys.stderr, flush=True))
        sources.register(GlobalZirconSQLiteSource(args.data, cache_path, age_channel=args.zircon_age_channel))
    else:
        sources.register(JsonObservationSource(args.data))
    external = ExternalDataRegistry()
    if args.external_manifest:
        register_manifest(external, args.external_manifest)
    interpreter = None; planner = None; policy = None; scope_advisor = None; action_advisor = None; claim_advisor = None; evidence_reviewer = None; node_writer_advisor = None; writer_advisor = None; goal_verifier_advisor = None; final_answer_advisor = None
    if args.model:
        default_key_env = "DEEPSEEK_API_KEY" if args.provider == "deepseek" else ("ANTHROPIC_AUTH_TOKEN" if args.provider == "anthropic" else "OPENAI_API_KEY")
        key_env = args.api_key_env or default_key_env
        api_key = os.environ.get(key_env)
        if not api_key:
            parser.error(f"--model requires the environment variable named by --api-key-env ({key_env})")
        if args.provider == "deepseek":
            reasoner = DeepSeekStrictToolReasoner(
                api_key=api_key, model=args.model, base_url=args.base_url or "https://api.deepseek.com/beta",
                reasoning_effort=args.reasoning_effort, max_tokens=args.max_model_tokens,
            )
        elif args.provider == "anthropic":
            reasoner = AnthropicCompatibleJsonReasoner(base_url=args.base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic"), auth_token=api_key, model=args.model)
        else:
            reasoner = OpenAICompatibleJsonReasoner(base_url=args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), api_key=api_key, model=args.model)
        interpreter = QuestionInterpreter(InterpretationAdvisor(reasoner))
        planner = ScientificPlanner(
            advisor=PlanAdvisor(reasoner), review_advisor=PlannerReviewAdvisor(reasoner),
        )
        scope_advisor = ScopeResolutionAdvisor(reasoner)
        policy = ModelGuidedResearchPolicy(ToolSelectionAdvisor(reasoner), CapabilityResearchPolicy())
        action_advisor = ResearchActionAdvisor(reasoner)
        claim_advisor = ClaimEntailmentAdvisor(reasoner)
        evidence_reviewer = EvidenceReviewAdvisor(reasoner)
        node_writer_advisor = IsolatedNodeConclusionAdvisor(reasoner)
        writer_advisor = WriterSubgraphAdvisor(reasoner)
        goal_verifier_advisor = GoalPlanVerificationAdvisor(reasoner)
        final_answer_advisor = FinalAnswerSynthesisAdvisor(reasoner)
    elif not args.test_rules:
        parser.error("Scientific research requires --model. --test-rules is reserved for deterministic integration tests and does not represent autonomous research.")
    else:
        planner = ScientificPlanner(allow_test_rules=True)
    agent = ScientificResearchAgent(sources, external, interpreter=interpreter, planner=planner, research_policy=policy, scope_advisor=scope_advisor, action_advisor=action_advisor, claim_advisor=claim_advisor, evidence_reviewer=evidence_reviewer, node_writer_advisor=node_writer_advisor, writer_advisor=writer_advisor, goal_verifier_advisor=goal_verifier_advisor, final_answer_advisor=final_answer_advisor, run_store=RunStore(args.state_db))
    if args.max_outer_rounds < 1 or args.max_inner_steps < 1:
        parser.error("--max-outer-rounds and --max-inner-steps must both be at least 1")
    result = (
        agent.resume(args.resume_run_id, max_outer_rounds=args.max_outer_rounds, max_inner_steps=args.max_inner_steps)
        if args.resume_run_id
        else agent.run(args.question, max_outer_rounds=args.max_outer_rounds, max_inner_steps=args.max_inner_steps)
    )
    if args.as_json:
        print(json.dumps(to_dict(result), ensure_ascii=False, indent=2))
    else:
        print(result.report.markdown)


if __name__ == "__main__":
    main()
