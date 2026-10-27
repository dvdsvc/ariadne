from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage


OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
sys.path.insert(0, str(OUTPUTS))

import fixed_u_epistemic_research_agent as fixed


class NamedModel:
    model_name = "gpt-cache-test"


class ProviderCacheAdapterTest(unittest.TestCase):
    def test_normalizes_langchain_usage_into_role_scoped_pi_counters(self) -> None:
        adapter = fixed.LangChainChatAdapter(model=NamedModel())  # type: ignore[arg-type]
        response = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 600,
                "output_tokens": 10,
                "total_tokens": 610,
                "input_token_details": {
                    "cache_read": 480,
                    "cache_creation": 120,
                }
            },
        )

        adapter.record_response(response, cache_scope="R")

        self.assertEqual(response.usage_metadata["cacheRead"], 480)
        self.assertEqual(response.usage_metadata["cacheWrite"], 120)
        self.assertEqual(
            adapter.cache_usage(),
            {
                "cacheRead": 480,
                "cacheWrite": 120,
                "byScope": {"r": {"cacheRead": 480, "cacheWrite": 120}},
            },
        )

    def test_reads_official_openai_compatible_raw_usage_when_langchain_has_no_detail(self) -> None:
        adapter = fixed.LangChainChatAdapter(model=NamedModel())  # type: ignore[arg-type]
        response = AIMessage(
            content="ok",
            response_metadata={
                "token_usage": {
                    "prompt_tokens_details": {
                        "cached_tokens": 320,
                        "cache_write_tokens": 80,
                    }
                }
            },
        )

        adapter.record_response(response, cache_scope="D")

        self.assertEqual(
            adapter.cache_usage()["byScope"]["d"],
            {"cacheRead": 320, "cacheWrite": 80},
        )

    def test_keeps_read_and_write_distinct_when_the_provider_reports_both(self) -> None:
        adapter = fixed.LangChainChatAdapter(model=NamedModel())  # type: ignore[arg-type]
        response = AIMessage(
            content="ok",
            response_metadata={
                "token_usage": {
                    "prompt_tokens_details": {
                        "cached_tokens": 320,
                        "cache_write_tokens": 80,
                    }
                }
            },
        )

        adapter.record_response(response, cache_scope="U")

        self.assertEqual(
            adapter.cache_usage()["byScope"]["u"],
            {"cacheRead": 320, "cacheWrite": 80},
        )

    def test_openai_affinity_key_is_role_scoped_and_glm_stays_implicit(self) -> None:
        openai = fixed.LangChainChatAdapter(
            model=NamedModel(),  # type: ignore[arg-type]
            provider="openai",
            supports_prompt_cache_key=True,
        )
        glm = fixed.LangChainChatAdapter(
            model=NamedModel(),  # type: ignore[arg-type]
            provider="glm",
        )

        self.assertEqual(
            openai._cache_request_kwargs("R"),
            {"prompt_cache_key": "fixed-u:openai:gpt-cache-test:r"},
        )
        self.assertEqual(openai._cache_request_kwargs("D"), {"prompt_cache_key": "fixed-u:openai:gpt-cache-test:d"})
        self.assertEqual(glm._cache_request_kwargs("R"), {})

    def test_cache_ordered_context_keeps_static_cards_before_stage_artifacts(self) -> None:
        packet = fixed._cache_ordered_context(
            {
                "technical_task": {"task_id": "T-1"},
                "registered_data_sources": [{"source_id": "S-1"}],
            },
            {
                "current_stage": {"stage_id": "stage-2"},
                "dag_dependency_stage_results": [{"stage_id": "stage-1"}],
            },
        )

        self.assertEqual(
            list(packet),
            [
                "technical_task",
                "registered_data_sources",
                "current_stage",
                "dag_dependency_stage_results",
            ],
        )
        with self.assertRaisesRegex(ValueError, "both stable and dynamic"):
            fixed._cache_ordered_context({"technical_task": {}}, {"technical_task": {}})


if __name__ == "__main__":
    unittest.main()
