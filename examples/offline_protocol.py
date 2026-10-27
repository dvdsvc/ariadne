"""Exercise the real fixed-U graph with the original scripted protocol fixture.

All model responses and the pendulum tool are fixtures. This demonstrates
routing, identity assignment and evidence-matrix assembly, not scientific skill.
"""
import json

from langgraph.checkpoint.memory import InMemorySaver
import fixed_u_epistemic_research_agent as fixed
from work.test_fixed_u_epistemic_research_agent import ParallelAssessmentAdapter
from work.test_thin_epistemic_research_agent import (
    RAW_QUESTION, REGISTERED_DATA_SOURCES, TOOL_CALLS, pendulum_compute, tool_call,
)


def main():
    TOOL_CALLS.clear()
    model = ParallelAssessmentAdapter()
    model.researcher_tool_script = [
        tool_call('pendulum_compute', {'mode': 'group_difference'}, 'demo-contrast'),
        tool_call('pendulum_compute', {'mode': 'leave_one_out'}, 'demo-sensitivity'),
    ]
    graph = fixed.build_graph([pendulum_compute], checkpointer=InMemorySaver(),
                              registered_data_sources=REGISTERED_DATA_SOURCES)
    config = {'recursion_limit': 96, 'configurable': {'thread_id': 'offline-protocol'}}
    result = graph.invoke({'question': RAW_QUESTION}, context={'model': model}, config=config)
    state = graph.get_state(config).values
    rows = state['epistemic_conclusion']['evidence_matrix']
    assert len(rows) == 2 and len(state['evidence_ledger']) == 2
    assert result['final_answer']
    summary = {
        'mode': 'scripted protocol fixture; no model API; no measured pendulum result',
        'cognitive_projects': len(state['cognitive_items']),
        'technical_packets': len(state['evidence_ledger']),
        'department_conclusions': len(state['department_conclusions']),
        'matrix_rows': len(rows),
        'tool_modes_executed': sorted(TOOL_CALLS),
        'writer_reached': bool(result['final_answer']),
        'matrix_explanations': [row['explanation'] for row in sorted(rows, key=lambda r:r['c_id'])],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
