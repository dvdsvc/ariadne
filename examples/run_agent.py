"""Explicit live launcher; requires a selected model, question and local database."""
import argparse
import json
import os
from pathlib import Path
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['mix','fixed-u'], default='mix')
    parser.add_argument('--provider', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--database', type=Path, action='append', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--api-base')
    parser.add_argument('--output', type=Path, default=Path('runs/local'))
    parser.add_argument('--recursion-limit', type=int, default=128)
    parser.add_argument('--enable-temporary-python', action='store_true')
    args = parser.parse_args()
    for database in args.database:
        if not database.is_file():
            parser.error('Each --database must name an existing local file.')
    os.environ['MIX_DISABLE_SANDBOX_PYTHON'] = '0' if args.enable_temporary_python else '1'
    from langgraph.checkpoint.sqlite import SqliteSaver
    import fixed_u_epistemic_research_agent as fixed
    import mix_epistemic_research_agent as mix
    from scientific_infrastructure import ScientificInfrastructure
    from work.lightweight_langgraph_trace import LightweightTaskTrace
    agent = mix if args.variant=='mix' else fixed
    substrate = ScientificInfrastructure.from_database_paths(args.database)
    adapter = agent.make_model_adapter(args.provider, model=args.model, api_base=args.api_base)
    # A fresh per-launch directory prevents accidental overwrite or task reuse.
    output = args.output / uuid.uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    trace = LightweightTaskTrace(output/'trace.jsonl', metadata={
        'trace_contract':'showcase-explicit-live/v1', 'variant':args.variant,
        'model':args.model, 'question':args.question})
    try:
        with SqliteSaver.from_conn_string(str(output/'checkpoint.sqlite')) as saver:
            graph = agent.build_graph(substrate.langchain_tools(), checkpointer=saver,
                                      registered_data_sources=substrate.registered_data_sources())
            config = {'configurable':{'thread_id':output.name}, 'recursion_limit':args.recursion_limit}
            for part in graph.stream({'question':args.question}, context={'model':adapter},
                    config=config, stream_mode='tasks', subgraphs=True, version='v2'):
                trace.on_part(part)
            state = graph.get_state(config).values
            answer = state.get('final_answer')
            if not answer:
                raise RuntimeError('The graph stopped without a final answer; inspect local trace.')
            (output/'answer.md').write_text(answer, encoding='utf-8')
            trace.finish('completed')
    except BaseException as error:
        trace.finish('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed', error=error)
        raise
    print(json.dumps({'output':str(output), 'status':'completed'}))


if __name__=='__main__':
    main()
