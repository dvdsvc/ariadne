"""Compute real statistics through the preserved tools on six synthetic rows."""
from contextlib import closing
import hashlib
import json
from operator import add
from pathlib import Path
import sqlite3
import tempfile
from typing import Annotated

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from scientific_infrastructure import ScientificInfrastructure


class State(TypedDict):
    technical_task: dict
    messages: Annotated[list, add_messages]
    objective_ledger: Annotated[list, add]
    archive_index: Annotated[list, add]


def main():
    with tempfile.TemporaryDirectory(prefix='deepresearch-demo-') as directory:
        database = Path(directory) / 'synthetic.sqlite'
        with closing(sqlite3.connect(database)) as connection:
            connection.execute('CREATE TABLE events (event_age REAL, event_sigma REAL)')
            connection.executemany('INSERT INTO events VALUES (?, ?)',
                                  [(100,1),(102,1),(104,2),(106,2),(108,2),(110,2)])
            connection.commit()
        before = hashlib.sha256(database.read_bytes()).hexdigest()
        substrate = ScientificInfrastructure.from_database_paths([database])
        # The live runner separately makes this choice configurable.
        tools = [t for t in substrate.langchain_tools() if t.name != 'sandbox__python']
        builder = StateGraph(State)
        builder.add_node('tools', ToolNode(tools))
        builder.add_edge(START, 'tools')
        builder.add_edge('tools', END)
        graph = builder.compile()
        state = {'technical_task': {'task_id':'synthetic-task', 'target_epistemic_item':'synthetic-item'},
                 'messages':[], 'objective_ledger':[], 'archive_index':[]}

        def invoke(name, args, call_id):
            state['messages'].append(AIMessage(content='', tool_calls=[{
                'name':name, 'args':{'local_intent':'synthetic demonstration', **args},
                'id':call_id, 'type':'tool_call'}]))
            state.update(graph.invoke(state))

        invoke('data__tool', {'source_id':'synthetic', 'request':{
            'operation':'query', 'table_id':'events',
            'columns':['event_age','event_sigma'], 'filters':[], 'row_limit':100}}, 'read-six')
        invoke('time__weighted_age', {'input_from':['read-six'],
            'source_fields':{'age_ma':'event_age','age_error_ma':'event_sigma'},
            'inline_observations':[]}, 'weighted-six')
        invoke('time__kde', {'input_from':['read-six'],
            'source_fields':{'age_ma':'event_age'}, 'inline_observations':[]}, 'kde-six')
        records = state['objective_ledger']
        weighted = next(r for r in records if r['record_id']=='weighted-six')
        kde = next(r for r in records if r['record_id']=='kde-six')
        assert abs(weighted['result']['findings']['weighted_age_ma'] - 103.0) < 1e-9
        assert weighted['upstream_record_ids'] == ['read-six']
        assert kde['result']['calculation_status']=='computed'
        unchanged = before == hashlib.sha256(database.read_bytes()).hexdigest()
        assert unchanged
        print(json.dumps({'mode':'real calculations on six synthetic rows; no model API',
            'rows':6, 'weighted_age_ma':weighted['result']['findings']['weighted_age_ma'],
            'one_sigma_ma':weighted['result']['findings']['one_sigma_ma'],
            'kde_status':kde['result']['calculation_status'],
            'runtime_evidence_ids':[r['record_id'] for r in records],
            'calculation_upstream_ids':weighted['upstream_record_ids'],
            'database_unchanged':unchanged}, indent=2))


if __name__=='__main__':
    main()
