# Running and reproducing the local checks

[Overview](../README.md) · [Validation](validation.md) · [Data and tools](data_and_tools.md)

## Environment

Run from the repository root using Python 3.12. `uv sync --locked` installs the pinned LangChain/LangGraph versions from the original requirements together with the scientific substrate's required libraries. The lock file records the versions resolved for this showcase. A pre-existing local environment is not required. Optional plate dependencies are resolved in the lock but are not installed by the default sync.

```bash
uv sync --locked
uv run --locked python -B -m examples.offline_protocol
uv run --locked python -B -m examples.offline_science
uv run --locked python -B -m unittest discover -s work -p 'test_*.py'
uv run --locked python -B scripts/check_repository.py
```

The repository is installed as an editable package with four top-level research modules and the retained `scientific_research_agent` package. Original tests still insert an `outputs` directory into `sys.path`; imports resolve to the installed `src` modules, so no original test edit is necessary. Execute commands from the root so shared `work.*` fixtures and examples resolve consistently.

## Selecting the research graph

Use `mix_epistemic_research_agent.build_graph` for I/U/P/D/R/W. Use `fixed_u_epistemic_research_agent.build_graph` for the latest parallel explanation-assessment configuration. Both take registered technical tools, an explicitly supplied checkpointer and source descriptions. Neither is interchangeable with the older `ScientificInfrastructure.compile_graph`, which still constructs the thin predecessor. The showcase launcher calls the selected graph directly.

## Explicit live entry point

```bash
uv run --locked python -B -m examples.run_agent --help
uv run --locked python -B -m examples.run_agent \
  --variant fixed-u --provider deepseek --model YOUR_MODEL \
  --database data/private/observations.sqlite \
  --question 'State the exact registered data scope and scientific question here.'
```

Set the selected provider's environment variable in your terminal using its own secure configuration. Do not put credentials in commands committed to the repository, source code or trace metadata. The blank `.env.example` does not auto-load anything. Provider/model names are explicit because service deployments and schema support vary. Local validation verifies graph behavior and schema conversion; it does not certify provider acceptance of every structured call.

`--database` can be repeated. Each file's stem becomes its source ID unless a custom application explicitly supplies mappings. The tool inspects real physical table and field names; the researcher must use those exact names and bind scientific roles explicitly. Do not silently reuse the historical machine's source IDs or assume its private data has been distributed.

The command sends selected local scientific context to the chosen provider and can incur model charges. It creates a unique child directory of `--output` (default `runs/local`) and writes local semantic traces plus SQLite checkpoint state. A final answer is saved only when the selected graph actually produces one. Logs and checkpoints can contain observations, prompts, provider metadata and local paths, and are ignored by Git.

## Persistence and recovery

The original agents accept a LangGraph checkpointer. Tests exercise in-memory and SQLite persistence; historical runs also contain checkpoint continuations. The new small launcher deliberately starts a fresh thread for each invocation. It does not provide a resume CLI. Programmatic recovery must use the same graph/configuration and intended stored thread, inspect the restored constitution first and continue from that checkpoint. The historical completed run's launch-versus-checkpoint question mismatch is a concrete reason to verify restored state.

## Optional capabilities

`--enable-temporary-python` opts into the retained run-scoped Python operation facility; see its [isolation boundary](data_and_tools.md). The default launcher leaves it disabled. `uv sync --locked --extra plates` installs plate-related optional libraries, but versioned external reconstruction/boundary data must still be registered separately. Optional R/IsoplotR and detrital providers are not verified by the default checks.

## Verification scope

The [CI workflow](../.github/workflows/check.yml) uses a clean Python 3.12 environment and repeats the offline commands. During preparation, these commands were run locally; the hosted workflow has not been dispatched. No model API request, database download, public push or deployment is part of the preparation checks.
