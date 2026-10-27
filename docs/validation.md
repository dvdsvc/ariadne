# Local validation record

[Overview](../README.md) · [Running](running.md) · [Source manifest](../provenance/source_manifest.json)

Validation date: **2026-10-26**. Local macOS, Python 3.12.13, `uv` 0.11.16. This record concerns the independent showcase snapshot. No original research environment or source file was edited.

## Executed checks

|Check|Observed result|Evidence|
|---|---|---|
|Isolated package installation|Default dependency set installed successfully; original LangChain/LangGraph pins retained.|[pyproject.toml](../pyproject.toml), [uv.lock](../uv.lock)|
|Original offline test suite|**64 tests passed**, no failures or errors after dependency compatibility adjustment.|[Verification record](../evidence/local_verification.json), [original tests](../work/)|
|Fixed-U protocol example|Two C projects, two technical packets, two D conclusions and a two-row matrix reached Writer; model/tool responses are scripted fixtures.|[Saved protocol output](../evidence/offline_protocol.json)|
|Scientific tool example|Six synthetic rows yielded **103.0 ± 0.5773502692 Ma**; KDE returned `computed`; downstream IDs cite `read-six`; source database hash unchanged.|[Saved numerical output](../evidence/offline_science.json)|
|Live launcher interface|`--help` succeeds without importing a provider or making a request.|[Launcher](../examples/run_agent.py), [verification record](../evidence/local_verification.json)|
|Copied source integrity|48 source entries checked; 47 byte-identical copies and one documented default-path adaptation.|[Manifest](../provenance/source_manifest.json)|
|Publication payload and documentation|Source hashes, syntax, JSON/TOML/YAML/SVG parsing, bilingual heading structure, local links, candidate sizes and heuristic sensitive-token/path patterns checked.|[Repository checker](../scripts/check_repository.py)|

## Compatibility finding

The initial unconstrained resolution selected DuckDB 2.0.0. Nine original data-tool tests failed with an unfinished-result/`unregister` compatibility error in the unchanged relational engine. The showcase now declares `duckdb>=1.2,<2`, and its lock selects **1.5.6**. The original local research environment reports **1.5.5**. All 64 original tests passed with the showcase lock. This is a dependency compatibility adjustment; the query implementation and research algorithms were preserved.

## Meaning of the checks

The original tests use deterministic model adapters to make role decisions reproducible. They verify actual graph execution, task boundaries, plan authorization, context projection, runtime provenance, parallel structure, local persistence, provider adapter configuration/cache counters and selected numerical tools. The new protocol example reuses those existing fixtures and makes no measured pendulum claim. The numerical example executes real preserved tools independently of an LLM.

The repository checks validate source and document structure; no rendered website or visual browser inspection is claimed. CI configuration mirrors the local commands but has not run on GitHub. Historical evidence exports summarize archived files and do not replay their models or verify all numerical statements in their answers. Live provider schema acceptance, optional plate/R/detrital integrations, comparative task accuracy and cost have not been tested here.

The separate owner review report records absolute source paths, before/after fingerprints and archive checks. These personal paths are not embedded in the publication candidate.
