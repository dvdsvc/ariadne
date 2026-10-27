# Data, scientific tools and evidence

[Overview](../README.md) · [Methods](methods.md) · [Running](running.md)

## Registered observation sources

[`ScientificInfrastructure.from_database_paths`](../src/scientific_infrastructure.py) wraps existing SQLite, CSV and workbook tables through `DynamicTableObservationSource`. Source/table identities and physical column names are exposed through source descriptions and the data tool. Source readers remain read-only; analyses operate on selected execution values. Private research databases are not bundled.

The preserved [tabular adapter](../src/scientific_research_agent/tabular.py), [relational engine](../src/scientific_research_agent/relational_engine.py), [source adapters](../src/scientific_research_agent/data_adapters/__init__.py) and [data tool](../src/scientific_research_agent/data_tool.py) implement the underlying access. The older source package also includes `TotalTreeXlsxSource` for linked extraction tables. That adapter is a connection point between literature extraction and scientific analysis; it is not evidence that a complete extraction-to-discovery pipeline has been validated.

The newer LangChain `data__tool` interface uses a bounded request union. It exposes inspect, profile, sample, query, aggregate and relate operations. Original substrate functions retain richer operations, so their names must not be used interchangeably with the public agent tool schema.

## Field binding and execution values

The agent must explicitly bind physical columns to scientific roles such as `age_ma`, `age_error_ma`, coordinates and region. A statistical call can consume prior data-result IDs through `input_from` or declared inline observations. It does not infer an age column from arbitrary raw data. Original tests also replace a source's query method with an error after selection to check that downstream statistics consume the previously produced rows rather than silently reading the database again.

The [offline science example](../examples/offline_science.py) exercises this actual chain:

```text
synthetic SQLite → data__tool / read-six
                          ├→ time__weighted_age / weighted-six
                          └→ time__kde / kde-six
```

Six synthetic ages `(100,102,104,106,108,110)` Ma with uncertainties `(1,1,2,2,2,2)` Ma produce the inverse-variance weighted age 103.0 Ma. The expected one-sigma value is `sqrt(1/3)` Ma. This verifies a numerical tool and its input provenance; it is not a geological event interpretation.

## Tool families

The actual registry is defined in [`default_tool_registry`](../src/scientific_research_agent/tools_builtin.py) and adapted by [`ScientificInfrastructure.langchain_tools`](../src/scientific_infrastructure.py).

|Family|Implemented examples|Interpretation boundary|
|---|---|---|
|Data access|Table inspection, filtering, aggregation, table relations|Returns rows and computed descriptions, not scientific support|
|Temporal statistics|Inverse-variance age, temporal bins, KDE, GMM, dip test, change points, bootstrap|Results depend on selected records, uncertainty assumptions and chosen method|
|Spatial statistics|Age-coordinate regression, centroids, DBSCAN, projected coverage geometry|Spatial sampling and projection affect interpretation|
|Uncertainty and bias|Monte Carlo migration, leave-one-source/region-out, permutation nulls|A robustness calculation is a specific diagnostic, not universal validation|
|External boundary data|Registered reconstruction/geometry compatibility operations|Needs separately acquired, versioned external datasets and optional software|
|Archive and temporary computation|Read task archive; define run-scoped Python operations|Auditability and actual isolation must be considered separately|

Some package-backed adapters are exposed only when the required runtime is available. Plate reconstruction dependencies are in the optional `plates` group. IsoplotR needs R plus its package, and detrital functionality needs its own runtime. These optional scientific integrations were not exercised in the showcase verification. A static registry entry is not a successful execution result.

## Objective records and scientific conclusions

`_record_and_message` stores a tool call's objective record, upstream IDs, task/item identity, parameters, result and status. The call becomes a ToolMessage for the active R episode and an archive entry. Agent-generated `TechnicalResults` supply semantic results while code attaches provenance. D and U operate on the intended structured summaries rather than treating tool success as scientific support.

The evidence chain is `tool call → objective record → technical packet → D report/conclusion → U state/matrix → W answer`. The mixed and fixed-U variants use different higher-level records; their schemas are documented in the source and [methods](methods.md).

## Temporary Python operation boundary

The original infrastructure retains a temporary, run-scoped Python operation tool. It captures the submitted code, inputs, output/error and execution engine. Its subprocess has a restricted environment and timeout. On macOS it attempts `sandbox-exec`; if that platform facility is unavailable or disallowed, the code can fall back to an isolated Python subprocess. On other platforms the operating-system restriction is likewise not equivalent to a dedicated hardened sandbox.

The new [live launcher](../examples/run_agent.py) disables this tool unless `--enable-temporary-python` is supplied. This wrapper choice does not alter the original implementation, whose default is to expose it. Original tests that exercise known arithmetic snippets remain included. This facility should not be described as strong portable isolation for arbitrary hostile code.
