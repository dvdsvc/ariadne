# Provenance, contributions and release boundary

[Overview](../README.md) · [Source manifest](../provenance/source_manifest.json) · [Third-party notices](../THIRD_PARTY_NOTICES.md)

## Source selection

The showcase collects the actual I/U/P/D/R/W mixed graph, the newer fixed-U parallel variant, the thin predecessor needed by original tests, the scientific infrastructure adapter, original offline tests and a semantic trace observer. The infrastructure imports the earlier `scientific_research_agent` source package, so that package is retained in full to preserve its eager imports and internal research structure. Its old orchestration is a dependency/compatibility source, not the advertised six-role entry point.

Neither original source directory has an independent Git checkout. Local backup manifests record earlier experiments before D-controlled plan approval and before later R/D redesigns, but they are descriptive snapshots rather than attribution commits. No commit history has been invented or transplanted. The new showcase starts as a local Git repository with no remote; its original files remain unmodified outside the copy.

## Copy and edit boundary

The [manifest](../provenance/source_manifest.json) records original logical paths, source-group identities, SHA256 values for original and copied files, and each copy's status. Of 48 copied files, 47 are byte-identical. One configuration block in `src/scientific_infrastructure.py` replaces personal absolute database defaults with paths under `DEEPRESEARCH_DATA_DIR` (default `data/private`). Source IDs, header-row mappings and research functions remain intact. The live showcase entry point supplies explicit database paths rather than using those legacy defaults.

The new additions are README files, method/tool/experiment/run/validation/provenance documentation, a small SVG identity diagram, package metadata and dependency lock, blank environment template, Git exclusions, local QA and CI, two offline examples, an explicit live launcher and small sanitized verification records. These are repository-preparation work and must not be described as historical research innovations.

## Implemented project work versus dependencies

|Scope|Evidence-supported description|
|---|---|
|Research organization|Local role contracts, epistemic-state updates, task boundaries, D approval of R plans, context projections, frozen explanation assessment and evidence-matrix assembly are implemented in the copied project modules.|
|Engineering integration|LangChain provider adapters, LangGraph graph/subgraph execution, ToolNode, Send, SQLite checkpoint injection, registered local tables and structured tool wrappers.|
|Scientific computation|Registered analyses call established numerical methods and libraries. Wrapping KDE, GMM, dip tests, regression or bootstrap is scientific-tool integration, not invention of those algorithms.|
|Research exploration|Mixed iterative organization and fixed-U organization are alternative implementations. Historical traces and deterministic tests establish execution behavior; comparative benefit has not been measured.|
|Personal attribution|The implementation is present locally. Exact personal authorship, collaborator boundaries and any copied upstream project components require owner confirmation because independent Git evidence is absent.|

## External materials excluded

No private databases, downloaded literature/workbooks, model credentials, provider response dumps, large JSONL traces, checkpoint databases, virtual environments or historical backups are included in the candidate publication files. Public evidence exports retain hashes and aggregate statuses. The raw sources remain at their original locations and are listed in the separate local review report for the owner.

## Before public release

Confirm author names and contribution wording, ownership and redistribution permissions for the local source package and any reused project components, and the intended license. Check whether source-data names and experiment scope may be discussed publicly. A heuristic token/path scan and payload review are part of local QA; they do not establish legal permission or exhaustive absence of secrets. Publication is a separate owner action after reviewing this snapshot.
