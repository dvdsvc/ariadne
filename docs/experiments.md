# Experiments and evidence status

[Overview](../README.md) · [Methods](methods.md) · [Validation](validation.md) · [Historical audit JSON](../evidence/historical_runs.json)

## Offline checks of implemented behavior

The original tests exercise executable research mechanisms rather than merely document schemas: matching a claimed task to D's issued identity; D approval before an R plan takes effect; bounded DAG modification; tool-batch execution; input-error repair; scientific versus transient message projection; dependency-scoped data access; fixed C/T completion before reduction; independent-stage fan-out; dependent-stage release; provider-cache accounting; explicit field selection; runtime provenance; and numerical scientific tools.

Their model responses are scripted. Thus a graph can reach W deterministically without demonstrating that a real model will plan correctly or produce a sound scientific answer. The [validation page](validation.md) records the actual local commands and results.

## Historical live runs

Twelve overview artifacts were inspected, together with three selected raw JSONL traces. These are archival process records with different failures and checkpoint continuations, not twelve independent benchmark trials. A saved `running` status is an unfinished historical record and does not mean a process is currently active. The public [JSON summary](../evidence/historical_runs.json) contains filenames, SHA256 values, statuses and selected structural counts; raw prompts, observation rows, checkpoints and provider payloads are excluded.

|Selected trace|Observed execution|Interpretation|
|---|---|---|
|`mix_tasks_trace_20260903_193956.jsonl`|178 event records; 88 completed task nodes; one completed Writer node; 2,513-character nonempty answer.|A mixed-graph checkpoint continuation reached a written answer. The task-start payload contains a recovered zircon-isotope/ice-age constitution, while launch metadata contains an earlier descriptive-age-peak question. This is a recovered-task execution record, not a clean fresh run of the launch text.|
|`mix_tasks_trace_20260903_2231_lhasa_models_restart.jsonl`|326 event records; 160 completed task nodes; failed R progress/researcher nodes; no Writer answer.|Several technical episodes executed, followed by failure. Tool activity does not establish final scientific closure.|
|`fixed_u_trace_20260908_113841_resume_4.jsonl`|323 event records; 151 completed task nodes; two completed department-plan nodes; four completed technical finalize/researcher nodes; no Writer answer.|Parallel projects ran, but a local DAG contained an invalid dependency and other branches also failed. The recorded error includes `R plan dependencies must reference other stages in the same DAG`.|

Counts refer to task-end events observed in each selected segment, including nested graph tasks. They are not distinct scientific experiments or unique tool invocations. Repeated message payloads can repeat tool-call representations; only observed tool **names** are exported. Those three traces include `data__tool`, `archive__read` and `sandbox__python`. They do not establish that every registered scientific tool was used in those runs.

## What the completed run supports

The completed segment verifies that a real provider-backed, checkpoint-restored mixed system performed technical operations, updated higher-level state and reached Writer. Its answer is an artifact produced by the agent. This showcase does not independently recompute that answer's geological statistics, inspect every source observation or certify its causal interpretation. The final answer is represented by its length and hash rather than republished as a research finding.

The trace predates the current copied source files and does not record a source manifest or commit. Its SHA256 identifies the archived trace; it cannot bind that execution to the exact current source revision. Local protocol tests verify the current copied graph separately.

## Research findings about the system

The code and runs expose useful system-design questions. Iterative U/P selection can revisit a scientific agenda; a frozen C/T organization provides a different termination structure but still relies on valid R plans and model-generated assessments. A checkpoint can preserve valuable work while also carrying a task different from a new launch prompt. Runtime-owned identity prevents fabricated evidence identifiers, but does not by itself guarantee correct interpretation. The repository makes these questions inspectable through source, tests and bounded trace summaries.

No controlled mixed-versus-fixed-U accuracy, cost, convergence or ablation result is available. A credible next evaluation would freeze questions, data scopes, model settings and scoring criteria, then distinguish numerical correctness, provenance validity, completion, explanation discrimination and cost. This is an evaluation proposal, not a completed experiment.
