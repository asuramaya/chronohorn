# Chronohorn Python

This tree is the Python-side descendant layer for `Chronohorn`.

It is the main Python surface for the system repo:

- model-family implementations
- training/export entrypoints
- fleet helpers that belong to the system repo

It is not the shared kernel.

Reusable substrate, memory, routing, and readout mechanisms should live in
`opc` when they are genuinely family-neutral. Descendant-specific policy stays
here.

This means `chronohorn/python` owns the Python code that is specific to making
descendants runnable, trainable, exportable, and optimizable on real backends.

It is meant to stand on its own as part of the public `chronohorn` repo.
It should not require importing legacy sibling monorepo packages such as
`conker` or `carving_machine` in order to run its promoted surfaces.

It does build on the public `opc` kernel. Backend-neutral causal-bank family
logic now lives in `open_predictive_coder`, while backend-specific model and
training execution stays here.

Backend-specific architecture stays here. The MLX and Torch/NVIDIA causal-bank
implementations belong to `chronohorn`, even when parts of the family become
stable enough to inspire backend-agnostic abstractions in `opc`.

## Install

From the repo root:

```bash
python3 -m pip install -e .[train]
```

Optional extras:

- `.[torch]` for Torch/CUDA
- `.[metal]` for MLX/Metal

For sibling-repo work inside the shared workspace:

```bash
python3 -m pip install -e ../open-predictive-coder -e .[train]
```

Validated isolated package smoke:

```bash
uv venv .venv
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/chronohorn --help
.venv/bin/chronohorn fleet --help
.venv/bin/chronohorn observe --help
.venv/bin/chronohorn train --help
.venv/bin/chronohorn export --help
.venv/bin/chronohorn mcp --help
.venv/bin/chronohorn-mcp --help
```

That smoke does not validate backend extras or the kernel dependency. It proves
the installed package surface exposes its help and entrypoint layer cleanly.

The package surface is intentionally light but structured:

- `python -m chronohorn`
  - top-level entry surface
- `python -m chronohorn train`
  - package-level train surface
- `python -m chronohorn export`
  - export bundle surface
- `python -m chronohorn fleet`
  - package-level fleet launch and status surface
- `python -m chronohorn observe`
  - package-level runtime observer/store surface
- `python -m chronohorn mcp`
  - package-level MCP stdio server for agent access to runtime state
- `python -m chronohorn.export`
  - direct export package runner
- `python -m chronohorn.fleet`
  - direct fleet package runner
- `python -m chronohorn.observe`
  - direct observer/store package runner
- `python -m chronohorn.train`
  - direct train package runner
- `python -m chronohorn train train-causal-bank-mlx`
  - MLX/Metal causal-bank training on token shards
- `python -m chronohorn train train-causal-bank-torch`
  - Torch/CUDA causal-bank training on token shards
- `python -m chronohorn train sweep-static-bank-gate`
  - restartable static-bank-gate plateau sweep
- `python -m chronohorn train queue-static-bank-gate`
  - local static-bank-gate queue with lock/log handling
- `python -m chronohorn train measure-backend-parity`
  - backend parity measurement on a deterministic fixed batch

The promoted train and parity summaries now write a shared performance block:

- `performance_estimate`
  - analytical FLOP estimate for the configured causal-bank variant
- `performance`
  - observed throughput plus estimated sustained and interval TFLOPs
- `performance_log`
  - per-log interval performance rows at the training logging cadence

Promoted training result JSONs now also carry a shared forecast block:

- `forecast`
  - budget-limited projection for the current run
  - default budget name: `golf_v1`
  - default training budget: `9,500,000` TFLOPs
  - default artifact limit: `16 MB`
  - includes projected budget-step limit, projected wallclock, projected metric at budget, and artifact-budget viability

The promoted fleet surface consumes those same measurements for placement:

- it infers hardware family such as `apple` or `nvidia`
- it matches jobs against recorded throughput and TFLOPs
- it reports predicted duration when a row declares `work_tokens`
- it keeps the execution backend (`cpu`, `metal`, `cuda`) separate from the hardware family taxonomy
- it can project existing result JSONs onto the Golf budget:
  - `python -m chronohorn fleet forecast-results --path <result-or-dir>`
  - `python -m chronohorn fleet forecast-results --glob 'out/**/*.json'`
  - the public forecast rows expose compute-axis utilization, probe-density overhead, uncertainty bands, and a conservative decision signal

The promoted observer surface now normalizes runtime state into one small record store:

- `python -m chronohorn observe pipeline`
  - ingest manifests, launch records, results, and forecast rows into a single run store
- `python -m chronohorn observe status`
  - summarize merged run state for terminal use
- `python -m chronohorn observe query-records`
  - filter raw runtime records by kind/source/family/name/status
- `python -m chronohorn mcp`
  - expose the same runtime store and pipeline through a Heinrich-style MCP tool server

The observer layer exists so Chronohorn can be agent-native without duplicating Heinrich's role:

- Chronohorn observer/MCP owns runtime facts, planning state, queue state, result state, and forecasts
- Heinrich still owns external evidence packaging, validation bundles, and public-facing audit compression

The public Python surface is intentionally narrow. Package-level imports are
not treated as a broad stable API. For backend code, import the concrete
modules directly instead of relying on package-level re-export shims.

The internal structure is now moving toward the same split that `opc` uses for
kernel concerns:

- `chronohorn.engine`
  - generic runtime concerns
  - named competition budgets
  - result-summary extraction
  - budget forecasting
- `chronohorn.store`
  - normalized run records and merged run snapshots
- `chronohorn.pipeline`
  - stage-oriented runtime observer pipeline built on the store
- `chronohorn.families`
  - descendant family adapters
  - family-specific scan and training policy
- `chronohorn.train`
  - public train entrypoints and backend-specific runners
- `chronohorn.fleet`
  - orchestration and planner surfaces built on the engine layer
  - public forecast projection and decision-signal wrappers
- `chronohorn.observe`
  - terminal observer surface over manifests, launches, results, and forecasts
- `chronohorn.mcp`
  - tool registry for the Chronohorn MCP runtime server

The same rule applies to the export package. Import concrete helpers from:

- `chronohorn.export.bundle`
- `chronohorn.export.schema`
- `chronohorn.export.abi`

The command names above are the canonical public surface. Older alias-style
train names were removed so the package presents one stable command vocabulary.

The model layer is split by responsibility:

- `open_predictive_coder.causal_bank`
  - backend-neutral causal-bank family metadata, config, variant application, and frozen substrate construction
- `chronohorn.models.readouts_mlx`
  - MLX readout implementations
- `chronohorn.models.readouts_torch`
  - Torch readout implementations
- `chronohorn.models.causal_bank_mlx`
  - MLX backend model
- `chronohorn.models.causal_bank_torch`
  - Torch backend model

The training layer is now formalized around:

- `chronohorn.train.causal_bank_training_stack`
  - typed backend-specific training stack for the promoted causal-bank family
- `chronohorn.train.causal_bank_training_primitives`
  - backend-neutral training argument/runtime/config builders
- `chronohorn.families.causal_bank.adapter`
  - family-specific hooks for config validation, replay fixtures, performance estimation, and export wiring
- `chronohorn.engine.forecasting`
  - generic budget-limited projection for Chronohorn result JSONs

For monorepo source-tree work, Chronohorn will import a sibling
`open-predictive-coder/src` tree automatically if present. Public installs
should rely on the packaged `open-predictive-coder` dependency instead.

That keeps the public training path closer to named model and training units,
instead of ad hoc script-specific glue.

Intended package split:

- `chronohorn/`
  - descendant package
- `chronohorn/models/`
  - concrete model-family implementations
- `chronohorn/train/`
  - training entry surface and bridge implementations
- `chronohorn/export/`
  - export bundle ABI and export CLI
- `chronohorn/fleet/`
  - Python-side orchestration, telemetry, and runtime planning helpers
- `chronohorn/observe/`
  - runtime store/status/query CLI
