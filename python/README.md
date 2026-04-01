# Chronohorn Python

Python runtime surface for `Chronohorn`.

If `opc` is the shared kernel, this tree is the descendant execution layer:
backend-specific models, trainers, fleet logic, observer state, and MCP access.

## What It Does

`chronohorn/python` owns the Python code that turns descendant families into
real systems:

- MLX and Torch training
- fleet placement and queueing
- runtime observation and forecasting
- closed-loop frontier control
- export bundle emission
- agent-facing MCP access to run state

It is not the kernel. Reusable predictive mechanisms still belong in
`open-predictive-coder`.

It also does not replace `heinrich`. Chronohorn's Python side now has a
Heinrich-shaped store/pipeline/MCP layer, but that layer is for runtime facts
and decisions, not external evidence packaging.

## Install

From the repo root:

```bash
python3 -m pip install -e .[train]
```

Optional extras:

- `.[torch]` for Torch/CUDA
- `.[metal]` for MLX/Metal

For sibling-repo work:

```bash
python3 -m pip install -e ../open-predictive-coder -e .[train]
```

Validated package smoke:

```bash
uv venv .venv
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/chronohorn --help
.venv/bin/chronohorn fleet --help
.venv/bin/chronohorn control --help
.venv/bin/chronohorn observe --help
.venv/bin/chronohorn train --help
.venv/bin/chronohorn export --help
.venv/bin/chronohorn mcp --help
.venv/bin/chronohorn-mcp --help
```

## Package Surface

```bash
python -m chronohorn train ...
python -m chronohorn fleet ...
python -m chronohorn control ...
python -m chronohorn observe ...
python -m chronohorn export ...
python -m chronohorn mcp
```

Canonical MCP entrypoint: `python -m chronohorn mcp`

Installed console-script alias: `chronohorn-mcp`

Important commands:

- `python -m chronohorn train train-causal-bank-mlx`
- `python -m chronohorn train train-causal-bank-torch`
- `python -m chronohorn train measure-backend-parity`
- `python -m chronohorn fleet dispatch --manifest <manifest.jsonl>`
- `python -m chronohorn fleet drain --manifest <manifest.jsonl>`
- `python -m chronohorn fleet queue --manifest <manifest.jsonl>`
- `python -m chronohorn fleet transform --manifest <manifest.jsonl> --filter 'ex-j-*' --steps 5200 --output <out.jsonl>`
- `python -m chronohorn fleet forecast-results --path <result.json>`
- `python -m chronohorn fleet emit-causal-bank-matrix --regime exotic-16mb`
- `python -m chronohorn control recommend --manifest <manifest.jsonl>`
- `python -m chronohorn control act --manifest <manifest.jsonl>`
- `python -m chronohorn observe pipeline --manifest <manifest.jsonl>`
- `python -m chronohorn observe status --manifest <manifest.jsonl> --probe-runtime`
- `python -m chronohorn observe frontier --manifest <manifest.jsonl> --probe-runtime`
- `python -m chronohorn observe query-records --kind runtime_state`
- `python -m chronohorn mcp`

## Runtime Record Pipeline

The Python observer side now follows a small-stage pattern:

```text
tracked_state -> manifest -> runtime_state -> live_log -> launch -> result -> forecast
```

That data lands in:

- `chronohorn.store`
  - `RunRecord`
  - `RunStore`
  - merged `RunSnapshot`
- `chronohorn.pipeline`
  - stage runner that builds the store
- `chronohorn.observe`
  - terminal view over the store
  - includes raw frontier, feasible frontier, and tracked runtime notes from shared state
- `chronohorn.control`
  - closed-loop controller over manifests, store snapshots, and live fleet state
- `chronohorn.mcp`
  - agent-facing tool server over the same store

## Modules

- `chronohorn.engine`
  - budgets, performance accounting, result summaries, probes, forecasting
- `chronohorn.families`
  - family adapters and scan policy
- `chronohorn.models`
  - backend-specific model implementations
- `chronohorn.train`
  - trainers and backend runners
- `chronohorn.fleet`
  - placement, queueing, telemetry, forecast wrappers
  - `drain` — unattended manifest execution with auto re-dispatch and result pull-back
  - `results` — SSH-based result pull-back from remote containers
  - `auto_deepen` — slope-based auto-generation of next-horizon manifest rows
  - `manifest_transform` — filter and mutate manifest rows without editing scan code
- `chronohorn.control`
  - action ranking, promotion policy, and execution
- `chronohorn.observe`
  - observer CLI
  - `serve` — HTTP visualization server (6 tabs: curves, frontier, fleet, bpb/tf, config, manifests); Chrome app mode auto-launch; `/api/action` endpoint
- `chronohorn.export`
  - bundle ABI and export CLI
- `chronohorn.store`
  - normalized runtime record schema
  - `IncrementalStore` — hot-cached result layer for the runtime daemon
- `chronohorn.runtime_store`
  - hot-path store used by the unified runtime daemon
- `chronohorn.pipeline`
  - stage-oriented runtime observer pipeline with local result cache
- `chronohorn.runtime`
  - unified daemon: drain + fleet probe + viz + auto-deepen in one process
- `chronohorn.mcp`
  - tool registry for the Chronohorn MCP server (21 tools)

## Family / Engine Split

The internal split now mirrors the public regime:

- `chronohorn.engine`
  - generic runtime policy
  - budgets
  - forecasting
  - result summaries
  - optimizer/runtime metadata
- `chronohorn.families`
  - descendant-specific hooks
  - scan policy
  - replay/export wiring
- `chronohorn.train`
  - backend runners using those layers

That keeps Chronohorn from turning into a second kernel.

## Boundaries

- `open-predictive-coder`
  - backend-neutral family semantics and reusable primitives
- `chronohorn`
  - runtime execution, orchestration, replay, observation, and export
- `heinrich`
  - external validation and evidence packaging

For the full ownership split, read:

- [../docs/REPO_BOUNDARY.md](../docs/REPO_BOUNDARY.md)
- [../docs/FLEET.md](../docs/FLEET.md)
