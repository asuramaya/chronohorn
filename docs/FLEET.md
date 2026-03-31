# Fleet

`Chronohorn` now treats multi-machine execution as part of the live stack, not
as ad hoc shell history.

This surface exists for three concrete reasons:

- faster wall-clock frontier iteration
- reproducible runtime measurements
- disciplined snapshot and artifact economics

The backend model is deliberately heterogeneous:

- `cpu`: Linux snapshot jobs on `slop-*` for Rust artifact builders and full-val eval
- `metal`: local Mac jobs for MLX causal-bank training and bridge probes
- `cuda`: remote Linux container jobs for CUDA-native work when a real GPU-shaped job exists

The important constraint is that these backends share one control surface, not one fake runtime.

The runtime also keeps a more stable hardware taxonomy for planning:

- execution backend
  - `cpu`, `metal`, `cuda`
- accelerator family
  - `cpu`, `apple`, `nvidia`
- accelerator architecture
  - examples: `apple-silicon`, `nvidia-rtx-a4000`

That lets `Chronohorn` optimize placement without baking one device story into
the manifest format.

The optimization orientation is explicit:

- keep hardware busy with mutation search
- spend serial lanes on held-out bpb measurement
- spend wide CPU lanes on artifact/compiler mutations
- spend GPU lanes on dense training or bridge mutations

## Why

The hardware is not homogeneous:

- this Mac is Apple Silicon and `Metal` / `MLX`
- `slop-01` and `slop-02` are Linux + NVIDIA

Trying to pretend they are one interchangeable GPU fleet creates more complexity
than it removes.

The right split is:

1. one manifest
2. backend-specific launchers
3. disposable worktree snapshots on remote Linux boxes

The snapshot part matters. `Chronohorn` should stage the minimum runnable slice,
not copy unrelated repos or whole local data trees to remote nodes.

## Launch Surface

The promoted launcher is `python -m chronohorn fleet`, implemented in
[python/chronohorn/fleet/dispatch.py](../python/chronohorn/fleet/dispatch.py).

The legacy source-tree wrapper [scripts/dispatch_experiment.py](../scripts/dispatch_experiment.py)
just dispatches into that package surface.

It reads a JSONL manifest and launches jobs by `launcher` type.

Current supported launchers:

- `local_command`
  - local detached process
  - good for `metal` MLX jobs and local CPU probes
- `managed_command`
  - promoted generic launcher
  - runs locally when assigned to `local`
  - otherwise snapshots the tree and runs the same command remotely in Docker
  - preferred when a row should be movable across Apple / CPU / NVIDIA lanes
- `slop_causal_bank_eval_from_table`
  - wraps [scripts/slop_run_causal_bank_from_table.zsh](../scripts/slop_run_causal_bank_from_table.zsh)
  - remote snapshot + Rust container + full-val eval
- `slop_oracle_budgeted_build`
  - wraps [scripts/slop_build_oracle_budgeted_table.zsh](../scripts/slop_build_oracle_budgeted_table.zsh)
  - remote snapshot + Rust container + artifact build
- `slop_docker_command`
  - generic remote snapshot + detached Docker job
  - optional `gpu=true` for CUDA lanes

The dispatcher now also supports live placement:

- `host: "auto"`
  - choose a slop host at launch time from current fleet state
- `hosts: ["slop-01", "slop-02"]`
  - optional candidate set for auto placement
- `min_available_mem_gb`
  - refuse placement if the chosen host is below the memory floor
- planner telemetry
  - uses measured `tokens_per_second` and estimated sustained TFLOPs from real Chronohorn result JSONs
- `planner.reason`
  - explains why a host was selected
- `planner.predicted_seconds`
  - included when a row declares a token budget and matching telemetry exists
- `--status`
  - print current fleet state plus planned assignments without launching
- `--dry-run`
  - resolve placement and show the launch plan without starting jobs
- `--telemetry-glob`
  - add extra result JSON search paths for planning without changing code

It also suppresses duplicate launches:

- if a remote container with the same job name is already up, the row is reported as `already_running`
- if a local detached job still has a live PID in its launch record, it is also treated as `already_running`

Status is intentionally folded back into the dispatcher and internal probes:

- `python -m chronohorn fleet --status`
- `python -m chronohorn fleet --dry-run`

That keeps fleet state on one control surface instead of maintaining a separate user-facing watcher layer.

The important shift is that the fleet layer is no longer just “pick a free box.”
It is becoming a runtime planner that explains hardware assignment in terms of
speed, fit, and measured efficiency.

## Backend Guidance

### CPU

Use `cpu` for current `Chronohorn` work:

- table builds
- packed-compiler stages
- eval matrices
- audit runs

These paths are dominated by Rust, hashing, sparse tables, and causal replay. They do not need GPUs.

### Metal

Use `metal` for local MLX descendant work on this Mac:

- `python -m chronohorn train train-causal-bank-mlx`
- local static-bank-gate queue commands built on `python -m chronohorn train queue-static-bank-gate`
- other backend-specific `chronohorn` MLX jobs

This is the immediate honest way to use the Mac GPU.

When launched from Codex itself, `metal` work may need unsandboxed execution. The backend is real; the limitation is the sandboxed tool context, not `MLX`.

### CUDA

Use `cuda` for Linux container jobs that actually benefit from NVIDIA GPUs:

- future CUDA-native distillation or teacher jobs
- PyTorch/HF ablations
- generic GPU probes and benchmark jobs

Current `Chronohorn` Rust builders are not good GPU candidates. Do not burn
slop GPUs on those just because they are available.

## Remote Model

Remote Linux jobs are snapshot-based:

1. sync the current dirty worktree
2. run in `/tmp/chronohorn-runs/<job>`
3. use Docker containers
4. keep outputs in remote scratch
5. pull back only logs / artifacts

This avoids:

- branch drift
- remote git state
- accidental edits to the source tree

Prefer curated `snapshot_paths` over syncing the whole repository root whenever
possible. The dispatcher supports that path specifically so runtime
organization improves launch speed instead of just documenting it.

When a `chronohorn` training or parity row depends on the `opc` kernel, stage
both surfaces explicitly:
- `source_dir`: monorepo root
- `snapshot_paths`: `chronohorn/python`, `chronohorn/data/...`, `open-predictive-coder/src`
- `remote_cwd_rel`: `chronohorn`

Do not assume `open_predictive_coder` is preinstalled on the node.

## Manifest Schema

Every manifest row is one JSON object.

Required:

- `name`
- `backend`
- `launcher`

Recommended research metadata:

- `resource_class`
  - `cpu_serial`, `cpu_wide`, `metal_gpu`, `cuda_gpu`
- `goal`
  - one-sentence statement of the bpb question the row answers
- `workload_kind`
  - examples: `training.frontier`, `training.parity`, `artifact.build`, `evaluation.fullval`
- `work_tokens`
  - explicit token budget for duration prediction

Launcher-specific fields:

### `local_command`

- `cwd`
- `argv` or `command`
- optional `env`
- optional `log_path`

### `managed_command`

- `command`
- optional `host`
- optional `hosts`
- optional `source_dir`
- optional `remote_cwd_rel`
- optional `env`
- optional `threads`
- optional `gpu`
- optional `image`
  - required only when the row may land on a remote Docker host

Use this when the row should be expressed once and the planner should decide
whether to keep it local or stage it remotely.

### `slop_causal_bank_eval_from_table`

- `host`
- or `host: "auto"`
- `checkpoint_path`
  - bundle directory or legacy `.npz`
- `summary_path`
  - summary JSON paired with the checkpoint source
- `artifact_bin`
- optional `threads`
- optional `val_tokens`
- optional `report_every`

Compatibility:

- `checkpoint_npz` still works as an alias for `checkpoint_path`
- `checkpoint_json` still works as an alias for `summary_path`

### `slop_oracle_budgeted_build`

- `host`
- or `host: "auto"`
- `train_tokens`
- `profile`
- optional `threads`
- optional `report_every`
- optional `oracle_stride`

### `slop_docker_command`

- `host`
- or `host: "auto"`
- `image`
- `command`
- optional `source_dir`
- optional `remote_cwd_rel`
- optional `sync_paths`
- optional `env`
- optional `threads`
- optional `gpu`

## Examples

Public portable example:

- [manifests/fleet_example.jsonl](../manifests/fleet_example.jsonl)

This manifest is intentionally local-only and uses environment-variable paths so
it can be copied without assuming `slop-*` hosts or `/Users/...` roots. It now
uses the generic `managed_command` surface rather than teaching only raw local
process rows.

Machine-local lab example:

- [manifests/fleet_lab_example.jsonl](../manifests/fleet_lab_example.jsonl)

This is a real internal lab manifest with host-local paths and `slop-*` usage.
Keep it for working research, but do not treat it as the public baseline.

Launch the whole manifest:

```bash
PYTHONPATH=python python3 -m chronohorn fleet \
  --manifest manifests/fleet_example.jsonl
```

Launch one job from the public smoke manifest:

```bash
PYTHONPATH=python python3 -m chronohorn fleet \
  --manifest manifests/fleet_example.jsonl \
  --job chronohorn-help
```

Inspect the fleet and planned placement:

```bash
PYTHONPATH=python python3 -m chronohorn fleet \
  --manifest manifests/bpb_research_queue.jsonl \
  --class cpu_serial \
  --status
```

Launch a whole research class:

```bash
PYTHONPATH=python python3 -m chronohorn fleet \
  --manifest manifests/bpb_research_queue.jsonl \
  --class cpu_wide
```

## Resource Classes

### `cpu_serial`

Use for causal full-val eval rows that cannot saturate a box by themselves.

Policy:

- run many at once
- keep threads low, often `1`
- use these to turn idle cores into more bpb measurements

### `cpu_wide`

Use for artifact builders and compiler stages that parallelize internally.

Policy:

- one per box or one plus a small serial tail
- set `CHRONOHORN_THREADS` high
- use these for table/compiler mutations, not final eval

### `metal_gpu`

Use for local MLX causal-bank mutations on the Mac.

Policy:

- one dense job at a time
- keep it aligned with current lower-bpb hypotheses, not random smoke tests

### `cuda_gpu`

Use for Linux GPU rows once a real CUDA-native compression mutation exists.

Policy:

- one job per GPU
- do not waste this lane on current CPU-shaped Rust builders

## Lower-Bpb Default

The default fleet reading of `Chronohorn` should now be:

1. keep at least one `cpu_wide` artifact mutation alive
2. fill the spare CPU cores with `cpu_serial` held-out evals
3. keep the Mac on one `metal_gpu` mutation
4. only add `cuda_gpu` rows when they answer a real compression question

Dynamic placement should follow the same rule:

- `cpu_serial`: send the next row to the host with the most spare serial capacity
- `cpu_wide`: send the next builder to the host with the most free RAM and no active wide lane
- `metal_gpu`: keep local, but refuse launch under memory pressure
- `cuda_gpu`: pick the host with an actually free GPU

When telemetry exists, refine that rule with measured throughput:

- prefer the host with the lowest predicted duration for the declared `work_tokens`
- otherwise prefer the host with the strongest matching TFLOPs sample
- otherwise fall back to the capacity heuristics above

That is how the tooling should spend hardware in service of lower bpb rather than idle utilization for its own sake.

## Current Reading

`Chronohorn` now has one orchestration surface across all available hardware, but it does not lie about backend truth:

- `metal` is local MLX
- `cpu` is remote Rust/container work
- `cuda` is remote GPU container work

That is the correct way to saturate the hardware without pretending the Mac and slop boxes are the same machine.
