# Session 8 Technical Debt

Everything left broken, buggy, or avoided during the April 11 session.

## Broken

### 1. hein-fix-s8-exp2-bands4-bal01-50k re-launches on every dispatch
The job completed and results were pulled, but it's not registered as a DB job (dispatched via fleet_dispatch which writes launch records but the drain didn't ingest it as a job). Every subsequent `fleet_dispatch` on the heinrich-guided manifest re-launches it. **Fix:** either register it in the DB manually or add completed-result detection to the dispatch path.

### 2. hein-local16-s8-exp2-bands4-10k failed silently on slop-home
The run directory exists but has no results. No failure diagnostics captured because the failure happened before the drain could reconcile. The new `failure_reason`/`failure_log` columns exist but this job predates them. **Fix:** check k8s job logs manually, determine if it's an OOM (8GB GPU) or a code error.

### 3. Data provisioning init container untested
The k8s init container that auto-provisions training data from HuggingFace was added to `build_job_manifest` but has never actually fired on a real job. All recent jobs hit nodes that already had data. First job on a fresh node will be the real test. **Risk:** the inline Python script in the init container may fail due to pip install issues, HF rate limiting, or disk permissions.

### 4. slop-home SSH key setup is one-directional
Added slop-home's pubkey to slop-01/slop-02 authorized_keys, and added /etc/hosts entries on all nodes. But slop-01 and slop-02 don't have SSH keys of their own — they can't SSH to each other or to slop-home. Only slop-home→slop-01 and slop-home→slop-02 work. **Not urgent** since all fleet operations go through the local machine, but the init container's HF download bypasses this.

## Buggy

### 5. what_varied still has None-vs-value noise
Fixed the major issue (missing architecture axes) but some fields still show `None` vs actual value for runs with sparse json_blobs. The `_WHAT_VARIED_DEFAULTS` dict normalizes known fields but anything not in the dict still produces false variation. Runs ingested from result JSONs (not manifests) have sparse blobs.

### 6. accelerator_arch never populated
The `accelerator_arch` column was added to results and the code to populate it was written, but it reads from `job_json.accelerator_arch` which is only set by the fleet probe during dispatch. Jobs registered via `record_result()` from pulled result JSONs don't have fleet probe data in their job_json. All existing results have NULL accelerator_arch. **Fix:** backfill from fleet host mapping (host → known GPU type).

### 7. source_sha captured but not stored in DB
`_capture_source_sha()` adds `source_sha` to the job dict before launch, and the `source_sha` column exists on the jobs table. But `record_launch()` doesn't write `source_sha` to the DB — the column was added to the schema but the INSERT/UPDATE in `record_launch` wasn't updated. The SHA is in the launch JSON file on disk but not queryable.

### 8. Checkpoint registration only works on remote pull
The checkpoint path/host registration in `pull_remote_result` only fires when pulling from a remote host. Locally-produced results (e.g., `cb-s8-experts-50k` which was copied manually) don't get registered. The leader's checkpoint location is unknown to the DB.

### 9. Forecast validation not automated
The forecast bias measurement (0.18 bpb) was done manually via ad-hoc queries. Task #10 said to build automatic comparison when a result extends a prior forecast. The measurement was done but the automation was not built — no `forecast_validation` table, no automatic comparison on ingestion.

## Avoided

### 10. tied_recursive readout safety gate
Three jobs crashed because `tied_recursive` is gated behind `--allow-experimental-recursive-readout`. The preflight check now catches this, but the underlying question — is tied_recursive ready? — was never answered. The heinrich-guided builds proposed `tied_readout` (different mechanism), not `tied_recursive`. The safety gate remains.

### 11. The decepticons loader can't load depth>1 readouts
Three checkpoints on Sharts (bands4-depth2-scan512-s8, bands4-depth3-512m-s12-25k, hein-bal01-s8-exp4-256m-bands4-10k) can't be MRI'd because the loader only creates `RoutedSquaredReLUReadout`. `TiedRecursiveReadout` (depth>1) and MLP-style readout need loader support. **Impact:** can't dissect the 3-layer readout model (1.843 bpb, 7.3 MB) which could inform whether deeper readout helps extract slow-mode features.

### 12. sp4096 data pipeline
The retokenization from sp1024 to sp4096 was discussed but not started. The parameter-golf repo has the tokenizer and downloader. The chronohorn data provision system (`data/provision.py`) hardcodes sp1024 paths. Needs: retokenize FineWeb-10B at sp4096, update provision to support variant selection, update the data init container.

### 13. Tied readout implementation in decepticons
The proposed next architecture uses tied readout (project to embed space, multiply by embed.T for logits). This doesn't exist in decepticons yet. It's the critical path for the next experiment and was not started.

### 14. No config space coverage tool
Discussed as a gap — "which (substrate_mode, readout_kind, readout_bands) tuples have results and which are unexplored?" — but not built. Would help identify untried combinations systematically.

### 15. 140+ completed k8s jobs accumulating
Cleaned up once during this session but they'll accumulate again. No automatic TTL or cleanup. The namespace gets clogged which may cause dispatch failures (as seen).

### 16. Probe-vs-final eval gap not corrected in DB
Measured a ~0.10 bpb gap between 4-batch probes and 32-batch finals. The saturation analysis and learning curves use probe data. No correction is applied — the learning curves and forecast asymptotes are all based on pessimistic probe numbers. The `eval_batches` column is now tracked but not used to adjust.
