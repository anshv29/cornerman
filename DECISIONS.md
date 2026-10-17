# Decisions

Each entry: what was chosen, what else was considered, and why. Newest at the bottom.

## 2026-10-17: Python 3.12, managed by uv
Considered: system Python 3.9, Python 3.13.
Chose 3.12 because the current numpy and scipy need at least 3.12, and every library we expect to use later (PyTorch, NumPyro, pandera) supports it well. 3.13 still has occasional wheel gaps in the GPU stack. uv handles the interpreter, the virtualenv and the lockfile, so `uv sync --locked` gives the same environment on the laptop, in CI and on Kaggle.

## 2026-10-17: Pyright strict, with targeted ignores for third-party gaps
Strict mode is on for all of `src/` and `tests/`. pandas, scipy and pandera have typing gaps (partially unknown return types, overloads the stubs don't cover). We don't spend time fighting these. When one hits, we silence that single line with a rule-specific comment, e.g. `# pyright: ignore[reportUnknownMemberType]`, so the ignore is visible and scoped. We do not turn off any strict rule globally, because that would also hide real mistakes in our own code. `pandas-stubs` and `scipy-stubs` are installed to keep the number of these ignores small.

## 2026-10-17: Ruff and pyright in pre-commit run through uv
The hooks call `uv run ruff` and `uv run pyright` instead of pinning separate hook versions, so the commit hook, CI and the editor all use the version in `uv.lock`. A `no-video` hook fails any commit that includes a video file, on top of `.gitignore`, because hard rule 2 matters more than convenience.

## 2026-10-17: Reading of the state compatibility rule
The spec gives one example (ground-top pairs with ground-bottom). The full pairing used in `taxonomy.py`:
- standing-neutral with standing-neutral
- cage-pressuring with cage-pressured
- clinch-controlling with clinch-controlled
- ground-top with ground-bottom
- ground-scrambling with ground-scrambling
- standing-opponent-hurt with standing-neutral or standing-cage-pressured. The hurt fighter has no "hurt" state of their own, so they stay in whichever standing state they're in.

This only spells out what the locked taxonomy implies. It doesn't add or remove any states.

## 2026-10-17: A small custom DAG runner instead of Prefect or Dagster
Considered: Prefect, Dagster, DVC pipelines (`dvc.yaml`), Make.
The part we need is narrow: run stages per fight in dependency order, skip anything whose inputs haven't changed, rerun one stage for one fight, survive a killed Kaggle session, and quarantine bad fights. That fits in about 200 lines (`src/cornerman/pipeline/runner.py`), which we can test fully and explain in an interview. Prefect and Dagster bring a server, a UI and their own state store, which we don't need on a laptop or inside a Kaggle notebook. DVC pipelines cache at the whole-stage level, not per fight, so rerunning one fight would be awkward. If we later need scheduling or a UI, the stage functions can be wrapped by Prefect without changes.

Cache key = hash of stage name, stage version, stage source code, the config fields the stage declares, the seed, the fight ID, external source files, and the output hashes of upstream stages. Using upstream *output* hashes (not their keys) means a stage that reruns but produces the same bytes doesn't force its downstream to rerun.

## 2026-10-17: DVC tracks data/derived only; fixtures stay in git
`data/derived/` (pipeline outputs) is versioned with `dvc add`, which writes a small `data/derived.dvc` pointer file into git and keeps the bytes in the DVC cache. We use `dvc add` rather than a `dvc.yaml` pipeline because our own runner already does the per-fight caching, so DVC only needs to snapshot the result.
`data/fixtures/` stays in plain git. It is tiny, CI needs it for the golden test, and there's no DVC remote yet, so CI couldn't pull it from DVC anyway.
There is no DVC remote yet. Every free option (DagsHub, Google Drive, a Backblaze free bucket) needs an account only Ansh can create. Until then the DVC cache lives only on the laptop, which is fine for Tier 0 because the derived data can be rebuilt from the fixture in seconds. A remote must exist before real fights are processed on Kaggle.
DVC's anonymous usage analytics are turned off in `.dvc/config`.

## 2026-10-17: data/ is ignored by default, and secrets are scanned on every commit
`.gitignore` ignores everything under `data/` except `data/fixtures/`, `data/.gitignore` and DVC pointer files (`data/*.dvc`). It's an allow-list rather than a block-list so that new files such as `fights.csv`, `fighters.csv` or anything else that could hold real fighter names are never committed by accident. This only works if the fixtures stay synthetic, with anonymized IDs.
Secret scanning runs in two places. pre-commit runs `detect-private-key` (from pre-commit-hooks) and gitleaks on the staged changes. CI runs gitleaks over the full git history plus `detect-private-key` on all files, so a commit made with `--no-verify` still gets caught. In CI we download the pinned gitleaks binary from its GitHub release instead of using `gitleaks-action`, because that action moved to a non-MIT license from v2 on, and the plain binary does the same scan.
