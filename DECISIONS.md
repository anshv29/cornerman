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

This only spells out what the locked taxonomy implies. It doesn't add or remove any states. Ansh approved this pairing on 2026-10-17.

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
`data/.gitignore` ignores everything under `data/` except `data/fixtures/`, itself, and DVC pointer files (`data/*.dvc`). The rules live in `data/.gitignore` rather than the root `.gitignore` because with `/data/*` in the root file, DVC stopped finding `data/derived.dvc` even though git still tracked it. `tests/unit/test_gitignore.py` pins the rules. It's an allow-list rather than a block-list so that new files such as `fights.csv`, `fighters.csv` or anything else that could hold real fighter names are never committed by accident. This only works if the fixtures stay synthetic, with anonymized IDs.
Secret scanning runs in two places. pre-commit runs `detect-private-key` (from pre-commit-hooks) and gitleaks on the staged changes. CI runs gitleaks over the full git history plus `detect-private-key` on all files, so a commit made with `--no-verify` still gets caught. In CI we download the pinned gitleaks binary from its GitHub release instead of using `gitleaks-action`, because that action moved to a non-MIT license from v2 on, and the plain binary does the same scan.

## 2026-10-17: Nothing on Kaggle is ever public, enforced in code
Every Kaggle CLI call goes through `cornerman.remote.kaggle`. It refuses `--public` / `-u` on any command, requires `"is_private": true` in a kernel's metadata before a push, and after each push pulls the metadata back from Kaggle and checks it again. A test checks every `kernel-metadata.json` under `kaggle/`. Kaggle datasets are private by default and we never pass the public flag. Reason: a public kernel or dataset could redistribute fight video (hard rule 2) or expose real fighter names. Video links are filled into kernels at build time from a local file, so they never sit in git.
The Kaggle CLI runs through `uvx --from kaggle==2.2.4`, so it is pinned and needs no global install. Login is OAuth (`kaggle auth login`), so there is no `kaggle.json`.

## 2026-10-17: Adult bouts only
We skip any bout that looks like it involves a minor: youth, junior and cadet divisions, and anything with "kids", "children", "U16"/"U18"-style age groups or an explicit age under 18 in the title or description. Junior is skipped even though some federations use it for 18 to 20 year olds, because we can't tell from a title which definition applies. On top of that, any fighter who appears in a skipped bout is dropped from the whole dataset, including their adult bouts, since we can't be sure when they turned 18. The filter lives in `cornerman.collect` and has its own tests. It runs before anything is downloaded, so video of a skipped bout never touches the laptop or Kaggle.

## 2026-10-17: Perception models (first pick, to be checked in the pilot)
Constraints: fits a Kaggle T4 (16 GB), installs with pip, open weights, and a license that's fine for a public repo. That last point rules out Ultralytics YOLO and BoxMOT (both AGPL-3.0) and Sapiens (CC BY-NC).
- Shot boundaries: TransNetV2 (`transnetv2-pytorch` 1.0.5, MIT). A small learned model trained for cuts and gradual transitions, which simple frame-difference methods (PySceneDetect's content detector) miss. PySceneDetect stays as a fallback.
- Person detection: RF-DETR Large (`rfdetr` 1.11.2, Apache 2.0). 56.5 COCO AP at about 7 ms on a T4 with TensorRT. The Apache-licensed sizes are N to L; XL and 2XL use a different license, so we stay at L. The segmentation variant (also Apache) is a candidate later for clinch and ground, where masks separate tangled bodies better than boxes.
- Tracking: BoT-SORT from Roboflow `trackers` 2.6.1 (Apache 2.0). Best of the light trackers in that package on SportsMOT (73.8 HOTA) and DanceTrack (57.8), and it compensates for camera motion, which matters for handheld amateur footage. McByte scores higher but pulls in SAM and Cutie; we only ever track three people, and identity across cuts is handled by our own identity stage, so the extra weight isn't worth it yet.
- 2D pose: ViTPose++ Large through Hugging Face `transformers` (`usyd-community/vitpose-plus-large`, Apache 2.0). ViTPose is the accuracy leader among permissive top-down models on COCO keypoints. RTMPose through `rtmlib` (Apache 2.0) is faster and is the fallback if throughput is too low.
- 3D body mesh for grappling: not chosen yet. Comes when Phase 3 shows where 2D breaks.
Versions are pinned in the `gpu` dependency group and model weights by Hugging Face commit. Throughput and the measurable proxies (both fighters tracked, identity switches per minute) from the 5-fight pilot decide whether any of these get swapped.

## 2026-10-17: Download inside the Kaggle job (smoke test passed)
The smoke kernel (`kaggle/smoke/`) ran on a Tesla T4 with internet on. yt-dlp 2026.08.19 fetched metadata for two public amateur fights, downloaded one full 163 s fight at 720p (55 MB in 5 s) and a 30 s section of the other. No bot check or 403. So the GPU job downloads its own video, processes it and deletes it, and video never touches the laptop.
yt-dlp warned that no JavaScript runtime was found and that YouTube extraction without one is deprecated. The job installs the `deno` pip package next to yt-dlp so this keeps working when YouTube tightens up.
The fallback stays documented but unbuilt: if Kaggle downloads start failing, download on the laptop and upload a private Kaggle dataset per batch (through the same privacy-checked wrapper), then delete the dataset version once the batch is processed. The perception code reads a local video file either way, so only the download step would change.

## 2026-10-17: Support Python 3.13 as well as 3.12
The smoke test showed Kaggle runs Python 3.13 with torch 2.11 (CUDA 12.8) preinstalled. Installing our package into that Python reuses Kaggle's torch instead of downloading a second 2 GB copy into a 3.12 environment on every run. `requires-python` is now `>=3.12,<3.14`. The laptop and the main CI job stay on 3.12 (the `.python-version` pin), and a second CI job runs the tests on 3.13.
