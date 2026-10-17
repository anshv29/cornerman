# Progress

## Needs Ansh

1. **A free DVC remote (not blocking).** Ansh said on 2026-10-17 this can wait. Recommended: DagsHub (free, no card). Sign up at https://dagshub.com with GitHub, connect the `anshv29/cornerman` repo, then share the DVC remote URL and an access token with the agent, for example by putting the token in a gitignored `.env` file. Until then the DVC cache lives only on this laptop.

## Kaggle setup (done 2026-10-17)
- Account `anshvaishnav29`, phone verified, so notebooks get GPU and internet.
- Logged in with `kaggle auth login` (OAuth). The credentials live in `~/.kaggle/credentials.json`. There is no `kaggle.json` and none is needed.
- The CLI isn't installed globally. Run it as `uvx kaggle ...`.
- Every kernel and dataset we create must be private. The push helper checks this in code before anything is uploaded.

## Current spend
$0.00 of the $50 cap. Every dollar needs Ansh's OK before it's spent. See `costs.md`.

## Done

### Tier 0, foundation (finished 2026-10-17)
- Repo skeleton: uv with Python 3.12, `pyproject.toml` with `uv.lock`, ruff, pyright strict (targeted line-level ignores only for pandas/scipy/pandera stub gaps), pytest, hypothesis.
- pre-commit: ruff, pyright, hygiene hooks, a hook that blocks video files, `detect-private-key`, and gitleaks.
- CI on GitHub Actions: lint, format, types and tests, plus a secrets job (gitleaks over the full history, detect-private-key). Green.
- `data/` is allow-list only (fixtures, DVC pointers, `data/.gitignore`), pinned by a test.
- Taxonomy module with the state compatibility table.
- Typed config (pydantic-settings, YAML plus seed).
- Pipeline runner with content-hash caching, atomic stage writes, per-fight quarantine, and a `cornerman run` CLI. Six stub stages run on the synthetic fixture fight. A golden test checks its event log byte for byte.
- Event table contract (pandera) with quarantine on failure.
- DVC tracks `data/derived/` (no remote yet).
- Recovery harness (coverage with a binomial test, SBC with a chi-square test on ranks), proven on the toy Dirichlet-multinomial model with empirical Bayes:
  - correct posterior: 90.8% coverage at a 90% level (p = 0.25), SBC passes;
  - deliberately overconfident posterior: 76.8% coverage, SBC p < 1e-40, both flagged;
  - EB recovers alpha 2.01, 6.22 and 26.4 for true values 2, 6 and 25.
- Docs: README, LEARNING.md (7 sections), DECISIONS.md, costs.md, AUDIT.md placeholder.
- 53 tests, about 2 seconds.

### Tier 1, Phase 1 (data and perception skeleton), mostly done 2026-10-17
- Kaggle wrapper (`remote/kaggle.py`): every call is privacy-checked. Public flag refused, `is_private: true` required, kernel metadata read back after push, and every kernel and dataset requested anonymously after upload (must not return 200).
- Footage: IMMAF YouTube channel, metadata only. 74 adult fights, 125 fighters, 18 with 2+ fights (`data/fights.csv`, `data/fighters.csv`, local only). Adult-only filter reads titles and descriptions; skipped 11 videos directly and 5 more by the fighter rule (all junior or youth).
- Model choices logged (TransNetV2, RF-DETR Large, BoT-SORT, ViTPose++ Large); AGPL and non-commercial licences avoided.
- Real pipeline (`pipeline: real`): ingest, shots, tracking, identity, pose, plus overlays and label-free health metrics. CPU end-to-end test on a synthetic video with stand-in models.
- Kaggle job: one worker per GPU on the T4 x2 machine, per-stage timings, resumable by batch (import takes only finished fights).
- YouTube blocked Kaggle after 2 downloads, so batches now download on the laptop, upload one private dataset, delete local copies, and `import` deletes the dataset.
- Pilot: 2 fights processed (the other 3 of the first batch were blocked by YouTube). Health numbers per fight: both fighters boxed in 67% and 69% of live frames, 3.8 and 2.4 role switches per minute, median keypoint confidence 0.87 and 0.85.
- Speed pass (FP16, prefetching, bigger batches, two GPUs): 0.42 to 1.15 footage minutes per session minute; per fight 1.38x to 1.53x faster. Outputs match the FP32 run (99.4% boxes matched, median IoU 0.997, keypoints 0.24 px apart at the median). Details in DECISIONS.md.
- Docs: LEARNING.md sections 8 to 15. 114 tests, all passing.

## Shaky or worth a second look
- **Perception weak spots seen in the pilot overlays.** In clinches against the fence the detector often draws one box around both fighters (the main reason both fighters are boxed only about two thirds of the time). The "referee" group picks up officials, photographers and spectators outside the cage. Walk-outs and between-round footage count as live (live share 95 to 100%). All need Phase 2 numbers before fixing.
- **Identity is unstable on 2 to 3% of boxes**: the FP16 rerun changed the role of that share even though boxes barely moved. Fighter A and B are also not yet linked to names; IMMAF uses red and blue kits, which should make that easy.
- **Throughput is still too low for 1000 fights on free quota**: about 200 session hours at the current rate. Next targets by measured stage time: shot detection (4 to 5 minutes per fight, mostly decoding every frame) and decoding each video three times.
- **YouTube may block the laptop too** if batches get large. One transient download failure happened; downloads now retry once.
- The stage code hash doesn't see changes in helper functions. Each stage has a manual `version` string to bump. Hashing the whole module would be stricter.
- Plug-in empirical Bayes ignores uncertainty in alpha and pi. Fine with hundreds of fighters, too narrow with few. Full Bayes comes in Tier 1.
- The first `pre-commit` run on a new machine downloads Go to build gitleaks, which takes a minute or two.

## Next
1. Finish Phase 1: run the 3 pilot fights YouTube blocked (`cornerman kaggle batch --n 3 --overlays 1`), so the pilot covers 5 fights as the spec asks.
2. A second speed pass before scaling (decode once for tracking and pose; cheaper shot detection), measured against the numbers above.
3. Then 50 fights (hard rule 10), then Phase 2: the automated gold set.
4. MLflow comes later, before the first real model fit (Phase 4).

## Later (noted 2026-10-17)
- **More amateur sources before Tier 2.** Only 18 of 125 fighters have 2+ fights in the IMMAF data. Per-fighter models need depth, so Tier 2 (scaling to 1000+ fights) must add more public amateur sources with repeat fighters. Same rules apply: adult bouts only, no pro promotion broadcasts.
- **Fight results before Tier 3.** Result, method, round/time and weight class are empty because IMMAF's YouTube descriptions don't include them. Scrape them from IMMAF's website (check its terms first) before Tier 3, where the simulator and Elo baseline need outcomes.

## Blocked
- Nothing blocking. Pushing derived data off the laptop waits on a DVC remote, which Ansh has said can wait.
