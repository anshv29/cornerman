# Progress

## Needs Ansh

1. **A free DVC remote (not blocking).** Ansh said on 2026-10-17 this can wait. Recommended: DagsHub (free, no card). Sign up at https://dagshub.com with GitHub, connect the `anshv29/cornerman` repo, then share the DVC remote URL and an access token with the agent, for example by putting the token in a gitignored `.env` file. Until then the DVC cache lives only on this laptop.

## Kaggle setup (done 2026-10-17)
- Account `anshvaishnav29`, phone verified, so notebooks get GPU and internet.
- Auth is an API token in `~/.kaggle/access_token` (set up 2026-10-18; `kaggle config view` shows `auth_method: ACCESS_TOKEN`). It takes precedence over the earlier OAuth login, whose 12-hour token lapsed for 30 minutes at each renewal (DECISIONS.md). There is no `kaggle.json`.
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
- Pilot: all 5 fights processed with the current code (crops and everyone-on-canvas poses for all 5 after a rerun of the 2 oldest on 2026-10-18). Health numbers per fight: both fighters boxed in 67% and 69% of live frames, 3.8 and 2.4 role switches per minute, median keypoint confidence 0.87 and 0.85.
- Speed pass (FP16, prefetching, bigger batches, two GPUs): 0.42 to 1.15 footage minutes per session minute; per fight 1.38x to 1.53x faster. Outputs match the FP32 run (99.4% boxes matched, median IoU 0.997, keypoints 0.24 px apart at the median). Details in DECISIONS.md.
- Second speed pass: each video decoded once into a frame cache next to the raw video; shot detection at about 25 fps (tracking and pose were already at 10 fps). Measured on Kaggle: 1.65 to 1.37 seconds of wall time per second of video, about 17% faster even with pose covering everyone on the canvas.
- Cage mask (people must stand on the canvas; per-shot canvas colour) and a motion-based fighting filter (walk-outs, breaks and announcements out; 9.3 and 9.5 minutes of fighting on the pilot fights).
- Identity redesign (Ansh approved): pose on everyone on the canvas, colour samples at gloves, shins and shorts, identity after pose without a GPU, red and blue corners linked to registry IDs (IMMAF titles list red first, checked on the score bar). Person crops about once a second in `data/crops/` (local only) so identity can be redone without the GPU.
- Docs: LEARNING.md sections 8 to 18 (section 12 rewritten for the new identity).

## Shaky or worth a second look
- **Top perception issue for Phase 3: clinch box-merging.** In clinches, against the fence and on the ground, the detector often draws one box around both fighters, so both fighters are boxed in only 67 to 69% of live frames on the pilot. Everything downstream (identity, pose, state) inherits it. First thing to try: the RF-DETR segmentation variant (Apache 2.0, same family as our detector), whose per-person masks separate tangled bodies better than boxes. Measure it against the Phase 2 gold set, split by state (standing, clinch, ground).
- **Identity is not solved.** Pipeline version (keypoint colour): right on 1 of 3 fights checked. Offline per-fight embedding (`experiments/identity_embedding.py`): people are separated in 4 of 5 fights, but picking the fighter pair automatically works on 2 of 5. See "Next".
- **Cage mask misses**: a person outside the cage but close to the camera (a cornerman in a cageside shot) overlaps the canvas in the image and is kept; in one fight, cageside cameras shooting through the fence let several officials through. A white or grey canvas (other promotions) falls back to plain colour distance, untested.
- **Fighting filter** is tuned on 2 fights (threshold 0.2 box heights per second gives 9.3 and 9.5 minutes). It should be checked on the 50 against each fight's round count, once results are scraped.
- **Throughput:** about 1.37 s of wall time per second of video per GPU; two GPUs run in parallel. Ansh's call: enough for the 74 IMMAF fights; no more speed work for now.
- **YouTube may block the laptop too** if batches get large. One transient download failure happened; downloads now retry once.
- The stage code hash doesn't see changes in helper functions. Each stage has a manual `version` string to bump. Hashing the whole module would be stricter.
- Plug-in empirical Bayes ignores uncertainty in alpha and pi. Fine with hundreds of fighters, too narrow with few. Full Bayes comes in Tier 1.
- The first `pre-commit` run on a new machine downloads Go to build gitleaks, which takes a minute or two.

## Next (waiting for Ansh)
Offline identity work (2026-10-18) did not reach the target of red and blue right on 4 of 5 pilot fights: best automatic result 2 of 5. The per-fight appearance embedding separates people well (4 of 5 fights); choosing which groups are the two fighters is what fails. Details and next ideas in DECISIONS.md, "Offline identity experiments". Stopping here as agreed, before the 50-fight run. Options for Ansh:
1. Keep going on identity: a fighter-versus-official classifier on body-masked DINOv2 features across fights (laptop only, no GPU), then recheck all 5.
2. Run the 50 fights now. Tracking, pose, colour samples and crops don't depend on identity, so identity can be redone offline later without GPU time; only the fighting-time filter and the health numbers depend on it.
3. MLflow comes later, before the first real model fit (Phase 4).

## Later (noted 2026-10-17)
- **More amateur sources before Tier 2.** Only 18 of 125 fighters have 2+ fights in the IMMAF data. Per-fighter models need depth, so Tier 2 (scaling to 1000+ fights) must add more public amateur sources with repeat fighters. Same rules apply: adult bouts only, no pro promotion broadcasts.
- **Fight results before Tier 3.** Result, method, round/time and weight class are empty because IMMAF's YouTube descriptions don't include them. Scrape them from IMMAF's website (check its terms first) before Tier 3, where the simulator and Elo baseline need outcomes.

## Blocked
- Pushing derived data off the laptop waits on a DVC remote, which Ansh has said can wait.
