# Progress

## Needs Ansh

1. **Kaggle login expired (blocks GPU runs).** The OAuth session from `kaggle auth login` ran out on 2026-10-17. Run `uvx --from kaggle==2.2.4 kaggle auth login` on the laptop (in Claude Code: `! uvx --from kaggle==2.2.4 kaggle auth login`) and finish the browser step. Then the 3 blocked pilot fights can run: `uv run cornerman kaggle batch --fight ft_a8bf4f59c3 --fight ft_b8991792ed --fight ft_c816f513c5 --overlays 1`.
2. **A free DVC remote (not blocking).** Ansh said on 2026-10-17 this can wait. Recommended: DagsHub (free, no card). Sign up at https://dagshub.com with GitHub, connect the `anshv29/cornerman` repo, then share the DVC remote URL and an access token with the agent, for example by putting the token in a gitignored `.env` file. Until then the DVC cache lives only on this laptop.

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
- Second speed pass (laptop-tested, not yet run on Kaggle): each video decoded once into a frame cache next to the raw video; shot detection at about 25 fps. Tracking and pose were already at 10 fps.
- Cage mask (people must stand on the canvas; per-shot canvas colour) and a motion-based fighting filter (walk-outs, breaks and announcements out; 9.3 and 9.5 minutes of fighting on the pilot fights). Pose runs only on fighting frames.
- Docs: LEARNING.md sections 8 to 18.

## Shaky or worth a second look
- **Top perception issue for Phase 3: clinch box-merging.** In clinches, against the fence and on the ground, the detector often draws one box around both fighters, so both fighters are boxed in only 67 to 69% of live frames on the pilot. Everything downstream (identity, pose, state) inherits it. First thing to try: the RF-DETR segmentation variant (Apache 2.0, same family as our detector), whose per-person masks separate tangled bodies better than boxes. Measure it against the Phase 2 gold set, split by state (standing, clinch, ground).
- **Identity is wrong on the pilot fight, and it decides whether the 50-fight run is useful** (see DECISIONS.md, "Identity is the blocker"). Colour clustering can't separate IMMAF's mostly black kits (red or blue trim) from each other or from the referee's black uniform. Three cheap feature and naming changes didn't fix it and were reverted. Needs Ansh's call, below.
- **Cage mask misses**: a person outside the cage but close to the camera (a cornerman in a cageside shot) overlaps the canvas in the image and is kept. A white or grey canvas (other promotions) falls back to plain colour distance, untested.
- **Fighting filter** is tuned on 2 fights (threshold 0.2 box heights per second gives 9.3 and 9.5 minutes). It should be checked on the 50 against each fight's round count, once results are scraped.
- **Throughput:** 1.15 footage minutes per Kaggle session minute before the decode-once change, which should save roughly 400 to 500 s per fight but is not yet measured on Kaggle. Ansh's call: enough for the 74 IMMAF fights; no more speed work for now.
- **YouTube may block the laptop too** if batches get large. One transient download failure happened; downloads now retry once.
- The stage code hash doesn't see changes in helper functions. Each stage has a manual `version` string to bump. Hashing the whole module would be stricter.
- Plug-in empirical Bayes ignores uncertainty in alpha and pi. Fine with hundreds of fighters, too narrow with few. Full Bayes comes in Tier 1.
- The first `pre-commit` run on a new machine downloads Go to build gitleaks, which takes a minute or two.

## Decision for Ansh: identity before the 50-fight run
Pose runs only on boxes labelled as fighters, so running 50 fights with the current identity would compute poses for the wrong people, and they'd need redoing. Proposed fix, cheap and testable on the laptop:
1. Run pose on every person on the canvas (fighters, referee, the odd cornerman), not just labelled fighters. With the cage mask that's about 2.5 people per frame instead of 2, so pose takes roughly a quarter longer.
2. Store a small colour sample at the keypoints where corner colours live: wrists (gloves), ankles and shins (guards), hips (shorts).
3. Assign identity afterwards, on the laptop, from those samples: the referee has no gloves, and red and blue gloves and guards separate the corners. This also links fighter A/B to the red/blue corner, so to the names.
Identity can then be redone offline as often as needed, without GPU time. The alternative is to run the 50 now with the current identity and accept a pose rerun later.

## Next
1. Kaggle login (Needs Ansh 1), then run the 3 blocked pilot fights. They'll use the new code (decode once, cage mask, fighting filter), which also measures the decode-once saving on Kaggle.
2. Identity, per Ansh's decision above.
3. Then 50 fights (hard rule 10), then Phase 2: the automated gold set.
4. MLflow comes later, before the first real model fit (Phase 4).

## Later (noted 2026-10-17)
- **More amateur sources before Tier 2.** Only 18 of 125 fighters have 2+ fights in the IMMAF data. Per-fighter models need depth, so Tier 2 (scaling to 1000+ fights) must add more public amateur sources with repeat fighters. Same rules apply: adult bouts only, no pro promotion broadcasts.
- **Fight results before Tier 3.** Result, method, round/time and weight class are empty because IMMAF's YouTube descriptions don't include them. Scrape them from IMMAF's website (check its terms first) before Tier 3, where the simulator and Elo baseline need outcomes.

## Blocked
- GPU runs: Kaggle login expired (Needs Ansh 1).
- The 50-fight run: waits on the identity decision above.
- Pushing derived data off the laptop waits on a DVC remote, which Ansh has said can wait.
