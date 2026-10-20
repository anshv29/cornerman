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
Free Kaggle GPU used so far: about 23 hours in total. On 2026-10-19 and 10-20: about 3.5 h for the last 24 fights, about 1.7 h of VLM pilots, 0.6 h for a failed full run (out of memory), 3.5 h for the full VLM run.

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
- 50 fights processed (2026-10-18), 45 of them in 3 batches of 15 on free quota.
- Identity experiments in `experiments/` (offline, laptop only): best is 20 of 30 held-out fights right.
- Docs: LEARNING.md sections 8 to 19.

### Identity round 3 and the start of Phase 2 (2026-10-19)
- All 74 IMMAF fights processed (the last 24 on free Kaggle quota, about 3.5 h).
- Identity round 3, developed on the 30 already-seen fights: roles per crop (Viterbi along each tracklet), two-pass fighting time, and a per-fight confidence score from four label-free checks. Dropped after trying: fighting-time-only grouping with ground fill-in (worse), and a behavioural referee detector (its labels were 35 to 50% precise). Dev: 21 of 30.
- Held-out test on the 24 new fights, method fixed and committed first: 18 of 24 right (75%), no red/blue swaps. Flag at 0.4: unflagged 15 of 19 right, 2 of 6 wrong fights caught. Confidence AUC 0.83. Verdicts in `experiments/identity_v3_test_verdicts.csv`.
- The method is in the pipeline: a `crop_features` stage (DINOv2-small over the body mask) and the `identity` stage writing `roles.parquet` (role per box), `quality.json` (confidence, flag) and `fighting.parquet`. Matches the frozen experiment exactly on all 33,492 test crops. `cornerman identity-quality` lists flags; `usable_fights` gives the rest.
- Ansh's calls (2026-10-19): flag threshold 0.5 (chosen after the test; re-validate on the next fresh fights). On the shipped classifier it flags 35 of 74 fights, 39 usable (the 0.5 estimate had mixed two classifier versions; corrected in DECISIONS.md). Flagged fights only for population-level numbers (`population_fights`), never per fighter. ViCoS yes, for evaluation only.
- Phase 2 started: plan and licence checks (DECISIONS.md), label-free checks (`eval/label_free.py`).
- Fixes: Kaggle downloads retry when the connection drops; import no longer re-copies an older batch or wipes stages run on the laptop.
- Docs: LEARNING.md section 20.

### Phase 2, silver set first build (2026-10-19 to 10-20)
- Clips (`gold-clips`, `gold-cut`): 400 clips of 3 s from the 39 usable fights, 100 per stratum (apart, close, ground, merged), clips with both fighters boxed first. Cut on the laptop from a re-download checked against the processed video, red corner in a red box and blue in a blue box, 10 fps, 480 px. Local only; on Kaggle only as a private dataset deleted after each job.
- ViCoS jiu-jitsu (`gold/vicos.py`): annotations only (keypoints and hand-labelled positions, pinned by hash, under `data/external/`, never committed), mapped to our ground states after checking from the keypoints which athlete each label names.
- Labeller A, rules (`gold/rules.py`, `gold-rules`, `gold-vicos`): coarse labels. On ViCoS, with thresholds fixed first: standing versus ground 97%; who is on top 82% right, 2% wrong, 16% unsure.
- Labeller B, VLM (`gold/vlm.py`, `gold/vlm_run.py`, `kaggle/vlm`, `gold-vlm push|import`): Qwen3-VL-4B chosen over 8B after three pilots (8B ran out of memory, was 2.4 times slower and no more consistent). Two prompts; impossible or undecidable answers dropped.
- Silver combiner (`gold/silver.py`, `gold-silver`). Full run with both prompts: 129 silver clips of 400, almost all standing-neutral (115) plus 14 ground; none for clinch or cage pressure, because the questions prompt named the wrong fighter on the fence (checked on the footage). Ansh's rulings (2026-10-20): a clinch against the fence is a clinch; cage pressure only at striking range with a back near the fence; standing over a downed opponent is ground-top. Prompts and rules changed to match, so the old answers are void.
- Oversampled sample: 1,000 clips (apart 100, ground 100, close 400, merged 400), the first 400 unchanged. The VLM job now writes answers as they come, works through strata evenly and stops starting clips after 11 h, so running out of the weekly quota loses nothing.
- Docs: LEARNING.md section 21, DECISIONS.md silver-set entry.

## Shaky or worth a second look
- **Top perception issue for Phase 3: clinch box-merging.** In clinches, against the fence and on the ground, the detector often draws one box around both fighters, so both fighters are boxed in only 67 to 69% of live frames on the pilot. Everything downstream (identity, pose, state) inherits it. First thing to try: the RF-DETR segmentation variant (Apache 2.0, same family as our detector), whose per-person masks separate tangled bodies better than boxes. Measure it against the Phase 2 gold set, split by state (standing, clinch, ground).
- **Identity is good enough to start, not solved.** 75% of held-out fights right; the failures are ground mixing and the other fighter leaking into a corner through clinches. The confidence score ranks fights well but its threshold is soft (Needs Ansh, item 1). Clinch box-merging (above) is the shared root cause.
- **Clip strata are a rough proxy.** "ground" leans on torso angle and wide boxes and "merged" also catches close-ups of one fighter. They only spread the sample.
- **The silver set leans easy.** The rules need both fighters boxed separately, so tangled clinches and pins rarely become silver. Numbers measured on it must say "silver" and be split by stratum.
- **The VLM's clinch-control answers are unreliable.** In the pilot the two prompts often named opposite fighters as controlling. Silver drops those clips, so clinch-controlling/controlled may stay thin.
- **Kaggle CLI quirks:** `kaggle datasets create` creates the dataset but sometimes fails to parse the reply ("Expecting value"); rerunning the push works because it then adds a version. Once, a kernel push reported success but Kaggle kept the old version, so the "complete" status belonged to the previous run. After a push, check that the status is QUEUED or RUNNING.
- **Cage mask misses**: a person outside the cage but close to the camera (a cornerman in a cageside shot) overlaps the canvas in the image and is kept; in one fight, cageside cameras shooting through the fence let several officials through. A white or grey canvas (other promotions) falls back to plain colour distance, untested.
- **Fighting filter** is tuned on 2 fights (threshold 0.2 box heights per second gives 9.3 and 9.5 minutes). It should be checked on the 50 against each fight's round count, once results are scraped.
- **Throughput:** about 1.37 s of wall time per second of video per GPU; two GPUs run in parallel. Ansh's call: enough for the 74 IMMAF fights; no more speed work for now.
- **YouTube may block the laptop too** if batches get large. One transient download failure happened; downloads now retry once.
- The stage code hash doesn't see changes in helper functions. Each stage has a manual `version` string to bump. Hashing the whole module would be stricter.
- Plug-in empirical Bayes ignores uncertainty in alpha and pi. Fine with hundreds of fighters, too narrow with few. Full Bayes comes in Tier 1.
- The first `pre-commit` run on a new machine downloads Go to build gitleaks, which takes a minute or two.

## Next
1. The VLM run on all 1,000 clips with both prompts (pushed 2026-10-20; about 8.7 h, which may not fit this week's remaining free quota; anything left is finished after the weekly reset, with `--per-stratum` or by excluding clips already labelled). Then `gold-silver`, and a by-eye spot check of silver clips per state.
2. Report silver coverage and spot-check precision per state, not only overall (clinch and ground are the thin ones).
3. Phase 3 can then start: per-state accuracy of the rules labeller against silver, and Dawid-Skene over labellers A and B.
4. MLflow comes later, before the first real model fit (Phase 4).

## Later (noted 2026-10-20)
- **Identity confidence per segment, not per fight.** Flagging whole fights drops half the data (35 of 74). Most flagged fights are wrong only in parts (a ground scramble, a post-fight shot), so a confidence per segment (per shot, or per minute of fighting time) would let the clean parts of flagged fights be used. The four checks can be computed on any span of frames; the threshold would need its own check.

## Later (noted 2026-10-17)
- **More amateur sources before Tier 2.** Only 18 of 125 fighters have 2+ fights in the IMMAF data. Per-fighter models need depth, so Tier 2 (scaling to 1000+ fights) must add more public amateur sources with repeat fighters. Same rules apply: adult bouts only, no pro promotion broadcasts.
- **Fight results before Tier 3.** Result, method, round/time and weight class are empty because IMMAF's YouTube descriptions don't include them. Scrape them from IMMAF's website (check its terms first) before Tier 3, where the simulator and Elo baseline need outcomes.

## Blocked
- Pushing derived data off the laptop waits on a DVC remote, which Ansh has said can wait.
- Using the ViCoS jiu-jitsu set waits on Ansh's licence call.
