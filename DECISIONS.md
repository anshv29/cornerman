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

## 2026-10-17: YouTube blocks Kaggle after a few downloads; videos go through a private dataset
The 5-fight pilot downloaded and processed its first two fights, then the next three failed in under 2 seconds each with YouTube's "Sign in to confirm you're not a bot". So the smoke test's conclusion (download inside Kaggle) held for two or three videos, not for batches. We don't pass YouTube cookies: that would mean putting Ansh's logged-in session on a Kaggle machine.
The fallback is now the default (`cornerman kaggle batch --source dataset`): the laptop downloads the batch's videos into gitignored `data/raw/upload/`, uploads them as one private Kaggle dataset (`cornerman-videos`), and deletes the local copies right after the upload. The kernel attaches the dataset, `ensure_video` links each video from `/kaggle/input` instead of downloading, and `cornerman kaggle import` deletes the dataset once the results are home. Raw video exists only while a batch is being processed.
Privacy for datasets is checked the same way as for kernels: the public flag is refused, and after upload we request the dataset with no credentials and fail if Kaggle answers 200 (a public dataset returns 200; a private or missing one returns 403 or 404). Kernel pushes now get the same anonymous check on top of reading their metadata back.
`--source youtube` still exists for small batches, and the three blocked pilot fights are left for the next batch.

## 2026-10-17: Speed pass: FP16, prefetching, bigger batches, one fight per GPU
**Why.** The first pilot ran at 2080 s for an 826 s fight and 1901 s for an 836 s fight, 2.3 to 2.5 times slower than real time, on one of the machine's two T4s. That's about 0.42 minutes of footage per minute of Kaggle session, so 1000 fights (about 14,000 minutes of footage) would need roughly 550 session hours, far beyond the free weekly quota.
**What changed.**
- RF-DETR runs through `model.inference(compile=False, dtype=torch.float16)`, as its own log suggests. `compile=False` because batch sizes vary (the last batch of a fight is short).
- ViTPose++ runs in FP16; heatmap peak finding stays in FP32 so FP16 only touches the network.
- Detector batches go from 8 to 16 frames and pose batches from 16 to 32. Frames are decoded on a background thread, 64 frames ahead, while the GPU works.
- Shot detection gets its 48x27 frames from a multi-threaded ffmpeg scale instead of decoding and resizing in Python.
- The job runs one worker process per GPU, each pinned with `CUDA_VISIBLE_DEVICES` before torch loads, and deals fights out round-robin. Two fights at once is simpler than splitting one fight's frames across cards, and nothing is shared between them.
- `perception.half` is part of the tracking and pose cache keys, so FP16 and FP32 outputs never get mixed in the cache.
**Result** (same two fights, same T4 machine):

| | FP32 pilot | Speed pass |
|---|---|---|
| Fight 1 (826 s of video) | 2080.6 s | 1363.1 s (1.53x faster) |
| Fight 2 (836 s of video) | 1901.3 s | 1379.3 s (1.38x faster) |
| Both fights, wall time | 3982 s, one after the other | 1448 s including install (2.75x faster) |
| Footage minutes per session minute | 0.42 | 1.15 |

Per fight, time is now 1.65 times the video length. Stage times for the speed pass: shots 210 to 301 s, tracking 546 to 598 s, pose 511 to 567 s, ingest and identity under 5 s. The pilot didn't log per-stage times, so we can't split the gain by stage.
**Outputs still match** (`cornerman compare`, FP32 pilot vs FP16, both fights): same number of shots (36 and 19), 99.4 to 99.5% of boxes matched at IoU at least 0.5 with median IoU 0.997, 99.6% of matched boxes in the same track grouping, 97.3 to 98.1% with the same role, keypoints 0.24 px apart at the median and 2.8 to 3.2 px at the 95th percentile (on 1280x720 frames), with 92 to 93% within 2 px. The health metrics are unchanged to two decimals. The 2 to 3% role disagreement says identity is unstable on borderline tracklets: tiny box changes flip them between groups. That's a weakness of the identity step, not of FP16, and is noted for Phase 2.
FP16 is now the default in `configs/kaggle_real.yaml`, and its outputs replace the FP32 pilot outputs in `data/derived/` (the FP32 ones are kept locally in `data/derived_fp32/` for reference).
**Still too slow for 1000 fights on free quota:** about 200 session hours at 1.15 footage minutes per minute, around 7 weeks of Kaggle's weekly allowance. The next targets, by measured stage time: shots (4 to 5 minutes, mostly decoding every frame of a 50 fps video for a model that needs 48x27 images), and decoding each video three times (shots, tracking, pose). Tracking and pose could share one decode pass, and shot detection could use a lower frame rate or hardware decoding.

## 2026-10-17: Second speed pass: decode once, smaller shot input (tracking and pose were already at 10 fps)
Ansh asked for three cheap changes. What we found and did:
- **Tracking and pose already ran at 10 fps** (every 5th frame of a 50 fps video, every 3rd of 30 fps). Only shot detection used every frame. No change needed.
- **Decode once.** H.264 can't skip frames cheaply: reading every 5th frame still decodes all of them. Measured on the pilot fight (826 s, 720p50) with software decoding, which is what Kaggle has: 164 s using every laptop core, 268 s on 2 threads. That matches the shots stage's 210 to 300 s on Kaggle, so each of the three stages was spending most of its time decoding. (OpenCV on the Mac took 20 s because it used the Mac's hardware decoder, which isn't representative.) Now one ffmpeg pass writes both the shot-detection frames and the 10 fps frames, as high-quality JPEGs, into the fight's raw folder (`perception/decode.py`). That pass took 166 s on the laptop, barely more than decoding alone, since JPEG encoding runs in parallel. Tracking and pose read the JPEGs. The cache is about 1.6 GB per fight and is deleted with the raw video, so it never reaches `data/derived` or DVC. Expected saving on Kaggle: two of the three decodes, roughly 400 to 500 s per fight. Not yet measured on Kaggle.
- **Downsampled shot detection.** TransNetV2 now sees about 25 fps (every 2nd frame of a 50 fps video) instead of every frame. That halves its work but doesn't change decoding, which was the real cost. Shot boundaries are mapped back to full-rate frame indices.
- ffmpeg comes from `imageio-ffmpeg` (a pip package with a static binary) when the system has none, so the laptop, CI and Kaggle all run the same code.
We stop here on speed, as Ansh asked: 74 IMMAF fights fit in about a week of free quota, and scale is limited by sources, not speed.

## 2026-10-17: Cage mask: keep only people standing on the canvas
Officials, photographers and spectators outside the cage passed the box-size filter and ended up in the "referee" group. The mask (`perception/arena.py`) keeps a person only if their feet are on the canvas.
- **Canvas colour.** The most common bright colour in a band across the lower middle of the frame (rows 45 to 75%). Two earlier versions failed on the pilot. A median of the bottom-centre picked up the dark broadcast score bar. A mode over all pixels picked up the dark arena. Hence "bright" (lightness above 110 of 255) and the higher band.
- **Per shot.** The main camera and the cageside camera saw the same canvas about 33 apart in chroma (white balance). With one fight-wide colour, 42% of frames found no canvas; learning the colour per shot (from up to 12 of its frames, trusted only within 45 chroma of the fight-wide colour) brought that to 3%.
- **Matching.** On a coloured canvas, chroma only (Lab a and b within 18), with a brightness floor of 30% of the canvas lightness, because exposure varies between cameras far more than hue does. A white or grey canvas falls back to full Lab distance. Holes are closed, large regions in the lower two thirds are kept, and the result is a convex hull.
- **Inside test.** The bottom-centre of the box within 3% of frame height of the hull. For boxes reaching into the bottom 15% of the frame, the feet are hidden (tight shots, score bar), so any point across the box's width at that line counts.
- **Result on the pilot fight:** 57% of all person boxes kept, 87% of the boxes the old run had labelled as fighters (that label included some outsiders). Checked on frames: the walk-out outside the cage door, officials and spectators outside the fence, and a person outside labelled `fighter_a` are dropped; fighters, the referee and cornermen inside are kept. Known miss: a person outside the cage but close to the camera, overlapping the canvas in the image (a cornerman in a cageside shot), is kept.

## 2026-10-17: Only live fighting counts: a motion filter for walk-outs, breaks and announcements
"Live shot" (two big people on screen) let in walk-outs, the minute between rounds and the result announcement. The pilot's tracks showed a clean motion signal: the median fighter speed was 0.3 to 0.6 box heights per second in rounds and about 0.1 in breaks. `perception/activity.py` marks a frame as fighting when the 10 s centred rolling median of fighter speed is at least 0.2, in a live shot. Non-fighting gaps under 8 s are then filled (a stoppage, or a clinch where the detector merges both fighters into one box) and fighting runs under 15 s dropped. Walk-outs outside the cage are already removed by the cage mask.
Label-free check: three 3-minute rounds should give a bit over 9 minutes. On the two pilot fights this gives 9.3 and 9.5 minutes, with segments matching the round structure (fight 1: about 0 to 211 s, 281 to 472 s and 540 to 754 s). A threshold of 0.15 let breaks in (10.5 and 10.8 minutes); 0.25 cut into rounds (7.9 and 7.2). The identity stage writes `fighting.parquet`, pose runs only on fighting frames, and the health metrics should be read over fighting frames.

## 2026-10-17: Identity is the blocker for a useful 50-fight run (finding, not yet fixed)
With the cage mask applied offline to the pilot fight, we looked at identity labels on six frames spread over the fight. They are wrong, and not only in the mask's version: the pilot's own labels (colour clustering, fighters = the two groups that stay closest) mix up the two fighters and the referee too. IMMAF kits are mostly black with red or blue trim, and the referee wears black. The colour features are hue-saturation histograms over all pixels, and hue on black pixels is noise; the canvas, a saturated light blue, also fills much of each box.
Tried offline, on one fight, judged on frames: (1) hue from saturated pixels only, plus dark and grey shares, with a shin band; (2) also dropping canvas-coloured pixels and the box edges; (3) naming the referee as the group with the least screen time instead of by proximity. None separated the two fighters: almost every tracklet came out "more blue than red", including the red-corner fighter's. All three were reverted, since none is proven better than the pilot version.
This matters for the run: pose only runs on boxes labelled as fighters, so wrong identity means poses for the wrong people and a rerun later. The fix we'd propose is in PROGRESS.md and waits for Ansh's call.

## 2026-10-18: Identity from keypoint colour, after pose (Ansh approved the proposal)
- **Pose on everyone on the canvas** in live shots, not just boxes labelled as fighters, so pose no longer depends on identity being right. With the cage mask that is about 2 people per frame; we estimated roughly a quarter more pose work than the pilot.
- **Colour at the parts where corner colour lives** (`perception/kitcolour.py`): small patches (4% of box height) at each wrist (gloves), halfway between knee and ankle (shin guards) and just below the hips (shorts), with canvas-coloured pixels left out, stored as median Lab per part.
- **Identity after pose, without a GPU** (`perception/identity.py`): per tracklet the median a and b of each part; cannot-link k-means into up to four groups (red, blue, referee, anyone else on the canvas); groups closer than 20 chroma units that never share a frame are merged back, because with only three people k-means otherwise splits one of them in two (it did, in the synthetic test). The reddest gloves and shins are the red corner, the bluest the blue corner, and of the rest the group with the most screen time is the referee.
- **Corners to names:** IMMAF titles list the red corner first. Checked on the two pilot fights against the broadcast score bar, which marks the blue name with a blue edge on the left and the red name with a red edge on the right; in both, the right-hand (red) name is the title's first name. `corner_fighters` encodes this; other sources need their own check.
- **Person crops**, about one per tracked person per second, 160 px high, saved to `data/crops/{fight_id}/` (on Kaggle, the kernel's private output), so identity can be redone with a different method later without the GPU or the video. They show fighters, so they stay local: `data/` is gitignored by allow-list and only `data/derived` goes into DVC.
- Stage order is now ingest, shots, tracking, pose, identity. Live shots moved into tracking; the fighting filter moved into identity, since it needs to know who the fighters are.

## 2026-10-18: Why the Kaggle OAuth login lapses, and the fix
The CLI (kaggle 2.2.4) stores a refresh token and gets a new 12-hour access token automatically, but its expiry check is `expiration < now - 30 minutes`, so it renews only half an hour after the old token has expired. During that half hour every call fails with "Authentication required", which is what stopped the batch on 2026-10-17. An API token in `~/.kaggle/access_token` takes precedence over OAuth in the CLI's lookup order (access token, then legacy `kaggle.json`, then OAuth), so it avoids the problem. Generating it is Ansh's step (PROGRESS.md, Needs Ansh). `access_token` is gitignored and gitleaks scans every commit.

## 2026-10-18: Results of the 3-fight run with the new code (decode once, cage mask, keypoint identity)
The 3 pilot fights YouTube had blocked (446, 735 and 1037 s of video) ran on Kaggle's T4 x2 in 2395 s of wall time, all three finished.
**Decode once, measured.** The fights differ from the earlier ones and pose now does more work (everyone on the canvas instead of labelled fighters), so the fair comparison is time per second of video:

| | Previous run (FP16, decoded 3 times) | This run (decoded once) |
|---|---|---|
| Wall time per second of video | 1.65 (both fights) | 1.58, 1.22, 1.38 (1.37 overall) |
| shots stage (now includes the one decode) | 0.25 to 0.36 | 0.17 to 0.32 |
| tracking | 0.66 to 0.72 | 0.45 to 0.64 |
| pose | 0.62 to 0.68 (fighters only) | 0.51 to 0.61 (everyone on the canvas) |

So per second of video the whole pipeline is about 17% faster, despite pose covering more people. For a typical 14-minute fight that's roughly 230 s saved, about half the 400 to 500 s we estimated; the rest of the expected saving went into the extra pose work. We can't split the two exactly without rerunning the same fights both ways, which isn't worth the quota.
**Identity, checked on crops (no labels yet).**
- One fight (full red kit against full blue kit): red, blue and referee groups are each mostly the right person; checked on a montage of 12 crops per group.
- Two fights (2023 Worlds: black kits with small red or blue trim, dark-red gloves): identity is wrong. The keypoint patches land in the right places (checked by drawing them on crops) but read little colour. In one fight the gloves are dark red (saturation around 108, brightness around 76 on OpenCV's 0 to 255 scale), below our "kit colour" floor; in the other both fighters' gloves and shin guards look black.
- An offline test on the crops with a pose-masked body region (torso and thighs) and counts of red and blue pixels separates the colourful fight cleanly but not the two black-kit fights, even with wider colour windows.
- Two other problems show up in the crops. Merged clinch boxes become long tracklets that contain both fighters. In one fight, cageside cameras shoot through the fence, so officials outside it overlap the canvas in the image and are kept by the cage mask.
Nothing here needs GPU to try next: the crops and poses are on the laptop. Ideas, in order: exclude merged boxes (wide boxes, or two people's keypoints in one box) from identity; a lower saturation floor for dark red with a hue window that avoids skin; and for fights where the corners are only distinguishable by small trim, a learned appearance embedding trained on the confident tracklets of the same fight.
**Import was slow:** Kaggle's output download fetches files one at a time, and 3,717 crops took over ten minutes. The job now zips each fight's crops into one file.

## 2026-10-18: Offline identity experiments: target not met (2 of 5 fights), stopping to check with Ansh
Goal (Ansh, time-boxed to one session): red and blue right on at least 4 of the 5 pilot fights, judged on crop montages. Everything ran on the laptop from the crops and poses; the only GPU use was a short free Kaggle run to give the two oldest pilot fights crops and everyone-on-canvas poses.
**Judging rule.** For each fight, a montage of 12 random crops per role. "Right" means at least 9 of 12 crops in both the red and the blue row show that corner's fighter as the main person, and red and blue aren't swapped.
**What was tried.**
1. *Drop merged boxes* (width-to-height ratio above 1.6 times the fight's median): about 20% of crops per fight. Kept in every later variant; it removes clinch boxes that contain both fighters.
2. *Dark-red colour window* (hue 155 to 180 or 0 to 5, saturation from 70): helps a little, not enough on its own. Black kits with dark trim stay ambiguous at tracklet level.
3. *Per-fight learned appearance embedding (the main bet)* (`experiments/identity_embedding.py`): DINOv2-small (Apache 2.0, revision `ed25f3a31f01`), patch features averaged over the pose body mask so background and canvas don't dominate; a linear projection to 64 dimensions trained per fight with a triplet loss, where crops of one tracklet are the same person and crops of tracklets on screen together are different people; cannot-link clustering into 8 groups; look-alike groups that never co-occur merged. A first version on whole-crop features clustered by camera and background (the triplet loss hit 0.001 trivially); masking to the body fixed that.
**Result.** The embedding separates people well: in 4 of 5 fights each large group is one person (one fight splits the blue fighter over two groups). What fails is the next step, deciding which groups are the two fighters. Referees and cageside officials stand close to fighters in the image, and a fighter's tracklets are spread over several groups.

| Fight | Groups are single people | Fighters = most close frames | Fighters = smallest median distance | If the right pair were chosen |
|---|---|---|---|---|
| ft_c816f513c5 (full red and blue kits) | yes | right | wrong | right |
| ft_b8991792ed (black kits, dark-red gloves) | yes | right | wrong | right |
| ft_9eec213a59 | yes | wrong (cageside official) | wrong | right |
| ft_a8bf4f59c3 (similar black kits) | mostly | wrong | right | right |
| ft_97394d8b1a | partly (blue fighter split) | wrong | wrong | partly |

Best automatic result: 2 of 5. With the fighter pair chosen correctly, red against blue by pooled colour was right in 4 of 5.
**Rules tried for picking the fighters, and why they failed:** most frames close together (favours whoever has most screen time, so the referee next to one fighter wins); smallest median distance (a coin flip between near-equal pairs: 0.71 against 0.72 in one fight); share of poses with both hands up (too noisy: officials gesture, fighters drop their guard). Bare skin above the knee was tried and dropped: the skin colour range under-detected darker skin, which makes it unfair as well as unreliable.
**What we'd try next, in order:**
1. A fighter-versus-everyone-else classifier across fights: a linear probe on the same body-masked DINOv2 features, trained on groups from fights where the pair is unambiguous, and checked on the others. Fight gear (gloves, shin guards, bare feet, shorts) is what separates fighters from officials, and it's the same across fights.
2. Use merged clinch boxes as evidence: they contain both fighters, so the groups whose crops look most like the merged crops are the fighters.
3. Only then revisit the tracker's ID switches that split one fighter across groups.
Nothing here needs new GPU time. The keypoint-colour identity in the pipeline stays as it is until one of these works.

## 2026-10-18: The 50-fight run
45 more fights ran in 3 batches of 15 (picked by `cornerman kaggle batch --n 15`, which favours fighters with the most fights), one after another, each well inside Kaggle's 12-hour session limit: 9,504 s, 9,061 s and 7,884 s of wall time for 11,210 s, 11,118 s and 10,708 s of video, on the T4 x2 machine (about 1.2 to 1.4 seconds of video per second of session). No fight failed or was quarantined. Each batch staged its videos on the laptop (about 2.8 GB, 4 minutes to download), uploaded one private dataset, and the dataset was deleted after import. About 7.3 hours of free GPU quota; roughly 11 hours used this week in total, so the weekly quota was never in danger. 50 fights are now in `data/derived/` with tracks, poses, colour samples and crops.

## 2026-10-18: Fighter-versus-official classifier for identity: 20 of 30 held-out fights right (target not met)
**Method** (`experiments/identity_fighters.py`, laptop only):
- *Labels without people:* a crop is "fighter" if its torso is close to horizontal (on the canvas) or one ankle is above the other knee (a kick); "not fighter" if its tracklet barely moves for at least 30 frames (standing officials, photographers). Checked on samples: about 75 to 80% of the fighter labels and about 95% of the not-fighter labels are right. A knee-skin cue was rejected earlier because it under-detected darker skin.
- *Classifier:* logistic regression on DINOv2-small features averaged over the pose body mask.
- *Per fight:* the per-fight embedding (triplet loss, now with a body hue histogram added) groups crops by person; fighter-like groups (mean probability at least 0.5) that are almost never on screen together are folded into one person; the two fighter-like people that do appear together and have the most screen time are the fighters; tracklets the classifier calls non-fighters (below 0.35) are taken out of the fighter groups; red and blue by pooled red and blue pixel share.
**Evaluation.** Development on the 5 pilot fights (with the classifier trained on batch 1); after that, nothing was changed. Test: train on 20 fights (pilot plus batch 1), assign the 30 fights of batches 2 and 3, which were not looked at before, and judge each on a montage of 12 random crops per corner. Right = at least 9 of 12 in both rows show that corner's fighter, and red and blue not swapped. Judged by the agent, not blind.
**Result:**

| Variant | Right |
|---|---|
| Classifier only | 20 of 30 (67%) |
| Classifier plus merged-clinch evidence (weight 0.5, fixed before the test) | 9 of 30 (30%) |

The merged-box evidence made things worse: the referee is in many clinch boxes (breaking fighters up) and dresses like the dark part of the merged crops, so he was picked as a fighter in 18 of the 19 fights it changed. It is not used.
In none of the 30 fights were red and blue swapped: the colour decision between the two chosen fighters held even for red-against-gold and blue-against-black-and-gold kits. All failures are mixing:
- ground scrambles where both fighters end up in one row (4 fights);
- officials, coaches and post-fight flag shots in a fighter row (4);
- a referee in a red shirt (1);
- red against gold, borderline at 8 of 12 (1).
**What we'd try next:** use only crops from fighting time (the motion filter), which drops walk-outs, flag shots and officials between rounds; leave ground positions and merged-looking boxes out of identity and assign them afterwards from the standing frames around them; and a "who is the referee" step (the person who separates fighters, separately trainable from the same behavioural labels) instead of relying on clothing.
Until then, identity is the experiment's output, kept locally; the pipeline's identity stage is unchanged.

## 2026-10-18: Identity round 3 on the dev fights, and the method fixed before the new test
Ansh asked for one last round, laptop only: fighting-time crops only, ground and merged boxes left out and filled in from nearby standing frames, a behavioural referee detector, and a per-fight confidence score. The 30 fights judged last round are no longer a clean test, so they became the dev set; the remaining 24 IMMAF fights (processed on Kaggle while this was built) are the new test. Code: `experiments/identity_v3.py`. Dev verdicts: `experiments/identity_v3_dev_verdicts.csv`.
**Judging rule** (same as last round, one change): 12 random crops per corner; right = at least 9 of 12 in both rows show that corner's fighter and red and blue aren't swapped. The change: crops come from fighting time only, since that's what later stages use. Judged by the agent, not blind; borderline 8-or-9 calls make the noise about two fights either way.
**Dev results (classifier trained on the same 20 fights as last round, assigned the 30):**

| Variant | Right of 30 |
|---|---|
| Last round's method (judged on all crops) | 20 |
| a: fighting-time grouping, standing separate crops only, fill-in, tracklet roles | 16 |
| b: roles per crop (Viterbi along each tracklet), two-pass fighting time | 21 |
| c: b plus fighting-time-only grouping | 21 |
| b with the fighter classifier voting on each crop (judged where it changed labels) | 19 |

- **Fighting time, role-free.** The stored `fighting.parquet` rests on the old identity (0 minutes on two fights). Everyone's motion gives a first estimate; it lets in post-fight scenes (flags, team jackets). A rule "at most 3 people on the canvas" cut some fights by more than half (cageside officials always in view) and was dropped. Two passes (identity, then the pipeline's filter on the chosen fighters' motion, then identity again) is what b uses.
- **Standing-only with fill-in (variant a) made things worse.** In exchanges many standing crops overlap the other fighter's box, so too much was left to the fill-in step, and continuity across the gap put one fighter in the other's row.
- **Referee from behaviour didn't work on this footage.** "Someone moves in between two people who were in contact, and they then come apart", checked on montages of 4 and then 8 fights: 35 to 50% of the labels were referees; the rest were coaches outside the fence close to the camera, post-fight crowds and fighters. Requiring similar size and foot line removed most labels without making them clean. Not used; the referee stays "of the other people, the one most often on screen with the fighters".
- **What helped: roles per crop.** The tracker swaps IDs when fighters tangle, so one tracklet holds both fighters and any per-tracklet label is partly wrong. b keeps last round's choice of the two fighters, then scores each crop against red, blue and the nearest other person in the per-fight embedding, with a Viterbi pass along each tracklet that pays 2 (in units of 0.1 cosine) to change person. One fight went from a red row of mostly the blue fighter to clean.
- **b and c tie** at 21; c fixes officials and flag shots in 4 fights and breaks 2 ground-heavy ones. b was chosen because the confidence score works better on it (below).
**Confidence score.** Four label-free checks, each scaled to 0..1 with ranges set by reasoning (not fitted), averaged: one corner on two boxes in the same frame (0 good, 10% bad), the fighter classifier's score of the less fighter-like chosen person (0.6 to 0.85), share of fighting-time crops given a corner (0.3 to 0.7), and how often red and blue are the closest pair when a third person is in view (0.2 to 0.6). Other checks (pair choice margin, colour margin, crop colour purity, per-crop decisiveness) separated right from wrong no better than chance on dev. On b's 30 dev fights the score's AUC is 0.74. The flag threshold, 0.4, is the one number picked on dev: it flags 6 fights (5 wrong, 1 right), and 20 of the 24 kept are right (83%). It catches officials and flag shots, not ground mixing. 0.4 sits in a narrow gap (0.373 to 0.397), so expect it to move on new fights.
**Fixed for the test, before any of the 24 test fights is looked at:** `FROZEN` and `FLAG_BELOW = 0.4` in `experiments/identity_v3.py` as of this commit. The fighter classifier is retrained on all 50 dev fights (its labels are automatic, no verdicts involved). Report: right of 24, and right among the fights not flagged.

## 2026-10-19: Identity round 3, held-out test on the last 24 IMMAF fights
The 24 fights nobody had looked at were processed on Kaggle (free quota, about 3.4 h) and run once through the frozen method (variant b, `FLAG_BELOW = 0.4`, fighter classifier trained on the 50 dev fights). Judged with the dev rule (12 random fighting-time crops per corner, at least 9 of 12 in both rows, no swap) before reading the flag column. Verdicts in `experiments/identity_v3_test_verdicts.csv`.
- **Right: 18 of 24 (75%).** Dev was 21 of 30 (70%). No fight had red and blue swapped. Three of the 18 were borderline right (9 of 12 in a row), one of the 6 borderline wrong (8 of 12).
- **The 6 wrong:** ground mixing (2), one fighter in both rows (1), the referee and both fighters mixed (1), the other fighter leaking into a row through clinches (2). Officials and flag shots, the dev failure the flag was tuned on, caused none of them.
- **Split by the flag at 0.4:** 5 flagged, 19 not. Not flagged: 15 of 19 right (79%). The flag caught 2 of the 6 wrong fights. Of the 5 flagged, 3 were right.
- **Confidence as a ranking works; the threshold doesn't.** AUC 0.83 on test (0.74 on dev, 0.77 pooled over 54). All 6 wrong fights score between 0.32 and 0.49; the 13 fights at 0.5 or above are all right. On the 54 fights pooled, a threshold of 0.5 keeps 36 fights of which 32 are right (89%) and catches 11 of 15 wrong; 0.4 keeps 43 with 35 right (81%). 0.5 is picked after seeing the test, so 89% is an optimistic estimate, not a test result.
**Answer to "can flagged fights just be excluded?"** Not at 0.4: excluding them barely raises accuracy (75% to 79%) and leaves 4 wrong fights in. At 0.5 exclusion is worth it (about 9 in 10 kept fights right, two thirds of fights kept), at the cost of a threshold that has not had its own clean test. The pipeline keeps 0.4 as frozen until Ansh decides; `identity_flag_below` in the config is the one number to change, and `quality.json` stores the raw confidence so any later threshold can be applied without rerunning.
