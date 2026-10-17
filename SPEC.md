# Cornerman: build spec (v3)

Owner: Ansh. This file is the brief for an autonomous coding agent. Read all of it before writing code.

## What this is

A system that watches amateur MMA footage at scale, turns every fight into a sequence of (state, action) events for both fighters, and learns how each fighter actually fights: what they do in each situation, how that changes as they tire, how they react to the opponent, and how they compare to everyone else. On top of that it simulates fights between any two fighters and recommends how to beat a specific opponent.

End products:
1. Scouting report per fighter (tendencies, predictability, cardio, counter-striking, exploitable habits, all with uncertainty).
2. Style map: every fighter embedded in a learned style space, with "who fights like X."
3. Matchup simulator: Monte Carlo fights between any two fighters, with outcome probabilities and method of victory.
4. Live overlay demo: a fight video playing with the current state, next-action probabilities, and strike intensity drawn on top in near real time.
5. Between-round cornerman: feed it round 1 of a fight and it updates its beliefs about the opponent in real time (sequential Bayesian updating) and outputs round 2 adjustments.
6. A public web app and API: browse fighters, read scouting reports, explore the style map, run matchups, tweak counterfactuals ("what if he had better cardio").
7. A paper-style write-up with real evaluation, and a docs site.

The computer vision is plumbing. The resume value is the modeling: hierarchical Bayes, hidden semi-Markov models, Hawkes processes, learned style embeddings, stochastic-game simulation, and honest calibration. Get perception to "good enough and measured," then move on.

## Hard rules

1. Footage: public amateur MMA footage only. No pro promotion broadcasts.
2. Never commit or redistribute video. Raw video lives in gitignored `data/raw/` and is deleted after processing. Only derived data (tracks, poses, events) is kept.
3. Every model is evaluated against baselines. No result is reported without one.
4. Every perception component gets a measured accuracy number on the automated gold set (Phase 2). No "looks right" checks.
5. Before choosing any model (detector, tracker, pose, VLM), check what's current and pick the best open-weight option that fits the GPU. Pin versions. Log the choice and why in `DECISIONS.md`.
6. Plain READMEs and docs. No marketing language.
7. Budget: the agent runs on Ansh's Claude subscription, which is separate. GPU time and any hosted APIs are capped at $60 USD total. Prefer free compute. Never add a pay-as-you-go API key or anything that can auto-bill. Paid GPU only from prepaid credit Ansh already loaded. Log every dollar in `costs.md`.
8. Ansh does not label footage. All labeling, including evaluation data, is automated.
9. Token discipline: never pull video frames into your own context except a handful for debugging. All video processing happens in code on the GPU. Pipeline code never calls Claude or any paid model to label footage.
10. Scale only after a pilot: full pipeline on 5 fights, check numbers, then 50, then go big.
11. Every tier must end in a working, documented, resume-ready state before the next tier starts. If the project stopped after any tier, it should still look finished.
12. Ansh has to be able to defend every part of this in an interview. For each component, write a section in `LEARNING.md`: what problem it solves, why this approach over the alternatives, how the math or mechanism works step by step, what breaks it, and 3 questions an interviewer would ask with answers. Plain flowing explanation, no fluff. The modeling sections (Layers 4 and 5) get the most depth.
13. Hosting and services: free tiers only, no card on file anywhere that can auto-bill.
14. Engineering standards in the "Engineering platform" section apply from day one, not as a cleanup at the end.

## Environment and models

- The agent runs in Claude Code on Ansh's laptop. Build with Opus for planning, math, architecture, and hard debugging; use Sonnet for routine scripting. Fable is not used during the build. It comes in later for an upgrade pass (see the end of this file).
- Heavy GPU work does not run on the laptop:
  - First choice: Kaggle notebooks (free weekly GPU quota).
  - Second: Google Colab free tier.
  - Fallback: Vast.ai or RunPod on prepaid credit only. Shut instances down the moment a job finishes.
- Every GPU job is resumable (checkpoint per fight) because free sessions get killed.
- Process in batches: download, extract, save derived data, delete raw video.

## Taxonomy (locked v1, do not change without asking)

States (10), from each fighter's own perspective:
- standing-neutral
- standing-cage-pressuring
- standing-cage-pressured
- standing-opponent-hurt
- clinch-controlling
- clinch-controlled
- ground-top
- ground-bottom
- ground-scrambling
- momentum flag (modifier on top of any state, not a state itself)

Actions (8):
- strike
- shoot takedown
- clinch entry/exit
- pass guard / advance position
- submission attempt
- sweep / reversal attempt
- stand back up
- reset / circle out

Every moment produces one state per fighter, and the two must be compatible (ground-top for one means ground-bottom for the other).

## Architecture

### Layer 0: data
- Collect public amateur fight videos with yt-dlp. Metadata per fight in `data/fights.csv`: event, date, fighters, weight class, result, method, round/time.
- Fighter registry (`data/fighters.csv`): canonical fighter IDs, name variants, linked across events. Use video titles, descriptions, and public amateur record sites where their terms allow. Repeat fighters are the most valuable data in the project.
- Target: 1000+ fights after the pilot. Prioritize depth (many fights per fighter) over breadth.
- Dedupe, drop anything under 480p.

### Layer 1: perception
1. Shot boundary detection. Drop crowd shots, slow-mo, and graphics. Detect replays explicitly and match them to the live footage (used as a free consistency test, then excluded from counts).
2. Person detection + multi-object tracking.
3. Identity: fighter A, fighter B, referee. Use shorts/glove color clustering, position, posture, and appearance re-ID. Keep identity stable across cuts.
4. Pose per fighter: 2D first, 3D body mesh recovery where 2D breaks in grappling.
5. Arena geometry: segment cage/ring boundary and mat, estimate a per-shot homography to mat coordinates. Cage-pressure states come from measured distance to the fence.
6. Strike detection: limb velocity spikes toward the opponent. Landed vs missed from contact cues (limb endpoint reaching opponent's body region plus a head/torso displacement response).
7. Hurt detection: post-strike signatures like stumbles, buckled knees, sudden loss of posture, covering up. Feeds standing-opponent-hurt.

Known hard part: two bodies tangled in clinch and on the ground. Lean on hip height, combined silhouette shape, contact, and torso orientation rather than individual keypoints.

### Layer 2: event extraction
- State classifier: features from Layer 1 into one of the 10 states, smoothed over time.
- Action detector: skeleton-based temporal action recognition on pose sequences, plus rules where rules are reliable.
- Output per fight: `data/events/{fight_id}.parquet` with t, fighter, state, action, landed, confidence, source.

### Layer 3: weak supervision
Several noisy labelers, combined:
- A: rules over pose/geometry features.
- B: open-weight video VLM on the GPU, labeling 2 to 4 s clips against the taxonomy with a strict JSON schema.
- C: the trained skeleton model once it exists.
Aggregate with Dawid-Skene (EM), which estimates each labeler's confusion matrix without ground truth. Train final classifiers on the aggregated soft labels. Iterate: retrain C, re-aggregate.

### Layer 4: fighter models

4a. Hierarchical conditional action model.
n_{f,s} ~ Multinomial(theta_{f,s}), theta_{f,s} ~ Dirichlet(alpha_s * pi_s).
pi_s is the population policy in state s, alpha_s is how hard fighters shrink toward it. Fit by empirical Bayes (Dirichlet-multinomial marginal likelihood), then full Bayes in NumPyro, and compare. Posterior mean (n_{f,s,a} + alpha_s * pi_{s,a}) / (N_{f,s} + alpha_s).

4b. Noisy-state sequence model.
True state is latent; perception output is the emission, with the emission matrix initialized from measured confusion rates. Hidden semi-Markov model with explicit duration distributions per state. Compare against a plain HMM and a continuous-time Markov chain.

4c. Hawkes striking model.
Bivariate Hawkes per fight: lambda_i(t) = mu_i(t) + sum over j and past strikes t_k of fighter j: alpha_ij * exp(-beta * (t - t_k)).
mu_i = baseline aggression, alpha_ii = combination tendency, alpha_ij = counter-striking tendency. Fit by MLE, pool across fights hierarchically. Same math as order-flow models in market microstructure; say so in the README.

4d. Cardio and time dynamics.
Let tendencies drift over the fight: mu_i(t) = mu_i0 * exp(-gamma_i * t) gives a per-fighter cardio decay rate gamma_i. Let theta_{f,s} vary by round with a logistic-normal random walk. Report who fades, who doesn't, and how a fighter's game changes late.

4e. Style embeddings.
Learn a vector per fighter two ways and compare:
- Interpretable: non-negative / Poisson tensor factorization of the fighter x state x action count tensor. Components should read as styles (wrestler, pressure striker, counter striker, scrambler).
- Predictive: a small sequence model (GRU or tiny transformer) predicting the next action from recent history, with a learned fighter embedding. Add an opponent interaction term so P(action | state, fighter, opponent) depends on the opponent's embedding.
Outputs: 2D style map, nearest-neighbor "who fights like X," and matchup effects (does this fighter change their game against wrestlers?).

4f. Predictability.
Per fighter per state: conditional entropy H(A | S = s) and KL divergence from the population policy. Headline scouting numbers.

### Layer 5: game layer

5a. State values. Define a per-moment reward (positional advancement, control time, strikes landed, finishes). Estimate V(s) with fitted value iteration or TD on the empirical transition kernel.

5b. Best response. Given an opponent's posterior policy, compute which of your actions in each state has the highest expected value. Only recommend where the posterior is tight. Same idea as best response in the CFR poker project.

5c. Fight simulator. A semi-Markov stochastic game: sample state durations from 4b, each fighter's actions from 4a/4e, strikes from 4c with cardio from 4d, and a finish hazard driven by accumulated damage and dominant positions. Monte Carlo 10k fights per matchup: win probability, method (KO/TKO, submission, decision), round.
Validation:
- Posterior predictive checks: simulated vs real distributions of control time, takedown attempts, strike counts, finish rates.
- Held-out outcome prediction (log loss, Brier) vs an Elo baseline built from fighters' records.
Counterfactuals: expose simulator parameters (cardio decay, takedown rate, counter-striking alpha) so a user can ask "what if fighter A had fighter B's cardio" and see the outcome distribution shift.

5d. In-fight Bayesian updating (the actual cornerman).
Before a fight, the opponent's policy prior is their posterior from past fights (or the population prior for an unknown opponent, refined by their style-embedding neighbors). As events from the current fight stream in, update with the conjugate Dirichlet-multinomial step and update Hawkes and cardio parameters online. After each round, output: what changed in our beliefs, how confident we are, and the best-response adjustments for the next round. Evaluate by replaying held-out fights round by round: does round 1 updating improve round 2 next-action log loss over the pre-fight prior? Also test against unknown opponents (no past fights), which is the realistic amateur case.

### Layer 6: products
- Scouting reports (HTML per fighter).
- Style map (interactive).
- Matchup simulator page: pick two fighters, see the outcome distribution.
- Live overlay demo: render a held-out fight with state label, next-action probability bars, and Hawkes intensity curves drawn on the video. Short clips of this go in the README.
- `paper/`: method, evaluation, limitations, and where amateur data falls short.

## Engineering platform

Code quality
- Python packaged with `uv` and `pyproject.toml`, typed throughout, checked with pyright. `ruff` for lint/format. pre-commit hooks.
- Config with pydantic settings (or Hydra). Every run is reproducible from a config file plus a seed. No magic numbers buried in code.

Pipeline
- Every stage (download, shots, tracking, pose, geometry, events, labels, models) is a function from versioned inputs to versioned outputs, cached by content hash. Rerunning skips anything unchanged. Any stage can be rerun for one fight.
- A DAG runner ties the stages together (Prefect, Dagster, or a small custom one; pick and log the reason).
- Derived data versioned with DVC. Storage: Parquet files, queried through DuckDB as the analytics layer.
- Data contracts: schema and sanity checks (pandera) on every event table. Bad fights get quarantined with a reason, not silently dropped.

Testing
- Unit tests for everything with logic in it.
- Parameter recovery tests for every statistical model: simulate data from known parameters, fit, check the true values land inside the credible intervals at the right rate. Use simulation-based calibration for the Bayesian models. This is the most important test suite in the repo.
- Property-based tests (hypothesis) for math utilities, e.g. Hawkes log-likelihood matches a brute-force version, Dirichlet updates are order-invariant.
- Golden tests: a tiny fixture fight whose expected event log is checked in, so perception regressions get caught.
- CI on GitHub Actions (free for public repos): lint, types, tests on every push. GPU stages are tested on CPU with tiny fixtures.

Experiment tracking
- MLflow (local) or a free-tier tracker. Every model fit logs config, data version, metrics, and plots. The paper's numbers come from tracked runs only.

Performance
- Batched GPU inference, mixed precision, ONNX/TensorRT export where it helps. Benchmark throughput as minutes of footage processed per GPU-minute and report it in the README.

Serving
- FastAPI service: endpoints for fighters, scouting reports, style neighbors, matchup simulation, counterfactuals. Auto-generated OpenAPI docs.
- Web frontend over the API: fighter pages, style map, matchup simulator with counterfactual sliders, round-by-round cornerman replay, embedded overlay clips. Keep it thin, all logic lives in the Python API.
- Public deploy on free tiers (e.g. Hugging Face Spaces or similar). Serve precomputed data plus the simulator, which is cheap on CPU. Video inference stays offline.
- Dockerfile for the API so anyone can run it locally with one command.

Docs
- README: what it does, overlay GIFs, headline results with baselines, one-command quickstart.
- Docs site (mkdocs) built from `LEARNING.md`, the method write-ups, and the API docs.

## Roadmap

Tier 0, foundation (before any real modeling)
- Repo skeleton, uv, pyright, ruff, pre-commit, CI green.
- Pipeline DAG with content-hash caching and DVC, running on one fixture fight with stub stages.
- Parameter recovery test harness, proven on a toy Dirichlet-multinomial model.
- `LEARNING.md` and `DECISIONS.md` started.

Tier 1, core (resume-ready on its own)
- Phase 1: data + perception skeleton. 50 fights downloaded with metadata; tracking, identity, pose running end to end on 5 fights with overlay videos.
- Phase 2: automated gold set.
  - External labeled combat-sports datasets (e.g. grappling position datasets, boxing punch datasets), mapped onto the taxonomy where they overlap. Check licenses.
  - Silver consensus set: 300 to 500 stratified clips where the strongest VLM (two prompts) and the rules labeler all agree. Call it silver in the write-up.
  - Label-free checks: state compatibility violation rate, replay-pair consistency, impossible-transition rate.
  - Optional 100-clip spot-check set with a hotkey tool. Never wait for it.
- Phase 3: event extraction + weak supervision. Per-class precision/recall vs each labeler. Target 80%+ frame-level state accuracy; if stuck, document and continue.
- Phase 4: models 4a, 4b, 4f. Next-action prediction on a temporal holdout (train on each fighter's earlier fights, predict later ones). Log loss, Brier, reliability diagram. Baselines: global frequencies, per-state population frequencies, raw per-fighter frequencies. Report honestly if the hierarchical model doesn't win.
- Scouting report v1.

Tier 2, ambitious
- Scale to 1000+ fights.
- Strike landed/missed and hurt detection.
- 4c Hawkes, 4d cardio, 4e style embeddings + style map.
- Scouting report v2.

Tier 3, moonshot
- 5a values, 5b best response, 5c simulator with posterior predictive checks, outcome prediction vs Elo, and counterfactuals.
- 5d in-fight Bayesian cornerman with round-by-round replay evaluation.
- Live overlay demo.
- Full paper.

Tier 4, ship it
- FastAPI service, web frontend, public deploy, Dockerfile.
- Docs site.
- Throughput benchmarks in the README.
- Final pass on `LEARNING.md` so it covers every component.

## When to stop and ask Ansh
- Any taxonomy change.
- GPU/API spend reaching $40 (ask before going further), never past $60.
- Free GPU quota exhausted and the next step needs paid GPU.
- A result that contradicts an earlier one in a way you can't explain.
- End of each tier, with a short summary of what works, the numbers, and what's shaky.

Otherwise make the call, log it in `DECISIONS.md`, and keep going.

## Fable upgrade pass (later, not during the build)
When Ansh switches the agent to Fable, it does this, in order:
1. Audit every evaluation for leakage (same fight or same fighter-time in train and test, replays double counted, tuning on the test set).
2. Re-derive and check the math in 4a to 4e and 5c: likelihoods, priors, identifiability, whether parameters are actually recoverable on simulated data with known ground truth.
3. Review the worst perception failure cases and fix the biggest error sources.
4. Simplify and refactor code that grew messy.
5. Rewrite the paper and README for clarity and honesty.
Log findings in `AUDIT.md`.

## Repo layout
```
cornerman/
  src/cornerman/
    pipeline/      DAG, stages, caching
    perception/    detection, tracking, identity, pose, geometry, strikes, hurt
    events/        state + action extraction
    weak_labels/   labelers + Dawid-Skene
    models/        hierarchical, hsmm, hawkes, cardio, embeddings, values, simulator, cornerman
    eval/
    api/           FastAPI service
    contracts/     data schemas and checks
  web/             frontend
  kaggle/          GPU job notebooks/scripts
  tests/           unit, recovery, property, golden
  labeling_tool/
  products/        reports, overlay rendering
  docs/            mkdocs site
  paper/
  data/            raw/ gitignored, rest tracked by DVC
  .github/workflows/
  Dockerfile
  pyproject.toml
  LEARNING.md
  DECISIONS.md
  AUDIT.md
  costs.md
  README.md
```
