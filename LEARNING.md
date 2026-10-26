# Learning notes

One section per component: what problem it solves, why this approach, how it works step by step, what breaks it, and questions an interviewer would ask, with answers. Sections are added as each component is built. Tier 0 covers the foundation: the taxonomy, the pipeline, data contracts, DVC, the first statistical model, and how we test that statistical models are honest.

---

## 1. The taxonomy and state compatibility

**Problem.** Every later model counts things like "how often does fighter F shoot a takedown when standing neutral". Those counts only mean something if every fight is described in the same fixed vocabulary. So we lock a small set of states (where the fight is, from one fighter's point of view) and actions (what that fighter chose to do).

**Why this approach.** States are written from each fighter's own perspective (ground-top vs ground-bottom), not from a neutral camera view ("ground, A on top"). That way a fighter's policy is a function of *their* situation, and the same model works for both corners without special cases. The cost is that the two fighters' states are tied together: if one is ground-top the other must be ground-bottom.

**How it works.** `taxonomy.py` defines the 9 states and 8 actions as string enums, plus a table of which pairs of states can happen at the same moment. Momentum is a separate true/false flag and can sit on top of any state. The table is symmetric (if A can be in s while B is in s', then the reverse is also allowed), and a unit test checks that. The compatibility table gets used three ways: the event contract rejects impossible pairs, the stub generator only samples legal pairs, and later the perception output gets a "compatibility violation rate" as a label-free quality check.

**What breaks it.** Real moments that don't fit: an even clinch with nobody controlling, a fighter hurt while pinned on the ground, both fighters scrambling while one is still technically on top. In those cases the classifier is forced to pick the nearest state, and that choice adds noise. If the violation rate stays high in Tier 1, that's evidence the taxonomy needs a change, which needs Ansh's sign-off.

**Interview questions.**
1. *Why per-fighter states instead of one joint state?* Because the thing we model is each fighter's decision given their situation. A per-fighter state makes "fighter F's policy in state s" well defined and lets both fighters share one model. The joint state can always be rebuilt from the pair.
2. *How do you know the two per-fighter labels agree?* There is an explicit compatibility table, and the data contract rejects any table where both fighters are in an impossible pair at the same timestamp.
3. *Why so few states?* Every extra state splits the data further. With amateur fighters having maybe 3 to 10 fights each, a fighter might have only a handful of actions in a rare state. A small taxonomy keeps counts large enough to say anything.

---

## 2. The pipeline runner and content-hash caching

**Problem.** Processing a fight goes through many stages (download, shots, tracking, pose, geometry, events, labels, models). Some are expensive GPU jobs on free sessions that get killed. We need to rerun only what changed, resume after a crash, rerun one stage for one fight, and set bad fights aside without stopping the batch.

**Why this approach.** We wrote a small runner (`src/cornerman/pipeline/runner.py`) instead of using Prefect or Dagster. Those tools bring a server and their own state store, and their caching is per task run, not per fight. Make-style timestamp caching breaks when files are copied between Kaggle and the laptop, because copying changes timestamps. Content hashes don't have that problem.

**How it works.**
1. Each `Stage` has a name, a function, the stages it depends on, the config fields it reads, a version string, and optionally external source files.
2. Stages are put in dependency order with a depth-first topological sort, which also catches cycles and unknown dependencies.
3. For each fight and stage, the runner builds a cache key: the SHA-256 of a canonical JSON blob holding the stage name, version, a hash of the function's source code, the values of only the config fields that stage declares, the global seed, the fight ID, hashes of external source files, and the *output hashes* of its upstream stages.
4. If the stage's `manifest.json` has the same key, and re-hashing the files on disk matches the output hash saved in the manifest, the stage is skipped.
5. Otherwise the stage runs into a `.tmp` folder. Only when it finishes are the output hash and manifest written and the folder renamed into place. A killed session leaves at most a `.tmp` folder, which gets wiped on the next run, and never a half-written stage that looks finished.
6. If a stage raises `QuarantineError`, the runner writes `data/quarantine/{fight}.json` with the reason, marks that fight's later stages as blocked, and moves on to the next fight.

Two details matter. Using upstream *output* hashes instead of upstream *keys* means that if a stage reruns but produces identical bytes, its downstream stages stay cached. Hashing only the declared config fields means changing, say, the pose keypoint count doesn't rerun shot detection.

Each stage also gets its own seed, derived by hashing (global seed, fight, stage name), so stages don't share a random stream and adding a new stage doesn't change the random numbers other stages see.

**What breaks it.** The code hash only sees the stage function's own source. If a helper it calls changes, the cache won't notice, which is why each stage also has a manual `version` string to bump. Non-deterministic output (GPU kernels, thread timing) means a rerun gives different bytes, so downstream stages rerun even when nothing meaningful changed. That's wasted work, but never wrong results. A stage that reads a config field without declaring it will be stale after that field changes. Code review is the guard for that.

**Interview questions.**
1. *Why hash content instead of checking timestamps?* Timestamps change on copy, checkout and sync, and they don't change if you edit a file and then revert it. A content hash changes exactly when the bytes change.
2. *What happens if a Kaggle session dies mid-stage?* The stage was writing into a temporary folder that is only renamed into place on success. Next run, the old manifest is either missing or doesn't match, so the stage runs again from scratch. Finished stages and fights are untouched.
3. *What's a case where your cache gives a wrong answer?* A stage that calls a helper function whose code changed, without the stage's version being bumped. The fix is the version string, and longer term we could hash the whole module the stage lives in.

---

## 3. Data contracts and quarantine

**Problem.** Perception will sometimes produce garbage: wrong fighter count, impossible state pairs, probabilities outside [0, 1]. If garbage gets into the event tables silently, every model downstream learns from it and we'd never know why.

**Why this approach.** pandera lets us write the contract for a pandas table in one place (`contracts/events.py`) and check it every time a table is written. We use lazy validation, so every failure is reported at once, not just the first. A failed fight isn't dropped. It gets *quarantined* with a reason, so we can count failures by cause and fix the biggest one.

**How it works.** Column checks: allowed states and actions come straight from the taxonomy enums, confidence is in [0, 1], fighter IDs must look like `F0001` (which also blocks real names from getting into the event tables), and the source must be one of the known labelers. Table checks: exactly one fight and two fighters, rows sorted by time, `landed` set only on strikes, and no moment where the two fighters' states are incompatible. That last check works by splitting the table by fighter, joining on timestamp, and testing each pair against the compatibility table.

**What breaks it.** Contracts catch things that are *impossible*, not things that are *wrong but plausible*. A fight with every state labeled standing-neutral passes. Those errors are caught by accuracy numbers on the gold set and by label-free checks in Tier 1, not by the contract.

**Interview questions.**
1. *Why quarantine instead of dropping bad rows?* Dropping rows quietly biases the data. If clinch moments fail more often, dropping them makes everyone look like a striker. Quarantining the whole fight with a reason keeps the dataset honest and gives a to-do list.
2. *Why validate at write time and not read time?* So bad data never lands on disk looking like good data, and the stage that produced it is the one that gets blamed.
3. *What can't the contract catch?* Plausible but wrong labels. Measuring those needs ground truth or proxies for it, like replay consistency.

---

## 4. Data versioning with DVC

**Problem.** Derived data (tracks, poses, events) is too big for git but has to be versioned, so that any number in the paper can be traced back to the exact data it came from.

**Why this approach.** DVC stores a small pointer file in git (`data/derived.dvc`, holding a hash of the folder) and keeps the actual bytes in a cache, and later in a remote. Checking out an old commit and running `dvc checkout` restores the matching data. We use `dvc add` rather than DVC's own pipeline feature because our runner already does finer-grained, per-fight caching. DVC only needs to snapshot the result.

**How it works.** `dvc add data/derived` hashes every file, copies them into `.dvc/cache` (content-addressed by MD5), and writes the pointer file. `data/.gitignore` stops git from tracking the bytes themselves. The fixture fight lives in plain git instead, because it is tiny and CI needs it without a DVC remote.

**What breaks it.** With no remote set up, the cache only exists on one laptop. That's fine for Tier 0, where everything can be rebuilt in seconds, but not once real fights are processed. Forgetting to run `dvc add` after a pipeline run means the pointer in git describes old data.

**Interview questions.**
1. *Why not just put the Parquet files in git?* Git keeps every version of every file forever, and clones would grow without limit. DVC keeps only pointers in git.
2. *How does a result in the paper link back to its data?* The run logs the git commit, and that commit's `.dvc` pointer names the exact data hash.
3. *Why not use DVC pipelines for the caching too?* DVC caches whole stages across all fights. We need per-fight granularity and per-fight quarantine.

---

## 5. The hierarchical Dirichlet-multinomial model (toy version of 4a)

**Problem.** We want each fighter's action probabilities in each state. Raw frequencies are useless for fighters with few observations: one takedown in one attempt would read as "always shoots". We need estimates that lean on the population when data is thin and on the fighter when data is rich.

**Why this approach.** The Dirichlet is conjugate to the multinomial, so the posterior is closed form, fast, exact, and has an interpretation you can explain in one sentence. Alternatives: a multinomial logistic model with random effects (more flexible, but needs MCMC and is harder to read), or raw frequencies plus additive smoothing (which is a special case of this model with a hand-picked prior). Empirical Bayes picks the prior from the data instead of by hand.

**How it works.** For one state with K actions:
- Each fighter f has a true action distribution theta_f, drawn from Dirichlet(alpha * pi). pi is the population average policy, and alpha measures how tightly fighters cluster around it. A large alpha means fighters are all alike. A small alpha means they vary a lot.
- Counts n_f ~ Multinomial(N_f, theta_f).
- Posterior: theta_f | n_f ~ Dirichlet(alpha * pi + n_f). You just add the counts to the prior parameters.
- Posterior mean: (n_f + alpha * pi) / (N_f + alpha). Writing w = alpha / (N_f + alpha), this equals w * pi + (1 - w) * n_f / N_f. So alpha acts like a number of "pretend observations" from the population. A fighter with N_f much larger than alpha is described mostly by their own data.
- Credible intervals: each component of a Dirichlet is Beta(a_k, a_0 - a_k), so intervals come from Beta quantiles.
- Empirical Bayes: integrate theta out to get the Dirichlet-multinomial marginal likelihood
  p(n_f | alpha, pi) = [N_f! / prod_k n_fk!] * [Gamma(alpha) / Gamma(N_f + alpha)] * prod_k [Gamma(n_fk + alpha pi_k) / Gamma(alpha pi_k)],
  and maximize the sum of its log over fighters. We optimize log(alpha), so it stays positive, and softmax logits for pi with the last one fixed at 0, so pi stays on the simplex, using L-BFGS-B. log(alpha) is bounded to [-5, 12], because when fighters truly don't differ the likelihood keeps rising as alpha goes to infinity.

**What breaks it.** Plug-in empirical Bayes treats the estimated alpha and pi as known, so intervals are a bit too narrow when there are few fighters. That's why the full Bayes version in NumPyro comes in Tier 1, for comparison. The model assumes a fighter's actions are independent draws given the state, but real actions come in streaks (combinations), which the Hawkes model in Tier 2 addresses. A single pi and alpha per state can't express "wrestlers vary a lot in how often they shoot, strikers don't". Style embeddings handle that later.

**Interview questions.**
1. *What does alpha mean in plain words?* It is how many observations' worth of trust we give the population average. A fighter with 5 actions in a state, when alpha is 20, is described mostly by the population. One with 500 actions is described mostly by themselves.
2. *Why is empirical Bayes OK here and when isn't it?* With hundreds of fighters, alpha and pi are pinned down well, so ignoring their uncertainty costs little. Our recovery test shows 90.9% coverage for 90% intervals with 200 fighters. With 10 fighters it would understate uncertainty, and full Bayes is the fix.
3. *What does alpha going to its upper bound tell you?* That the data shows no extra fighter-to-fighter variation beyond multinomial noise, so everyone gets shrunk all the way to the population. The test `test_empirical_bayes_finds_no_spread_when_there_is_none` checks exactly this.

---

## 6. Parameter recovery and simulation-based calibration

**Problem.** A statistical model can run without errors and still be wrong: a bug in the likelihood, a prior with the wrong scale, intervals that are too narrow. On real data we never know the true parameters, so we can't check. On simulated data we can.

**Why this approach.** Two complementary checks in `src/cornerman/eval/recovery.py`, written once and reused for every model in the repo:
- *Coverage.* Simulate data from known parameters, fit, and see whether the 90% intervals contain the truth about 90% of the time. This is the direct, easy-to-explain check.
- *Simulation-based calibration (SBC).* Draw parameters from the prior, simulate data, fit, then find the rank of the true value among posterior draws. If the posterior is correct, that rank is uniformly distributed. It checks the whole posterior shape, not just one interval, and the *shape* of the rank histogram tells you what is wrong.

**How it works.**
- Coverage: each replicate returns (truth, lower, upper) for a set of quantities. The hit count over n checks should be Binomial(n, 0.9) if intervals are honest, so we run a two-sided binomial test. Quantities must be independent, or the test is too confident. In the toy model, the K components of one fighter's theta are correlated because they sum to 1, so each fighter contributes only one component (component f mod K).
- SBC: with L = 99 posterior draws, the rank is in 0..99, so there are 100 equally likely values. These are put into 20 bins of 5, and a chi-square test checks they are flat. Each quantity is tested separately, with a Bonferroni correction across quantities.
- Why the SBC rank is uniform: if theta* ~ prior and data ~ p(data | theta*), then (theta*, data) has the same joint distribution as (theta, data) where theta is drawn from the posterior given data. So theta* is just another posterior draw, and among L + 1 exchangeable draws its rank is uniform.
- A test that always passes proves nothing, so we also feed the harness a deliberately broken posterior: the same mean with double the concentration, which halves the variance. Results with fixed seeds:

| check | correct posterior | overconfident posterior |
|---|---|---|
| 90% interval coverage (2000 checks) | 0.908, binomial p = 0.25 | 0.768, p = 2e-65 |
| SBC uniformity p-values (4 components, 1000 sims) | 0.42, 0.15, 0.27, 0.78 | all below 1e-40 |

  Empirical Bayes on 2000 simulated fighters recovers alpha = 2.01, 6.22 and 26.4 for true values 2, 6 and 25, and pi to within 0.015.

**What breaks it.** Recovery only proves the code fits the model it assumes. If real fights don't follow the model, recovery still passes. That's what posterior predictive checks on real data are for (Tier 3). Coverage checked at one parameter value can pass while failing elsewhere, which is why SBC draws parameters from across the prior and why the EB test sweeps alpha. Fixed seeds make the tests stable but mean each one checks one sample. The thresholds (p >= 0.01, Bonferroni) trade false alarms against power.

**Interview questions.**
1. *Why should SBC ranks be uniform?* If the truth is drawn from the prior and the data from the truth, then the truth is distributed exactly like a posterior draw given that data. Among L posterior draws plus the truth, all L + 1 are exchangeable, so the truth's rank is equally likely to be anything from 0 to L.
2. *What would a U-shaped rank histogram mean?* The truth keeps landing outside the bulk of the posterior draws, so the posterior is too narrow (overconfident). A hump in the middle means too wide. A slope means biased.
3. *How do you know your test can actually catch a bug?* We run it on a deliberately overconfident posterior and assert that it fails. It does, with coverage of 0.77 against 0.90 and SBC p-values below 1e-40.

---

## 7. Property-based tests

**Problem.** Example-based tests check the cases you thought of. Math bugs tend to hide in the ones you didn't think of: zero counts, tiny concentrations, unusual orderings.

**Why this approach.** Hypothesis generates hundreds of random inputs that satisfy a stated property and, when one fails, shrinks it to the smallest failing example.

**How it works.** Properties checked so far: (1) Dirichlet updates are order-invariant: adding observations one by one in any random permutation equals one batch update. This is what makes the in-fight cornerman's online updating valid later. (2) The posterior mean equals w * pi + (1 - w) * raw frequency exactly. (3) The Dirichlet-multinomial pmf sums to 1 over every count vector with a fixed total, which catches mistakes in the Gamma-function terms. (4) Config hashes don't depend on dict insertion order, so cache keys are stable.

**What breaks it.** A property that is too weak passes buggy code. Generated inputs need sensible ranges. Extremely small concentrations, for example, hit floating point limits that are not real bugs.

**Interview questions.**
1. *Give an example of a property test that would catch a real bug.* The pmf-sums-to-one test catches a missing multinomial coefficient or a sign error in a log-Gamma term, which example tests with one hand-computed value often miss.
2. *Why does order invariance matter for this project?* The cornerman updates beliefs as events stream in during a fight. If the update depended on order, beliefs would depend on how events were batched, not on what happened.
3. *What does shrinking do?* When a property fails, hypothesis repeatedly simplifies the failing input while it keeps failing, so you debug the smallest case instead of a random big one.

---

## 8. Footage collection, the adult-only filter, and the fighter registry

**Problem.** Every model later needs many fights per fighter, and every fight needs to be tied to the right two people. That starts with a list of fights, a list of fighters with stable IDs, and confidence that nothing in the data shows a minor or a pro promotion broadcast.

**Why this approach.** The first source is the official YouTube channel of IMMAF, the international amateur MMA federation. It posts complete single bouts from its championships with a consistent title format ("A vs. B | FULL FIGHT | 2023 ... Championships"), the footage is a fixed broadcast camera at 720p or 1080p, and the same fighters come back year after year. Parsing a fixed title format with one regular expression is far more reliable than trying to understand free text. We fetch metadata only (title, description, resolution, length). Video is downloaded later, inside the GPU job, and only for fights that passed every filter.

**How it works.**
1. `cornerman collect` lists the channel's uploads and keeps titles containing "FULL FIGHT".
2. For each one, yt-dlp fetches the metadata JSON, cached under `data/meta/` so reruns don't hit YouTube.
3. The title is parsed into fighter A, fighter B, the event, the year and a rematch number ("... 2").
4. The adult filter reads the title *and* the description. Bouts can sit under an adult event name and only say "Junior World Champion" in the description, so titles alone aren't enough. It flags youth, junior and cadet, age groups up to U21, words for children, and any stated age under 18. Any fighter who appears in a flagged video is then removed from the whole dataset, because we can't know when they turned 18. On the first run this dropped 11 videos directly and 5 more through that fighter rule.

   Titles and descriptions turned out not to be enough. IMMAF runs junior divisions inside its main championships, so "2025 IMMAF World Championships" can be a junior bout, and only the score bar says so ("JUNIORS / MEN / ..."). The filter therefore also reads YouTube tags and playlist names, and the on-screen graphics: 6 single frames are pulled from the stream at points spread over the video, the bottom quarter (where the score bar sits) is read by an OCR model (RapidOCR, small ONNX models on CPU), and the text goes through the same word check. Several frames are needed because any one frame can land on a replay, a break or a close-up without the bar. This caught 10 junior bouts that had passed the first version. Older events' graphics show no division at all, so for those the screen can't decide. Those fights go on hold (left out of everything, not deleted) and are checked against IMMAF's published results: a bout is adult if both fighters sit in the same adult bracket that year and neither appears in a junior bracket around then; junior if they are together in a junior bracket; otherwise unclear, and unclear bouts are dropped. Of 26 held fights, 18 were confirmed adult. The check was validated first on bouts whose division the screen did show, and it never called a junior bout adult.
5. Videos under 480p are dropped, then duplicates (same video, or same pairing at the same event and meeting).
6. Names are matched through a key that strips accents, case and punctuation, so "Paweł" and "Pawel" are one person. Each person gets an anonymous ID (`F0001`), kept stable across reruns by reading the previous registry. Fight IDs are a hash of the source and video ID, so neither ID reveals who or which video.

**What breaks it.** Graphics that never show the division (older events) leave the screen check blind, and OCR can misread. Two different people with the same name merge into one. One person whose name is spelled two ways that differ by more than accents (a transliteration, a dropped middle name) splits into two. Both hurt per-fighter models. The adult filter can miss a minor if nothing in the title or description says so, which is why it's deliberately aggressive (it skips "junior" even where junior means 18 to 20). And a single source is a single camera style, so perception tuned on it may not transfer to other footage.

**Interview questions.**
1. *Why not identify fighters by face?* Faces are small, often hidden by the guard, and we'd be building a face recognition system on people who never agreed to it. Names from the official title are more reliable and need no biometric data.
2. *How do you keep real names out of the public repo?* Everything with names lives in `data/`, which git ignores by default with an allow-list, and a test pins that. Every table that leaves the laptop uses `F0001`-style IDs, and the event contract rejects anything that doesn't look like one.
3. *Why drop a fighter's adult bouts because of one junior bout?* We can't tell a fighter's age from a title, and the cost of including a minor is much higher than the cost of losing a few fights. It's a precision-over-recall choice made on purpose.

---

## 9. Kaggle GPU jobs: privacy, downloading, and resuming

**Problem.** Perception needs a GPU, which means Kaggle's free T4s. Those sessions get killed, have a 12 hour limit, and anything we upload there could be made public by mistake.

**Why this approach.** Every Kaggle call goes through `cornerman.remote.kaggle`, which refuses the public flag, requires `"is_private": true` in a kernel's metadata before pushing, and after a push downloads the metadata back from Kaggle and checks it again. Checking after the push matters because it tests what Kaggle actually stored, not what we meant to send. Video links and the batch of fights are written into a copy of the job script at build time, so they live only inside the private kernel, never in git.
A smoke test showed YouTube doesn't block downloads from Kaggle, so the job downloads its own video, processes it and deletes it. Video never touches the laptop. The job installs our package from GitHub at an exact commit, which makes every run reproducible from the commit hash plus the config, and it uses Kaggle's own torch rather than downloading another copy.

**How it works.**
1. `cornerman kaggle batch --n 5` picks the next fights whose final stage (pose) isn't on the laptop yet, preferring fighters with the most fights.
2. It fills the fight list, config, repo and commit into `kaggle/perception/job.py`, checks privacy, pushes, and confirms on Kaggle.
3. The job loops over fights: run the pipeline (which writes each stage's output atomically), render an overlay for the first few, delete the raw video, append to `job_log.json`. It stops starting new fights after 10 hours.
4. `cornerman kaggle import` pulls the output and copies only fully finished fights into `data/derived/`. A fight that was cut off simply isn't imported, so the next batch picks it up again. That's the whole resume mechanism, and it works because the runner never leaves a half-written stage that looks complete.

**What breaks it.** YouTube could start blocking Kaggle's IP ranges. The fallback is to download on the laptop and upload a private Kaggle dataset per batch, deleted after processing. A commit that isn't pushed can't be installed by the job, so `batch` refuses to push in that case. Kaggle's Python and torch versions change without notice, which is why CI also tests on Python 3.13.

**Interview questions.**
1. *How do you make sure nothing goes public?* The check is in code on every call, not in a habit: the public flag is refused, metadata must say private, and after every push we read the kernel's metadata back from Kaggle and fail loudly if it isn't private. Tests cover each of those paths.
2. *What happens if the session dies on fight 4 of 5?* Fights 1 to 3 are complete on disk. If Kaggle saved the output, import copies those three and ignores fight 4's partial stages. If it didn't, nothing is imported. Either way the next batch is built from what's actually finished on the laptop, so fight 4 gets redone.
3. *Why install from a git commit instead of uploading code?* The commit hash pins exactly what ran, CI has already tested that commit, and there's no second copy of the code to drift.

---

## 10. Shot boundaries

**Problem.** Broadcast fight video cuts between cameras, replays and graphics. A tracker that runs across a cut will glue two different people together, and everything after that inherits the mistake.

**Why this approach.** TransNetV2 is a small network trained specifically to find cuts and gradual transitions (fades, wipes). It looks at tiny 48x27 frames, so it reads the whole video in a couple of minutes. Simpler methods that threshold the difference between consecutive frames miss gradual transitions and fire on fast motion, which is exactly what a fight has.

**How it works.** Every frame gets a probability that a transition is happening. Runs of frames above 0.5 are transitions, and the stretches between them are shots. Frames inside a transition belong to no shot and are skipped. Each shot is then marked live if at least half of its sampled frames show two or more people big enough to be on the mat, which removes crowd shots, close-ups and graphics.

**What breaks it.** Replays and slow motion are separate shots but still "live" by this test. Replays are counted twice until Phase 2 adds replay matching. A very fast whip pan can look like a cut, which only splits a shot in two (harmless: identity re-joins the tracks).

**Interview questions.**
1. *Why reset the tracker at every cut instead of letting it handle it?* A tracker predicts where a box moves next. Across a cut that prediction is meaningless, so any match it makes is luck. It's cleaner to start fresh and leave cross-shot linking to the identity step, which uses appearance.
2. *How would you tell a replay from live action?* Replays often come with a graphic wipe and slower motion, and they show content we've already seen. Matching a shot's pose sequence against earlier footage finds them; that's planned for Phase 2.
3. *What's the cost of a missed cut?* One tracklet contains two people. The cannot-link clustering can't fix that, so identity for that tracklet is wrong. That's why we use a learned detector here rather than a cheap one.

---

## 11. Person detection and tracking

**Problem.** Find every person in each sampled frame and follow each one through a shot.

**Why this approach.** RF-DETR Large is a transformer detector with about 56.5 COCO AP at real-time speed on a T4, under Apache 2.0. Popular alternatives (Ultralytics YOLO, BoxMOT) are AGPL, which doesn't fit a public repo we want others to reuse freely. For tracking, BoT-SORT from Roboflow's `trackers` package scored best among the light trackers on SportsMOT and DanceTrack, and it compensates for camera motion, which matters because broadcast cameras pan and zoom with the action.

**How it works.** We sample 10 frames per second (every 5th frame of a 50 fps video), batch them through the detector, and keep only people whose box is at least 20% of the frame height. That threshold drops the crowd and cornermen. BoT-SORT predicts where each existing track will be (a Kalman filter on box position and size), estimates the camera's motion from optical flow and corrects the predictions for it, then matches new boxes to predicted tracks: confident detections first, then weaker ones. Track IDs are made unique across shots by adding an offset at each cut. For each tracked box we also store a 64-number colour description (see identity).

**What breaks it.** Clinches and ground fighting: two bodies overlap, the detector may return one box for both or swap which is which. Tracks break on occlusion by the referee. The referee leaning in can be bigger than a fighter on the ground. None of these are fixed here; they show up as role switches and lower "both fighters visible" numbers in the health metrics.

**Interview questions.**
1. *Why sample at 10 fps instead of every frame?* Fighting moves fast, but actions last a few hundred milliseconds, so 10 fps still sees each one several times, and it cuts GPU time by five.
2. *Why does camera-motion compensation matter here?* When the camera pans, every person appears to move across the frame even if they stand still. Without compensation, the tracker's predictions are off by the pan and matches fail.
3. *How do you know the tracker works without labels?* We don't yet, fully. Until the gold set, we watch label-free numbers: how often both fighters are visible in live footage, and how often fighter A's box jumps to where fighter B just was.

---

## 12. Identity: red corner, blue corner, referee

**Problem.** Tracks break at every cut and occlusion, so a 15 minute fight becomes hundreds of short tracklets. We need to know which ones are the same person, which person is which corner, and which is the referee, and then which corner is which name in the fight list.

**What failed first, and why.** The first version described each box by a colour histogram over the whole box and clustered tracklets on that. On the pilot it mixed up both fighters and the referee. IMMAF kits are mostly black with red or blue trim, the referee wears black, and the light-blue canvas fills much of every box. Hue on black pixels is noise, so the histogram was mostly noise plus canvas. Three cheaper fixes (saturated pixels only, dropping canvas pixels, a different naming rule) didn't separate the fighters either. The lesson: the signal (red versus blue gloves and shin guards) was a few percent of each box, and averaging over the box drowned it.

**Why this approach.** Look where the signal is. Pose already finds wrists, knees and ankles, so we read colour exactly at the gloves, shins and shorts. That makes identity depend on pose, so pose now runs on everyone on the canvas (about 2 people per frame after the cage mask) and identity runs afterwards, without a GPU. That also means identity can be redone on the laptop as often as needed, and person crops are saved about once a second for trying other methods later.

**How it works.**
1. *Samples.* For each person and frame, a small patch at each wrist, halfway between each knee and ankle, and just below the hips. Canvas-coloured pixels are dropped and the median Lab colour is kept.
2. *Tracklet features.* The median a and b (Lab's red-green and blue-yellow axes, centred on zero) of gloves, shins and shorts. Red kit has a strongly positive; blue kit has b strongly negative; black kit and the referee sit near zero; bare skin is mildly positive on both.
3. *Cannot-link clustering.* Two tracklets seen in the same frame are different people, so they may not share a group. Up to four groups (red, blue, referee, anyone else on the canvas).
4. *Merge.* With only three people present, k-means with four groups splits one person in two. Groups whose colours are close and which never share a frame are merged, because that is one person seen at different times.
5. *Name.* The reddest gloves and shins are the red corner, the bluest the blue corner; of the rest, the group with the most screen time is the referee.
6. *Names.* IMMAF titles list the red corner first, which we checked against the broadcast score bar (red and blue edges beside the names). So the title's first fighter is red.

**What breaks it.** Seen on the pilot: it worked on a fight with full red and full blue kits, and failed on two 2023 Worlds fights where both kits were black with small trim and the gloves were dark red or black; the patches landed in the right places but read little colour. Also: events without red and blue corner colours (some promotions use black and white, or no colour code). Gloves out of view or blurred for a whole tracklet. Clinch boxes that contain both fighters, so a "tracklet" mixes red and blue. A referee in red or blue. The score-bar convention is checked for IMMAF only.

**Interview questions.**
1. *Why sample at keypoints instead of training a re-identification network?* The discriminating signal is known and tiny (glove and shin colour). Sampling exactly there uses that knowledge directly, needs no training data, and is easy to check by eye. A re-ID network trained on pedestrians would see two similar black kits.
2. *Why up to four groups and then merge, instead of exactly three?* Cornermen and officials do step onto the canvas, and forcing them into three groups would put them in a fighter's group. But when only three people are present, a fourth group steals half of someone. Merging groups that never co-occur and look alike handles both cases.
3. *How do you link a corner to a name without face recognition?* The federation's own convention: red corner listed first in the title, verified on the broadcast graphic that shows both names with corner colours. No biometrics needed.

---

## 13. 2D pose

**Problem.** Later stages (state classification, strikes, takedowns) need body keypoints for each fighter: where the hips, hands, knees and head are.

**Why this approach.** ViTPose++ Large is a top-down pose model: give it a person box and it returns 17 COCO keypoints with confidences. It is the strongest permissive-licence option on COCO keypoints (Sapiens is stronger but non-commercial). Top-down suits us because we already have tracked boxes, and only two people per frame need poses. It runs through Hugging Face `transformers`, pinned to a model revision hash, so the weights can't change under us.

**How it works.** For each sampled frame in a live shot, take every tracked person on the canvas (fighters, referee, the odd cornerman; identity is decided afterwards from these poses), crop and resize it to the model's input, predict a heatmap per keypoint, and read off the peak location and height as position and confidence. ViTPose++ is trained on several datasets with a separate output head per dataset; we pass the COCO head's index.

**What breaks it.** Grappling. When bodies tangle, a box contains limbs from both fighters and the model can put one fighter's knee on the other's leg. Keypoint confidence drops, which the health metrics track per fight. The spec plans to lean on hip height, silhouette and torso orientation in clinch and ground states rather than individual keypoints, and to try 3D body mesh where 2D breaks.

**Interview questions.**
1. *Top-down or bottom-up pose?* Top-down: we already have tracked boxes with identities, and only two people matter, so cost is small and accuracy is higher. Bottom-up wins with crowds, which we don't have.
2. *How do you use keypoint confidence?* It's stored with every keypoint. Downstream features will ignore points below a threshold, and the median per fight is a health number.
3. *What would you try first if pose fails in grappling?* Measure it first (Phase 2 gold set split by state), then use the segmentation variant of the detector to separate bodies, and features that don't need exact limbs, like hip height and the shape of the combined silhouette.

---

## 14. Perception health metrics (label-free)

**Problem.** Real accuracy numbers need a gold set (Phase 2). Until then we still need to notice when perception is broken on a fight.

**How it works.** Per fight, from the stage outputs: the share of sampled frames in live shots; in live frames, the share where both fighters have a box and where the referee has one; role switches per live minute (frames where swapping A and B would explain the movement better than keeping them, which is what an identity swap looks like); and the median keypoint confidence. `cornerman metrics` prints them.

**What breaks it.** These measure consistency, not truth. Labels that are consistently wrong (fighter A always tracked as the referee) look perfect here. They're an early warning, not an accuracy number, and the write-up never presents them as one.

**Interview questions.**
1. *Why bother with metrics that aren't accuracy?* They're free, they cover every fight, and a fight with both fighters visible only 30% of the time is broken whatever the gold set says.
2. *How does the role-switch count work?* Between consecutive frames, compare the total movement if A stayed A and B stayed B with the movement if they swapped. If swapping explains the motion better, count a switch.
3. *What's a failure these miss?* Consistent errors, as above, and good-looking poses on the wrong person.

---

## 15. Making perception fast enough: FP16, overlap, and two GPUs

**Problem.** The first pilot took 2.3 to 2.5 times as long as the video itself. At that rate 1000 fights would need hundreds of Kaggle session hours, and the free quota gives about 30 a week.

**Why this approach.** Three cheap changes, each aimed at a different idle resource, before anything that changes what the models compute:
- *FP16.* A T4's tensor cores do half-precision matrix maths several times faster than FP32, and inference doesn't need FP32's precision: the weights were trained with noise far larger than FP16 rounding. Half precision also halves the memory traffic for activations.
- *Overlap.* Before, the GPU sat idle while the CPU decoded the next batch of frames, and the CPU sat idle while the GPU ran. A background thread now decodes ahead into a queue. OpenCV releases Python's global interpreter lock while decoding, so a thread is enough; no extra processes.
- *Both GPUs.* Kaggle's T4 machine has two cards and the pilot used one. The job now runs one worker process per card, each restricted to its card with `CUDA_VISIBLE_DEVICES` before torch is imported, and gives each worker every other fight. Fights are independent, so this needs no communication and scales almost perfectly. Splitting one fight across two cards would need the tracker, which is sequential, to wait on both.

**How we know nothing broke.** Speed changes must not change the answers, so we reran the two pilot fights and compared outputs with `cornerman compare`. Track IDs are arbitrary numbers, so it matches boxes frame by frame with the Hungarian algorithm on IoU, then checks: are the same boxes there, are they grouped into the same tracks, do they get the same role, and how far apart are the keypoints. Result: 99.4%+ of boxes matched with median IoU 0.997, 99.6% same track grouping, 97 to 98% same role, keypoints 0.24 px apart at the median. FP16 is also part of the cache key, so the two kinds of output can never be mixed by accident.

**Result.** Per fight, 1.4 to 1.5 times faster, and with both GPUs a batch runs 2.75 times faster: 1.15 minutes of footage per minute of Kaggle session, up from 0.42.

**What breaks it.** FP16 can overflow in layers with large activations (attention softmax, some normalisations), which shows up as NaN boxes or keypoints. That's why the comparison is a required step, not a nice-to-have. Running two fights at once doubles host memory use and makes the two workers compete for CPU decoding, which is part of why the batch gain is 2.75x and not 2 x 1.5 = 3x. The biggest remaining cost is decoding each video three times (shots, tracking, pose).

**Interview questions.**
1. *Why did the role agreement come out at 97 to 98% when boxes agreed at 99.5%?* Identity is a global clustering over all tracklets. A tracklet near the boundary between two groups can flip when its features move by a rounding error. So the 2 to 3% is a measure of how unstable identity is, not of FP16 damage. It's a useful number in its own right: it marks the share of boxes whose role we shouldn't trust.
2. *Why not torch.compile or TensorRT?* Both give more speed but take minutes to build per session and are fussy about variable batch shapes. They're worth it once the pipeline stops changing; FP16 was a one-line change with a measurable check.
3. *How would you find the next bottleneck?* The runner now records each stage's wall time. Shot detection takes 4 to 5 minutes, mostly decoding every frame of a 50 fps video to feed a model that only needs 48x27 images, so that's the next target, followed by decoding the video once instead of three times.

---

## 16. Decoding each video once

**Problem.** Three stages (shots, tracking, pose) each read the video. Even though tracking and pose only use every 5th frame, a compressed video can't be read that way cheaply: H.264 stores most frames as differences from earlier ones, so to show frame 500 the decoder has to decode the frames it depends on, which in practice means all of them. Each read was a full decode.

**How we found it.** We timed it instead of guessing. A software decode of one 14-minute 720p50 fight took 164 s on the laptop with every core and 268 s on 2 threads, which matched the shots stage's 210 to 300 s on Kaggle. A first measurement with OpenCV on the Mac said 20 s, which would have pointed the other way: it was using the Mac's hardware video decoder, which Kaggle machines don't have. Lesson: measure on hardware like the target's, or measure the specific thing (software decoding) the target will do.

**How it works.** One ffmpeg process decodes the video once and splits the stream in two (a `split` filter). One branch keeps every 2nd frame and shrinks it to 48x27 for the shot model. The other keeps every 5th frame at full size and writes it as a high-quality JPEG named by frame index. A small meta file records the settings; if they change, the cache is rebuilt. The cache sits next to the raw video and is deleted with it, so it never becomes stored data.

**What breaks it.** Disk: about 1.6 GB of JPEGs per fight, twice that with two fights in parallel. JPEG re-encoding changes pixels slightly, so outputs differ a little from decoding directly (the compare tool measures that). If a later stage reruns after the raw folder was deleted, the video is fetched and decoded again.

**Interview questions.**
1. *Why not just seek to every 5th frame?* Seeking jumps to the nearest keyframe, often seconds away, and decodes forward from there; for every 5th frame that's more work, not less.
2. *Why JPEG and not keep frames in memory?* 8,260 full frames at 1280x720 is about 23 GB raw. JPEG at high quality is around 200 KB a frame, and encoding runs in parallel with decoding.
3. *Why was downsampling shot detection a small win?* It halves the shot model's work, but the expensive part was decoding, and you can't decode half a video.

---

## 17. Cage mask

**Problem.** People outside the cage (officials, photographers, spectators standing up, cornermen at the fence) are big enough in the frame to pass the size filter, and the clustering put them in the "referee" group.

**Why this approach.** The physical rule is simple: everyone who matters is standing on the canvas. So find the canvas in each frame and keep people whose feet are on it. Colour is enough to find the canvas because it's one large, uniformly coloured surface, which is much cheaper than training a segmentation model.

**How it works.**
1. *Canvas colour.* In a band across the lower middle of ~60 frames, take the most common bright colour. The band avoids the broadcast score bar at the bottom; "bright" avoids the dark arena; "most common" rather than median avoids the fighters standing in the middle.
2. *Per camera.* Different cameras see the canvas differently (white balance), so each shot learns its own colour from its own frames, trusted only if it's near the fight-wide colour. This took frames-without-canvas from 42% to 3% on the pilot.
3. *Canvas area per frame.* At 160x90, mark pixels near that colour, comparing only chroma (the a and b channels of Lab), since brightness varies with exposure much more than hue does. Close the holes left by logos and legs, keep the large regions, take the convex hull.
4. *Inside test.* The bottom-centre of a person's box (their feet) must be within a small margin of the hull. If the box reaches the bottom of the frame, the feet are hidden, so any point across the box's width just above the score-bar line counts.

**What breaks it.** Someone outside the cage but close to the camera, overlapping the canvas in the image, is kept. A white or grey canvas has no chroma to match, so it falls back to full colour distance, which is more fragile under lighting changes. A fighter lifted off the canvas in a slam is briefly dropped.

**Interview questions.**
1. *Why chroma and not full colour?* The pilot's cageside camera saw the canvas at about a third of the main camera's brightness but a similar hue. Ignoring lightness made the match survive exposure changes.
2. *Why a convex hull?* Fighters, logos and the referee cover parts of the canvas, which would punch holes in the region. The canvas is convex in the image (a polygon seen in perspective), so the hull fills those holes.
3. *How did you check it without labels?* By looking at frames (walk-outs and spectators dropped, fighters kept) and by counting how many boxes the previous run had labelled as fighters that survive (87%, with the rest including outsiders the old run mislabelled).

---

## 18. Fighting time: filtering walk-outs, breaks and announcements

**Problem.** Fight videos include the walk-out, the referee's instructions, a minute between rounds and the result announcement. None of it is fighting, but all of it looked "live" (two big people on screen), so it inflated counts and wasted pose time.

**Why this approach.** We looked at the pilot data first. During rounds the fighters' boxes move 0.3 to 0.6 box heights per second; in breaks they sit on stools at about 0.1. Motion is free (it comes from the tracks) and doesn't depend on the camera or the event's graphics. Reading the round clock from the broadcast graphic would be more exact but needs OCR tuned per event.

**How it works.** Per sampled frame, the fighters' mean speed in box heights per second (so it doesn't depend on zoom). A 10 s centred rolling median smooths over still moments inside a round. Frames at or above 0.2 in a live shot are fighting. Then gaps shorter than 8 s between fighting stretches are filled (a stoppage, a clinch where the detector merges both fighters into one box) and fighting stretches shorter than 15 s are dropped. The check is the total: three 3-minute rounds should give a bit over 9 minutes, and the pilot fights gave 9.3 and 9.5, with segments lining up with the rounds.

**What breaks it.** A long stalled clinch against the fence (slow, and often one merged box) can be cut out. Fighters bouncing in place during instructions can be counted. The threshold was set on two fights and needs checking against round counts on more.

**Interview questions.**
1. *Why speed in box heights per second?* Pixels per second depend on how far the camera is zoomed in; dividing by the person's box height makes the number roughly the same for a wide shot and a close-up.
2. *Why a median and not a mean?* One fast moment (a single punch) or one tracking glitch shouldn't flip a whole window. The median needs most of the window to be moving.
3. *How would you validate it properly?* Against the round clock: once results are scraped (they include the round and time of a finish), and with OCR of the broadcast clock on a sample of fights, compare predicted fighting time with the true round time.

### Version 2: read the clock

**What went wrong with motion.** On more fights, walk-outs and celebrations (fighters bouncing, waving flags) moved fast enough to count, and long grappling stretches where the two fighters merge into one box (the shot isn't "live") were never counted. Motion can't tell "moving because fighting" from "moving because celebrating".

**Why the clock.** The broadcast score bar shows the round clock. A clock counting down about one second per second is the most direct evidence a round is on, whatever the camera shows. So the `arena` stage reads the score bar once a second with OCR, and the `fighting` stage trusts the clock first.

**How it works.** For each pair of readings close in time, the clock is "running" if it went down by about the time that passed (within a tolerance), "stopped" if it didn't move. One trap: between rounds many broadcasts show a rest timer (a one-minute countdown) in the same place, which also counts down. It's spotted by its start, a jump up to 65 s or less, and everything that keeps counting from there is marked stopped. Then: running clock means fighting, even in a shot that isn't live; stopped means not. Before the first running clock and after the last one is walk-out and celebration. Where the clock can't be read, motion decides, plus a crowd check (at most 3 people on the canvas over 5 s: fighters and referee; walk-outs and celebrations bring cornermen and officials).

**How we checked it.** Three ways. (1) Totals: fighting minutes against rounds seen on the clock times 3 minutes, median 1.04 over 19 fights. (2) The fallback, using frames where the clock is known as labels and hiding it: precision 0.98, recall 0.54 on the test half, so without the clock we keep little that's wrong but miss half the fighting. (3) A blind look at 40 thumbnails, 20 v2 keeps and 20 it drops that motion kept: about 9 in 10 right both ways.

**What breaks it.** Fights with a score bar the OCR reads badly (one older style read in 19% of readings). And if the first running clock is read late, the walk-out trim cuts the start of round 1.

**Interview questions.**
1. *Why trust a broadcast graphic over the video itself?* It's the referee's own timekeeping, already computed. The video only gives proxies (motion, people count) that celebrations and stalls fool. The cost is coverage: where the graphic is missing or unreadable, you need the proxy anyway.
2. *How do you test a fallback when you don't have labels where it runs?* Make labels where you do have them: on frames where the clock is readable, hide it, run the fallback, and compare. The catch is that those frames might differ from the ones where the clock is unreadable (different broadcasts), so the number is a guide, not a guarantee.

---

## 19. Identity experiments: labels from behaviour, and an honest held-out test

**Problem.** Colour rules for "who is red, who is blue" broke on kits that are mostly black. Two experiments (`experiments/`) tried learned appearance instead. This section is about how they were built and judged, because the judging matters as much as the method.

**How it works.**
1. *Group people per fight.* DINOv2-small (a general pretrained image model) gives patch features; averaging them only over the body, located by pose, keeps the canvas and crowd out. A small projection is then trained per fight with a triplet loss, where crops from one tracklet are the same person and crops from tracklets on screen together are different people. The fight's own structure is the supervision. Clustering with the cannot-link rule turns tracklets into person groups.
2. *Find the fighters.* The hard part. A logistic regression across fights scores "fighter or not" from the same features. Its labels come from behaviour, not from anyone labelling crops: on the ground or kicking means fighter (officials don't do either), never moving means not a fighter. The labels are noisy (about 75 to 80% right for fighters), which a linear model tolerates.
3. *Red or blue.* Between the two fighter groups only, compare the pooled share of red and blue pixels on the body.

**How it was judged.** Development used the 5 pilot fights. Then the method was frozen, trained on 20 fights, and run once on 30 fights nobody had looked at. Each fight was judged on a montage of 12 random crops per corner (at least 9 of 12 right, red and blue not swapped). A second variant (merged-clinch evidence) was fixed in advance and reported alongside, instead of being kept only if it helped. Result: 20 of 30 for the classifier alone, 9 of 30 with merged evidence.

**What breaks it.** Ground scrambles put both fighters in one group. Crops from outside fighting time (walk-outs, flag shots, officials) leak into fighter groups. A referee in a red shirt looks like the red corner. Merged clinch boxes contain the referee as often as the fighters, which is why that evidence backfired.

**Interview questions.**
1. *Why train on labels from behaviour instead of labelling a few hundred crops?* The project rule is no hand labelling, and behaviour labels scale to any number of fights for free. The cost is noise, so the model is kept simple (linear) and checked on fights it never trained on.
2. *Why report the variant that did worse?* Choosing between variants after seeing test results turns the test set into a development set and inflates the number. Both variants were fixed before the test, so both get reported.
3. *Red and blue were never swapped, but 10 fights failed. What does that tell you?* The colour decision is reliable once the right two groups are chosen. The weak step is grouping and picking the fighters, so that is where the next work goes: only crops from fighting time, and ground positions assigned from the standing frames around them.

## 20. Identity round 3: roles per crop, a confidence flag, and a second clean test

**Problem.** Round 2 got red and blue right on 20 of 30 held-out fights. Its failures were mixing, not swapping: a tracklet that runs through a ground scramble or a tangle can follow one fighter and then the other, and giving the whole tracklet one role makes half of it wrong. Separately, later models need to know which fights to trust, without anyone checking montages.

**How it works.**
1. *People, then fighters, as before.* Per-fight embedding (DINOv2 features over the body, a small projection trained with a triplet loss), cannot-link clustering, and the fighter classifier pick the two fighters and the referee.
2. *Roles per crop, not per tracklet.* Along each tracklet, every crop is scored against the red, blue and "other" centres, and a Viterbi pass picks the best sequence of roles with a cost for each switch. A single odd crop doesn't flip the role, but a real swap halfway through a tracklet does. Crops called "other" that sit nearest the referee group become referee. Boxes between crops take the role of the nearest crop of the same tracklet.
3. *Fighting time twice.* The motion filter runs once on everyone (to pick crops) and again on the chosen fighters only (for the stats).
4. *Confidence.* Four checks that need no labels, each scaled to 0..1 by a range set by reasoning, then averaged: one corner on two boxes in the same frame, how fighter-like the less fighter-like fighter is, the share of fighting-time crops given a corner, and how often red and blue are the closest pair when a third person is in view. A fight below 0.4 is flagged; `usable_fights` leaves flagged fights out.

**What was tried and dropped.** Fighting-time-only grouping with ground crops filled in from standing frames (worse: 16 of 30), and a behavioural referee detector (the person who steps between the fighters), whose labels were only 35 to 50% precise.

**How it was judged.** The 30 fights used in round 2 had been looked at, so they became development fights. The method and threshold were fixed and committed, and only then were the last 24 IMMAF fights processed and judged once. Result: 18 of 24 right (75%). The confidence ranks fights well (AUC 0.83) but the threshold of 0.4 was too low: unflagged fights were 15 of 19 right, and the flag caught only 2 of 6 wrong ones. Every fight scoring 0.5 or more was right.

**Interview questions.**
1. *Why a Viterbi pass instead of a majority vote per tracklet?* A majority vote assumes a tracklet is one person. Trackers swap identities in tangles, so the assumption fails exactly where it matters. Viterbi lets the role change along the tracklet, and the switch cost stops single noisy crops from changing it.
2. *The confidence score had AUC 0.83 but the flag barely helped. How can both be true?* AUC measures ranking over all thresholds; the flag uses one threshold. The wrong fights all scored low, but in a band (0.32 to 0.49) above the cut-off picked on dev. The dev set had only 9 wrong fights, too few to place a threshold precisely.
3. *Why not just move the threshold to 0.5 and report 13 of 13?* That threshold was picked after seeing the test, so 13 of 13 measures the choice, not the method. The honest numbers are the frozen ones (18 of 24; 15 of 19 unflagged). A new threshold needs a new clean test, or at least a pooled estimate labelled as optimistic.

### Identity per shot instead of per fight (2026-10-26)

**Problem.** The confidence score flags a whole fight when identity is doubtful, and a flagged fight is left out of anything per fighter. But identity usually goes wrong in places: a clinch or a ground scramble where the tracker swaps the fighters, while the standing exchanges around it are fine. Throwing away the whole fight wastes those clean minutes, and with few repeat fighters every minute matters.

**How.** Score each camera shot on its own. A tracklet never crosses a cut, so an identity error lives inside shots. Each shot gets the same checks as a fight (one corner on two boxes, share of crops given a corner, whether red and blue are the closest pair), plus one fight-level term (are the two people picked fighters at all), and one extra condition: both fighters must be boxed together in at least 30% of the shot. That last one came from looking: shots where one fighter was barely tracked passed the other checks because there was little to check, and that is where the wrong ones hid.

**How it was checked.** Twice by eye, blind to the rule's decision: shots were shuffled and drawn as rows of crops with a number only, verdicts were written down, and only then was the key opened. Round 1 was used to change the rule, so round 2, on fresh shots, is the number to quote: 26 of 28 judgeable kept shots right (93%). It gains 15% more fighting time and takes repeat fighters with usable data from 6 to 14.

**Interview questions.**
1. *Why not just lower the fight-level threshold?* A lower threshold lets in whole fights with their bad stretches too. Per-shot scoring keeps the good stretches of a doubtful fight and drops the bad ones, so it gains data without lowering precision.
2. *Why did you need a second sample?* The first one was used to choose the extra condition. A number measured on the data you tuned on is optimistic, so the reported precision comes from shots drawn after the rule was fixed and committed.
3. *What does "can't judge" do to the number?* About a fifth of the round-1 shots had too few clean crops, mostly tangles. The precision is over judgeable shots, so it says nothing about tangled ones. Rule v2 mostly drops those, which is why only 2 of 30 kept shots were unjudgeable in round 2.

## 21. The silver set: labels from labellers that agree

**Problem.** Every perception number needs something to compare against (hard rule 4), and nobody is labelling clips by hand. The plan is a "silver" set: clips where several independent automatic labellers agree. Agreement isn't truth, but three labellers that fail in different ways rarely make the same mistake.

**How it works.**
1. *Clips.* 400 clips of 3 s from fights whose identity isn't flagged, in fighting time and inside one shot. A rough position proxy that needs no labels (torso angle, box shape and overlap, both fighters boxed or not) sorts clips into apart, close, ground and merged, and the sample takes 100 of each, spread over fights. Otherwise it would be almost all fighters standing apart.
2. *Corners drawn in.* Each clip shows the red-corner fighter in a red box and the blue-corner fighter in a blue box, from our identity output. The VLM is asked about "the fighter in the red box", so both labellers judge the same two people, even when both kits are black.
3. *Labeller A, rules.* Pose and box geometry: down if the torso is near horizontal or the hips are near ankle height; on the ground, the fighter with clearly higher hips is on top; standing and overlapping means clinch. Coarse labels only, because geometry can't see grips or the fence.
4. *Labeller B, a video VLM* (Qwen3-VL-4B, open weights, run on Kaggle) with two prompts that reach a state differently: one picks from the taxonomy, the other answers plain questions (on the ground? who's on top? clinched? pinned on the fence?) and code works out the state. An answer that is an impossible pair, or can't be decided, is dropped rather than guessed.
5. *Silver* = both prompts say live fighting, give the same pair of states, and the rules' coarse label fits.
6. *Oversampling where agreement is rare.* Only 18% of close clips and 4% of merged clips survived agreement in the first full run, so the next sample takes four times as many of those. Without hand labels, the way to get more of a rare, hard class is to label more candidates and keep the ones that pass.

**Checking a labeller on someone else's labels.** The rules were measured on ViCoS, a jiu-jitsu dataset with hand-labelled positions and keypoints (non-commercial licence, used for evaluation only). Thresholds were fixed before looking: standing versus ground 97% right; with someone on top, the right athlete 82%, the wrong one 2%, unsure 16%. Which athlete each label names (top, or guard player) was checked from the keypoints instead of assumed.

**What the pilots taught.** A small pilot (40 clips) before the full run caught four things a full run would have wasted hours on: the two prompts read "clinch against the fence" differently (Ansh then ruled that a clinch is a clinch wherever it happens, and cage pressure is only at striking range), the taxonomy didn't say what "one fighter standing over a downed one" is (now ground-top), the 8B model didn't fit the GPUs, and one model copy per GPU ran out of memory. Each fix is in DECISIONS.md.

**Position bias and the swap test.** A VLM asked "who controls the clinch, red or blue?" can favour whichever colour the prompt names first, whatever the footage shows. The direct prompt called red the controller in all 114 clinches, which can't be true. The test is cheap: ask the same question with the colours swapped in the text (blue named first) and keep a label only if it names the same fighter both times. A label that flips with the wording came from the wording. It also measures the bias: with red first the questions prompt picked red 30% of the time, with blue first 47%. After the filter, red is the controller in 10 of 26 silver clinches, no lean left.

**The lower tier.** Many tangled clips have one box around both fighters, so the rules can't say anything and the clip can never be silver. Where both prompts still agree, the label goes in a separate lower tier, reported apart and never mixed in. The by-eye check shows why: its state family was right on 16 of 16 sampled clips, but 2 of 6 directions (who is on top, who has their back to the fence) were wrong. The rules are what catches wrong directions, so without them only the family can be trusted.

**Interview questions.**
1. *Why ask the same model two different prompts instead of one prompt twice?* Asking twice with greedy decoding gives the same answer, so agreement says nothing. Two prompts that reason differently (pick a label versus answer facts) fail differently; when they agree, the label is more likely right. The pilot showed it: with the 2B model, the direct prompt put the wrong fighter on top on clips where the questions prompt was right.
2. *What bias does a consensus set have?* It keeps the cases the labellers find easy. Here that means fighters boxed separately (the rules need that), clear positions, and few transitions. Accuracy measured on silver clips overstates accuracy on tangled clinches and scrambles, so every number on it says "silver" and the hard strata are reported separately.
3. *Why pick the smaller model?* On the clips both models answered, the 4B's prompts agreed with each other more often than the 8B's, the two models gave identical labels wherever both were self-consistent, and the 4B was 2.4 times faster and fit the hardware. A bigger model is a hypothesis, not a default; the pilot didn't support it.

---

## 22. Pose in tangles: a negative result, and what to use instead

**Problem.** Tracking calls a shot live only when most of its frames show two people. In a long clinch or a ground exchange the detector draws one box around both fighters, so those shots fail the test and never got pose: about 197 of 626 video minutes. Rerunning the same pose model there would give one skeleton for two bodies, so the question was whether any off-the-shelf method separates tangled bodies.

**What was tried.** Three methods with permissive licences, on the same 96 frames (16 shots, one per fight, 6 frames each, seed 0):
- A, top-down (today): RF-DETR finds person boxes, ViTPose++ finds 17 keypoints inside each box.
- B, masks then top-down: RF-DETR Seg gives each person a pixel mask; ViTPose++ runs on each person's box with every other person's pixels greyed out, so it can't take the other fighter's limbs.
- C, bottom-up: RTMO finds all keypoints in the image and groups them into people in one pass, with no boxes first. Bottom-up models are built for crowds where people overlap, so this was the best bet on paper.

**How it was judged.** The three drawings of each frame side by side, in a random order per row, labelled X, Y, Z; each scored 2, 1 or 0 fighters with a clean skeleton (limbs on the right body, nothing joined across the two), or "na" if the frame wasn't fighting. The verdicts were committed before the key (which letter is which method) was opened. One agent judged everything: blind to the method, not to the footage.

**Result.** 68 frames could be judged (4 shots were breaks, ceremonies or close-ups).

| | A | B | C |
|---|---|---|---|
| standing clinch (23 frames), two clean | 65% | 65% | 39% |
| ground (45 frames), two clean | 2% | 7% | 4% |
| ground, at least one clean | 36% | 36% | 27% |

No off-the-shelf method separates ground tangles at broadcast resolution: all three draw one skeleton across both bodies, or the top fighter only. The bottom-up model, the one designed for overlap, did worst everywhere. Its grouping step needs to see enough of each body to tell them apart, and in a guard or a scramble most of the bottom fighter is hidden under the top one. On standing clinches top-down works most of the time, and masks don't beat plain boxes.

**What we do instead (Ansh, 2026-10-26).** Keep B (it ties A where pose works and gives masks for free) on all fighting frames at 2 fps. Keep a skeleton wherever it is clean, even if the other fighter's isn't. On the ground, decide who is on top from the masks (whose silhouette sits higher) rather than from keypoints, and check that rule against the ViCoS hand labels and the silver ground clips.

**Interview questions.**
1. *Why report a method that failed?* It saves the next person the experiment, and it explains a gap in the data: per-fighter numbers from ground frames come from coarser features, so they are less detailed than standing ones. A write-up that hides that overstates what the system knows.
2. *Why would bottom-up do worse in tangles, when it's meant for crowds?* Crowded scenes in its training data are mostly people standing next to or in front of each other, with most of each body visible. In a guard one body is mostly under the other. The keypoints it can't see it can't group, and the ones it sees get attached to the wrong person.
3. *How do you judge three methods fairly by eye?* Same frames, same colours and keypoint threshold, no boxes drawn, a random order per frame, verdicts written down and committed before opening the key. It's still one judge, so the numbers have wide error bars (45 ground frames).
