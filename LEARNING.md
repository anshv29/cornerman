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
