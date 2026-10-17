# Progress

## Needs Ansh

1. **Kaggle account with GPU access (blocks Tier 1 GPU work).**
   1. Sign up at https://www.kaggle.com (free).
   2. Go to your profile picture, then Settings, then "Phone verification", and verify a phone number. Without this, notebooks get no GPU and no internet.
   3. On the same Settings page, under "API", click "Create New Token". This downloads `kaggle.json`.
   4. On the laptop, run:
      ```sh
      mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
      ```
      Don't put it inside the repo. It's gitignored, and gitleaks scans for it, but keeping it out of the repo is safer.
   5. Tell the agent it's done. Don't add any payment method on Kaggle.
2. **A free DVC remote (not urgent; needed before real fights are processed on Kaggle).** Recommended: DagsHub (free, no card). Sign up at https://dagshub.com with GitHub, connect the `anshv29/cornerman` repo, then share the DVC remote URL and an access token with the agent, for example by putting the token in a gitignored `.env` file. Until then the DVC cache lives only on this laptop, which is fine because the current derived data rebuilds in seconds.

## Current spend
$0.00 of the $60 cap. See `costs.md`.

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

## Shaky or worth a second look
- The pairing for standing-opponent-hurt (with standing-neutral or standing-cage-pressured) is my reading of the spec, not something it states. Logged in DECISIONS.md. Needs Ansh's OK, since it touches the taxonomy.
- The stage code hash doesn't see changes in helper functions. Each stage has a manual `version` string to bump. Hashing the whole module would be stricter.
- Plug-in empirical Bayes ignores uncertainty in alpha and pi. Fine with hundreds of fighters, too narrow with few. Full Bayes comes in Tier 1.
- The first `pre-commit` run on a new machine downloads Go to build gitleaks, which takes a minute or two.

## Next: Tier 1, Phase 1 (data and perception skeleton)
Per SPEC.md, Tier 0 ends with a check-in with Ansh before Tier 1 starts.
1. Model selection pass (hard rule 5): check current open-weight detectors, trackers, pose models and video VLMs that fit a Kaggle T4/P100, then log the choices in DECISIONS.md.
2. Footage collection with yt-dlp: find public amateur MMA sources (not pro promotions), check their terms, and build `data/fights.csv` and `data/fighters.csv` locally (both gitignored, with anonymized IDs for anything public). Target 50 fights with metadata.
3. Kaggle job scaffolding in `kaggle/`: resumable per-fight jobs that download, process and delete video. Blocked on the Kaggle account above.
4. Replace the stub stages with real tracking, identity and pose on 5 fights, with overlay videos kept local.
5. Add MLflow (local) tracking before the first real model fit.

## Blocked
- GPU stages: waiting on the Kaggle account (Needs Ansh 1).
- Pushing derived data off the laptop: waiting on a DVC remote (Needs Ansh 2).
