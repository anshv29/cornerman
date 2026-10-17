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

## Shaky or worth a second look
- The stage code hash doesn't see changes in helper functions. Each stage has a manual `version` string to bump. Hashing the whole module would be stricter.
- Plug-in empirical Bayes ignores uncertainty in alpha and pi. Fine with hundreds of fighters, too narrow with few. Full Bayes comes in Tier 1.
- The first `pre-commit` run on a new machine downloads Go to build gitleaks, which takes a minute or two.

## Next: Tier 1, Phase 1 (data and perception skeleton), in progress
Ansh signed off on Tier 0 and started Tier 1 on 2026-10-17.
1. Kaggle smoke test: can yt-dlp download inside a Kaggle job? If YouTube blocks it, download on the laptop and upload a private Kaggle dataset instead.
2. Model selection pass (hard rule 5): shot boundary, detector, tracker, re-ID, 2D pose that fit a Kaggle T4/P100. Log in DECISIONS.md.
3. Footage collection with yt-dlp: public amateur sources only, adult bouts only (no youth, junior or cadet divisions). Build `data/fights.csv` and `data/fighters.csv` locally with anonymized fighter IDs. Target 50 fights.
4. Kaggle job scaffolding in `kaggle/`: resumable per-fight jobs, raw video deleted after each fight, every kernel and dataset private.
5. Real ingest, shots, tracking, identity and pose on 5 fights, with overlay videos kept local.
6. MLflow comes later, before the first real model fit (Phase 4).

## Later (noted 2026-10-17)
- **More amateur sources before Tier 2.** Only 18 of 125 fighters have 2+ fights in the IMMAF data. Per-fighter models need depth, so Tier 2 (scaling to 1000+ fights) must add more public amateur sources with repeat fighters. Same rules apply: adult bouts only, no pro promotion broadcasts.
- **Fight results before Tier 3.** Result, method, round/time and weight class are empty because IMMAF's YouTube descriptions don't include them. Scrape them from IMMAF's website (check its terms first) before Tier 3, where the simulator and Elo baseline need outcomes.

## Blocked
- Nothing blocking. Pushing derived data off the laptop waits on a DVC remote, which Ansh has said can wait.
