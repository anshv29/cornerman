# Cornerman

SPEC.md is the source of truth for this project. Read all of it at the start of every session.

## Every session
1. Read SPEC.md, then PROGRESS.md (create it on the first session).
2. Pick up where PROGRESS.md says. First session starts at Tier 0.
3. Before ending a session, update PROGRESS.md: what's done, what's next, what's blocked and why, current spend.

## How Ansh wants things
- Plain language in all docs, comments and commit messages. No em dashes, no marketing words.
- Keep LEARNING.md up to date as you build (hard rule 12 in SPEC.md). Explain the why and the how, not just the what.
- Commit small and often: one logical change per commit (a new module, a passing test, a fix), with a clear message saying what and why. No empty or padding commits.
- Once a GitHub remote named origin exists, push after every commit. If there's no remote yet, just commit locally.
- Do not add Co-Authored-By lines, session links, or any other attribution trailers to commit messages or PR descriptions.

## Things only Ansh can do
Creating accounts, logging in, API tokens, and loading prepaid GPU credit. When you hit one of these:
- write the exact steps he needs to take at the top of PROGRESS.md under "Needs Ansh",
- then keep working on anything that isn't blocked.
Known ones coming up: Kaggle account with phone verification (needed for GPU) and its API token, a GitHub repo.

## Never
- Add a pay-as-you-go API key or anything that can auto-bill.
- Commit or upload video files.
- Call Claude or any paid model from pipeline code.
- Use pro promotion footage.
- Put real fighter names in anything public (the GitHub repo, the web app, README examples). Use anonymized fighter IDs there. Real names stay in local data only.
