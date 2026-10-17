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

This only spells out what the locked taxonomy implies. It doesn't add or remove any states.
