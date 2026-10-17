# Cornerman

[![CI](https://github.com/anshv29/cornerman/actions/workflows/ci.yml/badge.svg)](https://github.com/anshv29/cornerman/actions/workflows/ci.yml)

Cornerman turns public amateur MMA footage into a log of (state, action) events for both fighters, then models how each fighter fights: what they do in each situation, how that changes as they tire, how they react to the opponent, and how they compare to everyone else. On top of that it will simulate fights between two fighters and suggest how to beat a given opponent. The full plan is in [SPEC.md](SPEC.md).

## Status

Tier 0 (foundation) is done. No real footage has been processed yet. What exists:

- A pipeline runner with content-hash caching, running six stub stages (ingest, shots, tracking, pose, geometry, events) on one synthetic fixture fight. The stubs make seeded fake data with the same shape the real perception stages will produce.
- A data contract for event tables. Fights that fail it are quarantined with a reason.
- Derived data versioned with DVC.
- A parameter recovery harness (interval coverage and simulation-based calibration), proven on a hierarchical Dirichlet-multinomial model. The correct posterior gives 90.8% coverage for 90% intervals. A deliberately overconfident one gives 76.8% and is flagged.
- CI on every push: ruff, pyright (strict), pytest, and a secret scan of the full git history.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest                                          # all tests, about 2 seconds
uv run cornerman run --config configs/fixture.yaml     # run the pipeline on the fixture fight
uv run cornerman run --config configs/fixture.yaml     # again: every stage is cached
uv run cornerman run --config configs/fixture.yaml --stage pose --force   # rerun one stage
```

Outputs go to `data/derived/{fight_id}/{stage}/`, each with a `manifest.json` holding its cache key and output hash.

For development, also run `uv run pre-commit install` once.

## Layout

```
src/cornerman/
  taxonomy.py      the locked 9 states, momentum flag, and 8 actions
  config.py        typed run config loaded from YAML
  pipeline/        runner, hashing, stage definitions, CLI
  contracts/       pandera schema for event tables
  models/          statistical models (so far: Dirichlet-multinomial)
  eval/            parameter recovery and SBC harness
configs/           run configs
data/fixtures/     synthetic fixture fight (in git)
data/derived/      pipeline outputs (in DVC, not git)
tests/             unit, property, recovery, golden
```

## Docs in this repo

- [LEARNING.md](LEARNING.md): how each component works and why, written for interview prep.
- [DECISIONS.md](DECISIONS.md): every tool and design choice, with the alternatives considered.
- [PROGRESS.md](PROGRESS.md): what's done, what's next, what's blocked.
- [costs.md](costs.md): every dollar spent on GPU or hosted services.

## Data rules

Only public amateur footage is used, never professional promotion broadcasts. Video is never committed or redistributed: it is processed and deleted, and only derived data (tracks, poses, events) is kept. Fighters appear only as anonymized IDs (`F0001`) in anything public.
