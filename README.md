# Cornerman

[![CI](https://github.com/anshv29/cornerman/actions/workflows/ci.yml/badge.svg)](https://github.com/anshv29/cornerman/actions/workflows/ci.yml)

Cornerman turns public amateur MMA footage into a log of (state, action) events for both fighters, then models how each fighter fights: what they do in each situation, how that changes as they tire, how they react to the opponent, and how they compare to everyone else. On top of that it will simulate fights between two fighters and suggest how to beat a given opponent. The full plan is in [SPEC.md](SPEC.md).

## Status

Tier 0 (foundation) is done. Tier 1 Phase 1 (data and perception skeleton) is mostly done: real perception has run on 2 amateur fights so far.

- **Fight list.** 54 adult amateur fights and 90 fighters from the IMMAF YouTube channel (15 fighters have 2 or more fights). Youth, junior and cadet bouts are skipped (checked in titles, descriptions, tags and the on-screen score bar), along with every bout of any fighter who appears in one. Where the score bar doesn't show the division, the bout is checked against IMMAF's published results and dropped unless it's confirmed adult. Fighters get anonymous IDs (`F0001`); real names stay on the laptop.
- **Perception pipeline.** Shot boundaries (TransNetV2), person detection (RF-DETR Large), tracking per shot (BoT-SORT), fighter A / fighter B / referee by cannot-link clustering on clothing colour, and 2D pose (ViTPose++ Large). Runs as private Kaggle GPU jobs, one fight per GPU.
- **Pilot, 2 fights.** Label-free health numbers, not accuracy (the gold set for real accuracy comes in Phase 2):

  | Fight | Both fighters boxed (live frames) | Role switches per minute | Median keypoint confidence |
  |---|---|---|---|
  | ft_97394d8b1a | 67% | 3.8 | 0.87 |
  | ft_9eec213a59 | 69% | 2.4 | 0.85 |

  ft_97394d8b1a was removed from the dataset later (one of its fighters had fought in a junior bout); its pilot numbers are kept as a record.

  Known weak spots: in clinches the detector often draws one box around both fighters; identity (fighter A, fighter B, referee) is unreliable when both kits and the referee's uniform are mostly black. A cage mask (people must stand on the canvas) and a motion filter for walk-outs and breaks have been added since the pilot; see [DECISIONS.md](DECISIONS.md).
- **Throughput.** 1.15 minutes of footage per minute of a Kaggle T4 x2 session, after an FP16 and two-GPU speed pass (0.42 before). Outputs before and after agree: 99.4% of boxes matched with median IoU 0.997, keypoints 0.24 px apart at the median. Details in [DECISIONS.md](DECISIONS.md).
- **Foundation (Tier 0).** Pipeline runner with content-hash caching and quarantine, pandera data contracts, DVC for derived data, and a parameter recovery harness (coverage and simulation-based calibration) proven on a hierarchical Dirichlet-multinomial model.
- CI on every push: ruff, pyright (strict), pytest on Python 3.12 and 3.13, and a secret scan of the full git history.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest                                          # all tests, a few seconds
uv run cornerman run --config configs/fixture.yaml     # run the pipeline on the fixture fight
uv run cornerman run --config configs/fixture.yaml     # again: every stage is cached
uv run cornerman run --config configs/fixture.yaml --stage pose --force   # rerun one stage
```

Outputs go to `data/derived/{fight_id}/{stage}/`, each with a `manifest.json` holding its cache key and output hash.

For development, also run `uv run pre-commit install` once.

Real fights (needs a Kaggle login via `uvx --from kaggle==2.2.4 kaggle auth login`; everything on Kaggle is created private and checked):

```sh
uv run cornerman collect --config configs/collect.yaml   # fight list from public metadata, into data/
uv run cornerman kaggle batch --n 5 --overlays 2         # stage videos privately, run the GPU job
uv run cornerman kaggle status kaggle/perception
uv run cornerman kaggle import                           # results into data/derived, delete staged videos
uv run cornerman metrics                                 # label-free health numbers per fight
```

## Layout

```
src/cornerman/
  taxonomy.py      the locked 9 states, momentum flag, and 8 actions
  config.py        typed run config loaded from YAML
  pipeline/        runner, hashing, stub stages, CLI
  collect/         fight list and fighter registry from public metadata, adult-only filter
  perception/      video I/O, shots, detection and tracking, identity, pose
  remote/          Kaggle jobs (privacy-checked), batches, private video staging
  products/        overlay videos
  contracts/       pandera schemas for events, fights, tracks and poses
  models/          statistical models (so far: Dirichlet-multinomial)
  eval/            recovery and SBC harness, perception health metrics, run comparison
configs/           run configs
kaggle/            GPU job scripts (private Kaggle kernels)
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
