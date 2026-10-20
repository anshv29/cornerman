"""Command line entry point.

`cornerman run --config configs/fixture.yaml` runs the pipeline.
`cornerman collect --config configs/collect.yaml` builds the local fight registry.
`cornerman kaggle push|status|pull|batch|import` drives Kaggle jobs, always
through the privacy checks in cornerman.remote.kaggle.
`cornerman metrics` prints label-free perception health numbers per fight.
`cornerman compare --a DIR --b DIR` checks two runs of one fight agree.
"""

import argparse
import json
import logging
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from cornerman.collect.run import collect
from cornerman.config import PerceptionConfig, load_settings
from cornerman.contracts.perception import load_roles
from cornerman.pipeline.runner import run
from cornerman.pipeline.stage import Stage
from cornerman.pipeline.stages import PIPELINE
from cornerman.remote import batch, kaggle, videos

KAGGLE_BUILD = Path("kaggle/build")
PERCEPTION_KERNEL = Path("kaggle/perception")


def pipeline_for(name: str) -> list[Stage]:
    """The stub stages, or the real perception stages (which need the gpu extra)."""
    if name == "stub":
        return PIPELINE
    from cornerman.perception.stages import REAL_PIPELINE

    return REAL_PIPELINE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cornerman")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run the pipeline")
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--fight", action="append", help="only this fight (repeatable)")
    p_run.add_argument("--stage", help="run only this stage and what it depends on")
    p_run.add_argument("--force", action="store_true", help="ignore the cache")
    p_run.add_argument("-v", "--verbose", action="store_true")

    p_col = sub.add_parser("collect", help="fetch fight metadata and build the registry")
    p_col.add_argument("--config", type=Path, required=True)

    p_kg = sub.add_parser("kaggle", help="push, check and pull private Kaggle jobs")
    kg = p_kg.add_subparsers(dest="kaggle_command", required=True)
    p_push = kg.add_parser("push", help="build a kernel folder and push it")
    p_push.add_argument("folder", type=Path, help="e.g. kaggle/smoke")
    p_push.add_argument(
        "--values", type=Path, help="local JSON file with build-time values (not in git)"
    )
    p_status = kg.add_parser("status", help="show a kernel's run status")
    p_status.add_argument("folder", type=Path)
    p_pull = kg.add_parser("pull", help="download a kernel's output")
    p_pull.add_argument("folder", type=Path)
    p_pull.add_argument("--dest", type=Path, required=True)
    p_batch = kg.add_parser("batch", help="push the next unprocessed fights to the GPU job")
    p_batch.add_argument("--config", type=Path, default=Path("configs/kaggle_real.yaml"))
    p_batch.add_argument("--fights-csv", type=Path, default=Path("data/fights.csv"))
    p_batch.add_argument("--n", type=int, default=0, help="next n unprocessed fights")
    p_batch.add_argument(
        "--fight", action="append", default=[], help="these fights, even if already done"
    )
    p_batch.add_argument("--overlays", type=int, default=0)
    p_batch.add_argument(
        "--source",
        choices=["dataset", "youtube"],
        default="dataset",
        help="dataset: download here and upload a private Kaggle dataset (YouTube blocks "
        "Kaggle after a few downloads); youtube: the job downloads itself",
    )
    p_batch.add_argument(
        "--reuse-dataset",
        action="store_true",
        help="attach the videos dataset already on Kaggle instead of uploading again",
    )
    p_import = kg.add_parser("import", help="pull the GPU job's output into data/")
    p_import.add_argument("--data", type=Path, default=Path("data"))
    p_import.add_argument(
        "--derived", type=Path, help="put fights here instead of data/derived (for comparisons)"
    )

    p_met = sub.add_parser("metrics", help="perception health numbers for processed fights")
    p_met.add_argument("--derived", type=Path, default=Path("data/derived"))
    p_met.add_argument("--fps", type=float, default=10.0)
    p_cmp = sub.add_parser("compare", help="check two runs of the same fight agree")
    p_cmp.add_argument("--a", type=Path, required=True, help="derived/{fight_id} of run A")
    p_cmp.add_argument("--b", type=Path, required=True)
    p_probe = sub.add_parser(
        "identity-probe", help="train the fighter-versus-official classifier for identity"
    )
    p_probe.add_argument("--fights", type=Path, required=True, help="file of fight IDs")
    p_probe.add_argument("--derived", type=Path, default=Path("data/derived"))
    p_probe.add_argument("--out", type=Path, default=PerceptionConfig().identity_probe)
    p_qual = sub.add_parser("identity-quality", help="identity confidence and flags per fight")
    p_qual.add_argument("--derived", type=Path, default=Path("data/derived"))
    p_gold = sub.add_parser("gold-clips", help="sample candidate clips for the silver set")
    p_gold.add_argument("--derived", type=Path, default=Path("data/derived"))
    p_gold.add_argument("--n", type=int, default=400)
    p_gold.add_argument("--clip-s", type=float, default=3.0)
    p_gold.add_argument("--seed", type=int, default=0)
    p_gold.add_argument("--out", type=Path, default=Path("data/gold/clips.parquet"))
    p_cut = sub.add_parser("gold-cut", help="cut the sampled clips (downloads each video once)")
    p_cut.add_argument("--clips", type=Path, default=Path("data/gold/clips.parquet"))
    p_cut.add_argument("--config", type=Path, default=Path("configs/kaggle_real.yaml"))
    p_cut.add_argument("--fights-csv", type=Path, default=Path("data/fights.csv"))
    p_cut.add_argument("--out", type=Path, default=Path("data/gold/clips"))
    args = parser.parse_args(argv)

    if args.command == "identity-probe":
        return _identity_probe(args.fights, args.derived, args.out)
    if args.command == "identity-quality":
        from cornerman.contracts.perception import identity_quality

        q = identity_quality(args.derived)
        q.to_parquet(args.derived / "identity_quality.parquet", index=False)
        print(q[["fight_id", "confidence", "flagged"]].round(3).to_string(index=False))
        print(f"{int(q['flagged'].sum())} of {len(q)} fights flagged")
        return 0
    if args.command == "gold-clips":
        return _gold_clips(args.derived, args.n, args.clip_s, args.seed, args.out)
    if args.command == "gold-cut":
        return _gold_cut(args.clips, args.config, args.fights_csv, args.out)
    if args.command == "kaggle":
        return _kaggle(args)
    if args.command == "collect":
        return _collect(args.config)
    if args.command == "compare":
        from cornerman.eval.compare_runs import compare_fight

        for k, v in compare_fight(args.a, args.b).items():
            print(f"{k:<18} {v:.4g}")
        return 0
    if args.command == "metrics":
        return _metrics(args.derived, args.fps)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    settings = load_settings(args.config)
    stages = pipeline_for(settings.pipeline)
    fights: list[str] = args.fight or settings.fights
    stage: str | None = args.stage
    known = {s.name for s in stages}
    if stage is not None and stage not in known:
        parser.error(f"unknown stage {stage!r}; choose from {sorted(known)}")

    results = run(stages, settings, fights, target=stage, force=args.force)
    for r in results:
        print(f"{r.fight_id:<24} {r.stage:<12} {r.status}")
    counts = Counter(r.status.value for r in results)
    print(", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    return 0


def _collect(config: Path) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tables = collect(load_settings(config).collect)
    repeat = int((tables.fighters["n_fights"] > 1).sum())
    print(f"{len(tables.fights)} fights, {len(tables.fighters)} fighters ({repeat} with 2+ fights)")
    print("skipped:")
    print(tables.skipped["reason"].str.split(":").str[0].value_counts().to_string())
    return 0


def _kaggle_user() -> str:
    return str(kaggle.read_kernel_metadata(PERCEPTION_KERNEL)["id"]).split("/")[0]


def _push(
    folder: Path, values: dict[str, object], metadata: dict[str, object] | None = None
) -> None:
    built = kaggle.build_kernel(folder, KAGGLE_BUILD / folder.name, values, metadata)
    print(kaggle.push_kernel(built))
    print(f"pushed {kaggle.read_kernel_metadata(folder)['id']}, confirmed private on Kaggle")


def _kaggle(args: argparse.Namespace) -> int:
    if args.kaggle_command == "batch":
        fights = pd.read_csv(args.fights_csv)
        if args.fight:
            chosen = fights[fights["fight_id"].isin(args.fight)]
        else:
            chosen = batch.select_fights(fights, Path("data/derived"), args.n)
        if chosen.empty:
            print("no fights to run")
            return 1
        print(f"{len(chosen)} fights: {', '.join(chosen['fight_id'])}")
        values = batch.build_values(chosen, args.config, args.overlays)
        metadata: dict[str, object] = {}
        if args.source == "dataset" and args.reuse_dataset:
            ref = videos.dataset_ref(_kaggle_user())
            if not kaggle.dataset_exists(ref):
                print(f"{ref} doesn't exist; run without --reuse-dataset")
                return 1
            kaggle.check_not_public(f"https://www.kaggle.com/api/v1/datasets/view/{ref}")
            metadata["dataset_sources"] = [ref]
        elif args.source == "dataset":
            settings = load_settings(args.config)
            ref = videos.stage_and_upload(
                chosen, _kaggle_user(), Path("data/raw/upload"), settings.perception.max_height
            )
            print(f"uploaded {len(chosen)} videos to {ref}, confirmed private")
            metadata["dataset_sources"] = [ref]
        _push(PERCEPTION_KERNEL, values, metadata)
        return 0
    if args.kaggle_command == "import":
        data: Path = args.data
        ref = str(kaggle.read_kernel_metadata(PERCEPTION_KERNEL)["id"])
        out = KAGGLE_BUILD / "perception_output"
        shutil.rmtree(out, ignore_errors=True)  # else an older batch's fights come along
        src = kaggle.pull_output(ref, out)
        derived: Path = args.derived or data / "derived"
        done = batch.import_outputs(src, derived, data / "overlays", data / "logs", data / "crops")
        print(f"imported {len(done)} finished fights")
        ref = videos.dataset_ref(_kaggle_user())
        if kaggle.dataset_exists(ref):
            kaggle.delete_dataset(ref)
            print(f"deleted {ref}, so no raw video stays on Kaggle")
        return 0
    folder: Path = args.folder
    ref = str(kaggle.read_kernel_metadata(folder)["id"])
    if args.kaggle_command == "push":
        values_file: Path | None = args.values
        values: dict[str, object] = json.loads(values_file.read_text()) if values_file else {}
        _push(folder, values)
    elif args.kaggle_command == "status":
        print(kaggle.kernel_status(ref))
    else:
        print(kaggle.pull_output(ref, args.dest))
    return 0


def _identity_probe(fights_file: Path, derived: Path, out: Path) -> int:
    """Behaviour labels on these fights' crops, then a logistic regression on their features."""
    import numpy as np

    from cornerman.perception.identity import behaviour_labels, train_probe

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for fid in fights_file.read_text().split():
        d = derived / fid
        crops = pd.read_parquet(d / "crop_features" / "crops.parquet")
        feats = np.load(d / "crop_features" / "features.npy")
        lab = behaviour_labels(
            pd.read_parquet(d / "tracking" / "tracks.parquet"),
            pd.read_parquet(d / "pose" / "poses.parquet"),
            crops,
        )
        ok = (lab >= 0) & crops["has_mask"].to_numpy(dtype=bool)
        xs.append(feats[ok])
        ys.append(lab[ok])
    y = np.concatenate(ys)
    train_probe(np.concatenate(xs), y).save(out)
    print(f"{out}: {int((y == 1).sum())} fighter and {int((y == 0).sum())} official crops")
    return 0


def _gold_clips(derived: Path, n: int, clip_s: float, seed: int, out: Path) -> int:
    """Clips from fights whose identity isn't flagged, stratified by a position proxy."""
    from cornerman.contracts.perception import usable_fights
    from cornerman.gold.clips import fight_clips, sample

    fights = usable_fights(derived)
    clips = pd.concat([fight_clips(derived / f, clip_s) for f in fights], ignore_index=True)
    chosen = sample(clips, n, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_parquet(out, index=False)
    print(f"{len(clips)} candidate clips in {len(fights)} usable fights")
    counts = pd.DataFrame(
        {"candidates": clips["stratum"].value_counts(), "chosen": chosen["stratum"].value_counts()}
    )
    print(counts.to_string())
    print(f"{len(chosen)} clips from {chosen['fight_id'].nunique()} fights -> {out}")
    return 0


def _gold_cut(clips_path: Path, config: Path, fights_csv: Path, out: Path) -> int:
    """Clips with the corners drawn in, local only (data/ is gitignored)."""
    from cornerman.gold.cut import cut_all

    settings = load_settings(config)
    clips = pd.read_parquet(clips_path)
    fights = pd.read_csv(fights_csv)
    urls = dict(zip(fights["fight_id"].astype(str), fights["url"].astype(str), strict=True))
    report = cut_all(
        clips,
        urls,
        settings.paths.derived,
        out,
        Path("data/raw/gold"),
        settings.perception.max_height,
        settings.perception.fps,
    )
    print(report["status"].str.split(":").str[0].value_counts().to_string())
    print(f"{len(list(out.glob('*.mp4')))} of {len(clips)} clips in {out}")
    return 0


def _metrics(derived: Path, fps: float) -> int:
    from cornerman.eval.perception_metrics import fight_metrics

    rows: list[dict[str, object]] = []
    for fid in sorted(batch.done_fights(derived)):
        d = derived / fid
        if not (d / "identity" / "fighting.parquet").exists():
            continue  # stub output (the fixture fight) or the older pilot layout
        m = fight_metrics(
            frames=pd.read_parquet(d / "tracking" / "frames.parquet"),
            tracks=pd.read_parquet(d / "tracking" / "tracks.parquet"),
            roles=load_roles(d),
            live=pd.read_parquet(d / "tracking" / "live.parquet"),
            poses=pd.read_parquet(d / "pose" / "poses.parquet"),
            step_seconds=1 / fps,
            fighting=pd.read_parquet(d / "identity" / "fighting.parquet"),
        )
        rows.append({"fight_id": fid, **m})
    if not rows:
        print("no processed fights")
        return 0
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
