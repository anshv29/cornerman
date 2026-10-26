"""Which pose method gives two clean skeletons on tangled clinch and ground shots, 2026-10-26.

    uv run python experiments/pose_tangle_check.py pick
    uv run python experiments/pose_tangle_check.py launch
    uv run python experiments/pose_tangle_check.py import
    uv run python experiments/pose_tangle_check.py sheet
    uv run python experiments/pose_tangle_check.py score VERDICTS_CSV

Tracking marks a shot live only when most of its sampled frames show two people.
Long clinch and ground shots merge the fighters into one box, fail that test, and
never got pose. Before rerunning pose on them (Ansh, 2026-10-26: free Kaggle GPU
only, with a method that separates tangled bodies, tested on 10 to 20 shots first),
three methods are compared on the same frames (kaggle/pose_test/job.py):
A today's boxes + ViTPose, B RF-DETR Seg masks + ViTPose with the other bodies
greyed out, C RTMO bottom-up.

pick: 16 shots that aren't live, at least 8 s long, with two people in 5 to 60% of
their frames (so mostly one merged box, not an empty or crowd shot), one per fight,
seed 0; 6 frames per shot, evenly spread.
launch: download each video (checked to be the one processed), save those frames
as JPEGs, delete the video, upload the frames as a private dataset, push the job.
import: pull the drawings and counts, delete the dataset.
sheet: one row per frame, the three methods' drawings side by side, cropped to the
people in that frame, in a random order per row labelled X Y Z; the key (which
letter is which method) is written to key.csv and read only by `score`.
A tile is judged 2 (both fighters have a clean skeleton: limbs on the right body,
nothing joined across the two), 1 (one fighter clean), 0 (neither) or na (the frame
isn't two fighters tangled: a break, a close-up, a referee in front).
"""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cornerman.contracts.perception import population_fights
from cornerman.remote import batch, kaggle, videos

DERIVED = Path("data/derived")
OUT = Path("data/pose_test")
KERNEL = Path("kaggle/pose_test")
BUILD = Path("kaggle/build")
SLUG = "cornerman-pose-test"
N_SHOTS, N_FRAMES, SEED = 16, 6, 0
TILE_W = 420


def pick() -> None:
    rows = []
    for fid in population_fights(DERIVED):
        live = pd.read_parquet(DERIVED / fid / "tracking" / "live.parquet")
        live["fight_id"] = fid
        rows.append(live)
    shots = pd.concat(rows)
    pool = shots[
        ~shots["live"]
        & (shots["end"] - shots["start"] >= 8)
        & shots["two_people_frac"].between(0.05, 0.6)
    ]
    one = pool.groupby("fight_id").sample(1, random_state=SEED)
    chosen = one.sample(min(N_SHOTS, len(one)), random_state=SEED)
    out = []
    for r in chosen.itertuples():
        frames = pd.read_parquet(DERIVED / str(r.fight_id) / "tracking" / "frames.parquet")
        f = frames[frames["shot_id"] == r.shot_id]["frame"].to_numpy()
        idx = np.linspace(0, len(f) - 1, N_FRAMES + 2)[1:-1].round().astype(int)
        out += [{"fight_id": r.fight_id, "shot_id": r.shot_id, "frame": int(f[i])} for i in idx]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(OUT / "frames.csv", index=False)
    print(f"{len(chosen)} shots from {len(pool)} candidates ({pool.fight_id.nunique()} fights)")


def launch() -> None:
    picked = pd.read_csv(OUT / "frames.csv")
    urls = pd.read_csv("data/fights.csv").set_index("fight_id")["url"]
    user = str(kaggle.read_kernel_metadata(KERNEL)["id"]).split("/")[0]
    ref = f"{user}/{SLUG}"
    folder = Path("data/raw/upload") / SLUG
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    try:
        for fid, g in picked.groupby("fight_id"):
            video = Path("data/raw/pose_test") / f"{fid}.mp4"
            try:
                videos.download(str(urls[fid]), video, 720)
                info = json.loads((DERIVED / str(fid) / "ingest" / "video.json").read_text())
                if videos.matches_processed(video, info) is None:
                    print(fid, "different video, skipped")
                    continue
                cap = cv2.VideoCapture(str(video))
                for f in sorted(g["frame"]):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                    ok, img = cap.read()
                    if ok:
                        cv2.imwrite(str(folder / f"{fid}__{f:07d}.jpg"), img)
                cap.release()
                print(fid, "ok", flush=True)
            finally:
                video.unlink(missing_ok=True)
        meta = {"title": SLUG, "id": ref, "licenses": [{"name": "unknown"}]}
        (folder / kaggle.DATASET_METADATA).write_text(json.dumps(meta, indent=2))
        kaggle.upload_dataset(folder, ref)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    values = {"REPO": batch.REPO, "COMMIT": batch.pushed_commit()}
    built = kaggle.build_kernel(KERNEL, BUILD / "pose_test", values, {"dataset_sources": [ref]})
    print(kaggle.push_kernel(built))


def do_import() -> None:
    meta = kaggle.read_kernel_metadata(KERNEL)
    kaggle.pull_output(str(meta["id"]), OUT / "output")
    res = json.loads((OUT / "output" / "results.json").read_text())
    errors = sum("error" in r[m] for r in res for m in "ABC")
    if errors:
        print(f"{errors} method runs failed; dataset kept for a rerun")
        return
    user = str(meta["id"]).split("/")[0]
    kaggle.delete_dataset(f"{user}/{SLUG}")
    print(f"pulled to {OUT / 'output'}, deleted {user}/{SLUG}")


def _crop(fid: str, frame: int, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """The people in the frame (tracking boxes, all of them), with a margin."""
    t = pd.read_parquet(
        DERIVED / fid / "tracking" / "tracks.parquet", columns=["frame", "x1", "y1", "x2", "y2"]
    )
    b = t[t["frame"] == frame]
    h, w = shape[:2]
    if b.empty:
        return 0, 0, w, h
    x1, y1, x2, y2 = b.x1.min(), b.y1.min(), b.x2.max(), b.y2.max()
    m = 0.25 * max(x2 - x1, y2 - y1)
    return int(max(0, x1 - m)), int(max(0, y1 - m)), int(min(w, x2 + m)), int(min(h, y2 + m))


def sheet() -> None:
    picked = pd.read_csv(OUT / "frames.csv")
    draw = OUT / "output" / "draw"
    rng = np.random.default_rng(SEED)
    key, rows = [], []
    for n, r in enumerate(picked.itertuples(), start=1):
        name = f"{r.fight_id}__{r.frame:07d}.jpg"
        imgs = {m: cv2.imread(str(draw / m / name)) for m in "ABC"}
        if any(v is None for v in imgs.values()):
            continue
        x1, y1, x2, y2 = _crop(str(r.fight_id), int(r.frame), imgs["A"].shape)
        order = rng.permutation(list("ABC"))
        tiles = []
        for letter, m in zip("XYZ", order, strict=True):
            key.append({"n": n, "tile": letter, "method": m, "fight_id": r.fight_id,
                        "shot_id": r.shot_id, "frame": r.frame})  # fmt: skip
            c = imgs[m][y1:y2, x1:x2]
            c = cv2.resize(c, (TILE_W, int(TILE_W * c.shape[0] / c.shape[1])))
            cv2.putText(c, f"{n}{letter}", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            tiles.append(c)
        hmax = max(t.shape[0] for t in tiles)
        tiles = [
            cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles
        ]
        rows.append(np.hstack(tiles))
    pages = OUT / "sheets"
    shutil.rmtree(pages, ignore_errors=True)
    pages.mkdir()
    for k in range(0, len(rows), 4):
        page = rows[k : k + 4]
        wmax = max(p.shape[1] for p in page)
        page = [
            cv2.copyMakeBorder(p, 0, 6, 0, wmax - p.shape[1], cv2.BORDER_CONSTANT) for p in page
        ]
        cv2.imwrite(str(pages / f"page_{k // 4:02d}.jpg"), np.vstack(page))
    pd.DataFrame(key).to_csv(OUT / "key.csv", index=False)
    print(f"{len(rows)} rows on {(len(rows) + 3) // 4} pages in {pages}")


def score(verdicts: Path) -> None:
    v = pd.read_csv(verdicts, dtype={"verdict": str}).merge(
        pd.read_csv(OUT / "key.csv"), on=["n", "tile"]
    )
    na = v.groupby("n")["verdict"].apply(lambda s: (s == "na").any())
    v = v[~v["n"].map(na)]
    print(
        f"{v.n.nunique()} judgeable frames from {v.groupby(['fight_id', 'shot_id']).ngroups} shots"
    )
    for m, g in v.groupby("method"):
        two = (g["verdict"] == "2").mean()
        one = (g["verdict"] == "1").mean()
        per_shot = g.assign(two=g["verdict"] == "2").groupby(["fight_id", "shot_id"])["two"].mean()
        print(f"{m}: two clean {two:.0%}, one {one:.0%}, none {1 - two - one:.0%}; "
              f"shots with two clean in at least half their frames: "
              f"{(per_shot >= 0.5).sum()} of {len(per_shot)}")  # fmt: skip
    res = json.loads((OUT / "output" / "results.json").read_text())
    counts = pd.DataFrame(
        [{"method": m, "n_10kp": r[m].get("n_10kp")} for r in res for m in "ABC" if m in r]
    )
    print("people with 10+ keypoints per frame (all frames), mean:")
    print(counts.groupby("method")["n_10kp"].mean().round(2).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pick":
        pick()
    elif cmd == "launch":
        launch()
    elif cmd == "import":
        do_import()
    elif cmd == "sheet":
        sheet()
    else:
        score(Path(sys.argv[2]))
