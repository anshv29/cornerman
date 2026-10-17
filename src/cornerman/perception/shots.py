"""Shot boundaries from TransNetV2's per-frame cut probabilities.

TransNetV2 scores every frame for "a cut or transition happens here". We turn
runs of frames above the threshold into boundaries and everything between them
into shots. The model only needs tiny 48x27 frames, so this pass is cheap.
"""

import numpy as np
import pandas as pd

TRANSNET_SIZE = (48, 27)


def shots_from_probs(probs: np.ndarray, fps: float, threshold: float) -> pd.DataFrame:
    """One row per shot: shot_id, first and last frame (inclusive), start and end in seconds.

    Frames inside a transition (probability above the threshold) belong to no shot.
    """
    is_cut = probs > threshold
    shots: list[tuple[int, int]] = []
    start: int | None = None
    for i, cut in enumerate(is_cut):
        if not cut and start is None:
            start = i
        elif cut and start is not None:
            shots.append((start, i - 1))
            start = None
    if start is not None:
        shots.append((start, len(probs) - 1))
    first = np.array([s for s, _ in shots], dtype=np.int64)
    last = np.array([e for _, e in shots], dtype=np.int64)
    return pd.DataFrame(
        {
            "shot_id": np.arange(len(shots), dtype=np.int64),
            "first_frame": first,
            "last_frame": last,
            "start": first / fps,
            "end": (last + 1) / fps,
        }
    )


def shot_of_frame(shots: pd.DataFrame, frames: np.ndarray) -> np.ndarray:
    """Shot ID for each frame index, or -1 for frames inside a transition."""
    first = shots["first_frame"].to_numpy()
    last = shots["last_frame"].to_numpy()
    i = np.searchsorted(first, frames, side="right") - 1
    inside = (i >= 0) & (frames <= last[np.clip(i, 0, None)])
    return np.where(inside, shots["shot_id"].to_numpy()[np.clip(i, 0, None)], -1)
