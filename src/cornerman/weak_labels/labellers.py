"""The labellers, the pair states they label, and how their votes become soft labels.

The class is the pair's state from red's side, so the two fighters' states are
compatible by construction (`fighter_states`):

    neutral, red-pressuring, blue-pressuring, red-clinch (red controls),
    blue-clinch, red-top, blue-top, scramble.

Standing-opponent-hurt and the momentum flag are not produced: they need strike
and hurt detection (Tier 2).

Labellers (see events/moments.py for how A, M and G are computed):
- R, the rules: A (pose and boxes, where the fighters are apart) and M (masks,
  where they touch) are one labeller. They are two halves of one rule set that
  share the same contact test, so their errors are not independent and Dawid-Skene
  must not count them as two witnesses.
- G, fence distance: neutral, red-pressuring, blue-pressuring. Speaks only where R
  says standing apart, so it carries no independent evidence about the family.
- B, the VLM (Qwen3-VL-4B) on the 780 silver-set clips. Its prompts are the same
  model on the same clip, so their errors are correlated: they enter as one
  labeller, what the prompts agree on (`vlm_label`). Not a cage labeller: its cage
  answers flip with the colour order 46% of the time.

Why two levels (DECISIONS.md, 2026-10-27). Dawid-Skene over the 8 pair states did
not settle on the real moments: free and constrained versions both drifted to
answers where classes lost their meaning. The labellers barely overlap (R and B
meet only on the 780 clips; G only where R says standing apart), so the 8-class
model is not identifiable, the same failure the recovery tests pin down on
simulated data. So:
- the family (standing, clinch, ground) comes from Dawid-Skene EM over R and B
  (`fit_family`). Two labellers don't identify it on their own, so B's family
  accuracy is fixed from its spot check (the anchor) and EM learns R's;
- the direction within the family (who presses, who controls, who is on top)
  comes from Dawid-Skene's E step with confusions fixed from measurements, since
  only one or two labellers ever speak to it (DIRECTION_TRUST).
The soft label is P(family) x P(direction | family).
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cornerman.taxonomy import State
from cornerman.weak_labels import dawid_skene as ds

PAIR_STATES = (
    "neutral",
    "red-pressuring",
    "blue-pressuring",
    "red-clinch",
    "blue-clinch",
    "red-top",
    "blue-top",
    "scramble",
)
FAMILIES = ("standing", "clinch", "ground")
FAMILY = {
    "neutral": "standing",
    "red-pressuring": "standing",
    "blue-pressuring": "standing",
    "red-clinch": "clinch",
    "blue-clinch": "clinch",
    "red-top": "ground",
    "blue-top": "ground",
    "scramble": "ground",
}
DIRECTIONS = {
    "standing": ("neutral", "red-pressuring", "blue-pressuring"),
    "clinch": ("red-clinch", "blue-clinch"),
    "ground": ("red-top", "blue-top", "scramble"),
}

R_FAMILY = {
    "standing": "standing",
    "clinch": "clinch",
    "ground-red-top": "ground",
    "ground-blue-top": "ground",
    "ground-even": "ground",
    "apart": "standing",
    "standing-contact": "clinch",
    "ground-top-unknown": "ground",
}
B_FAMILY = {
    "standing": "standing",
    "clinch": "clinch",
    "clinch-red": "clinch",
    "clinch-blue": "clinch",
    "ground-red-top": "ground",
    "ground-blue-top": "ground",
    "scramble": "ground",
}

DIRECTION_TRUST = {
    "G-called": 0.5,
    "G-neutral": 0.84,
    "R-top": 0.95,
    "B-top": 0.85,
    "B-controller": 0.75,
    "even": 0.5,
}
"""How far each directional vote is trusted, and where the number comes from:
- G-called 0.5: the fence rule's calls were right 10 of 20 in its by-eye check;
  the other 10 were neutral, none named the wrong fighter;
- G-neutral 0.84: 16 of 19 hard negatives (a fighter near a fence, no call) were
  neutral by eye;
- R-top 0.95: on ViCoS, where the rules name a top fighter they are right 82 of 84
  (pose) and 86 of 87 (masks); lowered for broadcast footage;
- B-top 0.85: silver ground direction 6 of 6 by eye, lower tier 4 of 6;
- B-controller 0.75: an assumption, not a measurement. The swap test keeps only
  controllers named the same way with the colours swapped, but the spot check could
  confirm 1 of 5 silver clinch controllers (1 plausible, 3 can't be told, 0 wrong);
- even 0.5: "nobody clearly on top" means scramble with this probability, else
  either fighter on top.
Agent-judged checks are sanity checks, so these are rough."""


def fighter_states(pair: str) -> tuple[State, State]:
    """(red's state, blue's state) for a pair state."""
    return {
        "neutral": (State.STANDING_NEUTRAL, State.STANDING_NEUTRAL),
        "red-pressuring": (State.STANDING_CAGE_PRESSURING, State.STANDING_CAGE_PRESSURED),
        "blue-pressuring": (State.STANDING_CAGE_PRESSURED, State.STANDING_CAGE_PRESSURING),
        "red-clinch": (State.CLINCH_CONTROLLING, State.CLINCH_CONTROLLED),
        "blue-clinch": (State.CLINCH_CONTROLLED, State.CLINCH_CONTROLLING),
        "red-top": (State.GROUND_TOP, State.GROUND_BOTTOM),
        "blue-top": (State.GROUND_BOTTOM, State.GROUND_TOP),
        "scramble": (State.GROUND_SCRAMBLING, State.GROUND_SCRAMBLING),
    }[pair]


def pair_of(red: object, blue: object) -> str | None:
    """The pair state for (red's, blue's) taxonomy states, or None if they don't make one.
    Standing-opponent-hurt counts as neutral (hurt isn't modelled yet)."""
    if not isinstance(red, str) or not isinstance(blue, str):
        return None
    for pair in PAIR_STATES:
        if fighter_states(pair) == (State(red), State(blue)):
            return pair
    standing = {State.STANDING_NEUTRAL, State.STANDING_OPPONENT_HURT}
    if State(red) in standing and State(blue) in standing:
        return "neutral"
    return None


def _family(red: object, blue: object) -> str | None:
    if not isinstance(red, str) or not isinstance(blue, str):
        return None
    fams = {s.split("-")[0] for s in (red, blue)}
    return fams.pop() if len(fams) == 1 else None


def vlm_label(row: Mapping[Any, object]) -> str | None:
    """Labeller B for one clip (a row of the silver table): what the prompts agree on.

    Both prompts must say the clip is live and agree on the state family. Standing
    (neutral, cage or hurt) is just "standing". A clinch has a controller only if the
    questions prompt and its colour-swapped run name the same one (the direct prompt
    called red in all 114 clinches, so it never picks the controller). On the ground the
    two prompts must give the same pair.
    """
    if row.get("live_direct") != "yes" or row.get("live_questions") != "yes":
        return None
    fam = _family(row.get("red_direct"), row.get("blue_direct"))
    if fam is None or fam != _family(row.get("red_questions"), row.get("blue_questions")):
        return None
    if fam == "standing":
        return "standing"
    if fam == "clinch":
        q = pair_of(row.get("red_questions"), row.get("blue_questions"))
        s = pair_of(row.get("red_questions_swapped"), row.get("blue_questions_swapped"))
        if row.get("live_questions_swapped") == "yes" and q is not None and q == s:
            return "clinch-red" if q == "red-clinch" else "clinch-blue"
        return "clinch"
    d = pair_of(row.get("red_direct"), row.get("blue_direct"))
    q = pair_of(row.get("red_questions"), row.get("blue_questions"))
    if d is None or d != q:
        return None
    return {"red-top": "ground-red-top", "blue-top": "ground-blue-top"}.get(d, "scramble")


def attach_vlm(moments: pd.DataFrame, silver: pd.DataFrame, clips: pd.DataFrame) -> pd.DataFrame:
    """Moments with labeller B (column "B") and the clip each falls in ("clip_id"),
    from the silver table (gold/silver.py) and the clips' times."""
    s = silver.merge(clips[["clip_id", "t0", "t1"]], on="clip_id")
    labels = [vlm_label(r) for r in s.to_dict("records")]
    out = moments.copy()
    out["B"] = None
    out["clip_id"] = None
    for fid, cid, t0, t1, b in zip(s["fight_id"], s["clip_id"], s["t0"], s["t1"], labels,
                                   strict=True):  # fmt: skip
        inside = (out["fight_id"] == fid) & (out["t"] >= float(t0)) & (out["t"] < float(t1))
        out.loc[inside, "clip_id"] = cid
        out.loc[inside, "B"] = b
    return out


def rules_label(a: object, m: object) -> str | None:
    """Labeller R: the pose rules where they speak, else the mask rules."""
    if isinstance(a, str):
        return a
    return m if isinstance(m, str) else None


def family_votes(moments: pd.DataFrame) -> np.ndarray:
    """(n, 2) family indices from R and B, ABSTAIN where silent."""
    out = np.full((len(moments), 2), ds.ABSTAIN)
    r = [rules_label(a, m) for a, m in zip(moments["A"], moments["M"], strict=True)]
    for j, (labels, fmap) in enumerate(((r, R_FAMILY), (moments["B"].tolist(), B_FAMILY))):
        out[:, j] = [FAMILIES.index(fmap[x]) if isinstance(x, str) else ds.ABSTAIN
                     for x in labels]  # fmt: skip
    return out


VLM_FAMILY_ACCURACY = 0.95
"""The anchor: the VLM's state family was right in 32 of 32 spot-checked clips (16 silver,
16 lower tier; agent-judged, DECISIONS.md silver second build). 0.95 rather than 1 since
a small check can't show it never errs."""


def fit_family(votes: np.ndarray) -> ds.DSResult:
    """Dawid-Skene over the family votes of R (column 0) and B (column 1).

    Two labellers don't identify free 3 x 3 confusions: on the real moments, free and
    constrained fits gave opposite answers (one erased clinch, the other ground). So B
    is the anchor: its confusion is fixed at VLM_FAMILY_ACCURACY, errors spread evenly,
    and EM learns R's confusion from the moments where both vote, the only ones that
    carry information about it. The class prior is then refitted on every voted moment
    with the confusions fixed, since the clips were sampled to over-represent clinch
    and ground.
    """
    n = len(FAMILIES)
    eye = np.eye(n)
    anchor = VLM_FAMILY_ACCURACY * eye + (1 - VLM_FAMILY_ACCURACY) / (n - 1) * (1 - eye)
    both = (votes != ds.ABSTAIN).all(axis=1)
    r = ds.fit(votes[both], [ds.compatible_init(eye), anchor], fixed=[1])
    prior = ds.fit_prior(votes, r.confusion, r.prior)
    voted = (votes != ds.ABSTAIN).any(axis=1)
    post = np.tile(prior, (len(votes), 1))
    post[voted] = ds.posterior(votes[voted], r.confusion, prior)
    return ds.DSResult(posterior=post, prior=prior, confusion=r.confusion, loglik=r.loglik,
                       voted=voted)  # fmt: skip


def _vote(dist: dict[str, float], toward: str, trust: float, options: tuple[str, ...]) -> None:
    """Multiply in one directional vote: `trust` on `toward`, the rest spread evenly."""
    rest = (1 - trust) / (len(options) - 1)
    for o in options:
        dist[o] *= trust if o == toward else rest


def direction(row: Mapping[Any, object]) -> dict[str, dict[str, float]]:
    """P(direction | family) for each family from one moment's directional votes,
    treating votes as independent given the truth (the E step with fixed confusions)."""
    t = DIRECTION_TRUST
    out = {f: {d: 1.0 for d in opts} for f, opts in DIRECTIONS.items()}
    st, cl, gr = DIRECTIONS["standing"], DIRECTIONS["clinch"], DIRECTIONS["ground"]
    g = row.get("G")
    if g == "neutral":
        _vote(out["standing"], "neutral", t["G-neutral"], st)
    elif g in ("red-pressuring", "blue-pressuring"):
        # Right or neutral in the check, never the other fighter (kept a little above 0).
        other = "red-pressuring" if g == "blue-pressuring" else "blue-pressuring"
        out["standing"][str(g)] *= t["G-called"]
        out["standing"]["neutral"] *= 1 - t["G-called"]
        out["standing"][other] *= 0.02
    b = row.get("B")
    if b in ("clinch-red", "clinch-blue"):
        _vote(out["clinch"], "red-clinch" if b == "clinch-red" else "blue-clinch",
              t["B-controller"], cl)  # fmt: skip
    rl = rules_label(row.get("A"), row.get("M"))
    for vote, trust in ((rl, t["R-top"]), (b, t["B-top"])):
        if vote == "ground-red-top":
            _vote(out["ground"], "red-top", trust, gr)
        elif vote == "ground-blue-top":
            _vote(out["ground"], "blue-top", trust, gr)
        elif vote in ("ground-even", "scramble"):
            _vote(out["ground"], "scramble", t["even"] if vote == "ground-even" else trust, gr)
    for f in out:
        total = sum(out[f].values())
        out[f] = {d: v / total for d, v in out[f].items()}
    return out


def soft_labels(moments: pd.DataFrame, family: np.ndarray) -> np.ndarray:
    """(n, 8) P(pair state) = P(family) x P(direction | family), per moment."""
    out = np.zeros((len(moments), len(PAIR_STATES)))
    cols = [c for c in ("A", "M", "G", "B") if c in moments]
    for i, row in enumerate(moments[cols].to_dict("records")):
        d = direction(row)
        for k, pair in enumerate(PAIR_STATES):
            f = FAMILY[pair]
            out[i, k] = family[i, FAMILIES.index(f)] * d[f][pair]
    return out
