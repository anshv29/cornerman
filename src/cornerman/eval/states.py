"""Evaluation of the state classifier (Phase 3), out of fold.

Against silver: each silver clip's state pair is taken as the truth for every moment
of the classifier's output inside the clip (moments are 0.5 s apart, clips 3 s, so
up to 6 per clip). Reported:
- moment-level accuracy over the 8 pair states and over the 3 families, next to the
  80% target in SPEC;
- per taxonomy state, precision, recall and F1, counting each fighter separately
  (a red-top moment is ground-top for red and ground-bottom for blue);
- clinch merged (clinch family) and split (who controls);
- by clip stratum (apart, close, ground, merged);
- baselines (hard rule 3): the majority class, and the rules alone (labeller R's own
  label, mapped to a pair state; clinch without a controller counts as red-clinch).

Silver needs the rules to agree with both VLM prompts, so it leans to easy clips and
flatters anything built on the same features; the rules baseline agrees with silver
by construction at the family level. So also:
- the lower tier (both prompts agree, rules abstain), family only;
- label-free rates on all moments: compatibility violations (zero by construction,
  since the class is a pair state) and impossible transitions (eval/label_free.py).
"""

import numpy as np
import pandas as pd

from cornerman.eval.label_free import impossible_transition_rate
from cornerman.taxonomy import State
from cornerman.weak_labels.labellers import (
    FAMILY,
    PAIR_STATES,
    fighter_states,
    pair_of,
    rules_label,
)

RULES_PAIR = {
    "standing": "neutral",
    "apart": "neutral",
    "clinch": "red-clinch",
    "standing-contact": "red-clinch",
    "ground-red-top": "red-top",
    "ground-blue-top": "blue-top",
    "ground-even": "scramble",
    "ground-top-unknown": "scramble",
}


def truth_moments(pred: pd.DataFrame, silver: pd.DataFrame, tier: str) -> pd.DataFrame:
    """Predicted moments inside clips of a tier ("silver" or "prompts_only"), with the
    clip's pair state ("truth"), family and stratum."""
    s = silver[silver[tier].astype(bool)].copy()
    s["truth"] = [pair_of(r, b) for r, b in zip(s["red_final"], s["blue_final"], strict=True)]
    s = s[s["truth"].notna()]
    out = pred.merge(s[["clip_id", "truth", "stratum"]], on="clip_id")
    out["truth_family"] = out["truth"].map(FAMILY)
    return out


def per_state(truth: pd.Series, pred: pd.Series) -> pd.DataFrame:
    """Precision, recall, F1 and support per taxonomy state, both fighters counted."""
    rows_t: list[str] = []
    rows_p: list[str] = []
    for t, p in zip(truth, pred, strict=True):
        tr, tb = fighter_states(str(t))
        pr, pb = fighter_states(str(p))
        rows_t += [tr.value, tb.value]
        rows_p += [pr.value, pb.value]
    yt, yp = np.array(rows_t), np.array(rows_p)
    out: list[dict[str, object]] = []
    for s in State:
        tp = int(((yt == s.value) & (yp == s.value)).sum())
        n_true, n_pred = int((yt == s.value).sum()), int((yp == s.value).sum())
        prec = tp / n_pred if n_pred else float("nan")
        rec = tp / n_true if n_true else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if n_pred and n_true and prec + rec > 0 else float("nan")
        out.append({"state": s.value, "support": n_true, "predicted": n_pred,
                    "precision": prec, "recall": rec, "f1": f1})  # fmt: skip
    return pd.DataFrame(out).set_index("state").round(2)


def accuracy(df: pd.DataFrame, col: str) -> dict[str, float]:
    fam = df[col].map(FAMILY)
    clinch_t = df["truth_family"] == "clinch"
    return {
        "moments": float(len(df)),
        "pair": float((df[col] == df["truth"]).mean()),
        "family": float((fam == df["truth_family"]).mean()),
        "clinch_merged_recall": float((fam[clinch_t] == "clinch").mean()) if clinch_t.any()
        else float("nan"),
        "clinch_split_recall": float((df.loc[clinch_t, col] == df.loc[clinch_t, "truth"]).mean())
        if clinch_t.any() else float("nan"),
    }  # fmt: skip


def with_baselines(df: pd.DataFrame, majority: str) -> pd.DataFrame:
    out = df.copy()
    out["majority"] = majority
    out["rules"] = [
        RULES_PAIR.get(rules_label(a, m) or "", "none")
        for a, m in zip(out["A"], out["M"], strict=True)
    ]
    return out


def label_free(pred: pd.DataFrame) -> dict[str, float]:
    """Impossible-transition rate over all moments' smoothed states (both fighters)."""
    rates: list[float] = []
    for _, g in pred.groupby("fight_id"):
        red = [fighter_states(p)[0].value for p in g["pred"]]
        blue = [fighter_states(p)[1].value for p in g["pred"]]
        ev = pd.DataFrame(
            {
                "t": np.r_[g["t"].to_numpy(float), g["t"].to_numpy(float)],
                "fighter": ["red"] * len(g) + ["blue"] * len(g),
                "state": red + blue,
                "action": [None] * (2 * len(g)),
            }
        )
        rates.append(impossible_transition_rate(ev))
    return {"impossible_transition_rate_median": float(np.median(rates)),
            "compatibility_violation_rate": 0.0}  # fmt: skip


def report(pred: pd.DataFrame, silver: pd.DataFrame) -> dict[str, object]:
    """Every number in the Phase 3 table, as plain data."""
    counts = pred["pred"].value_counts()
    majority = str(counts.index[0]) if len(counts) else PAIR_STATES[0]
    sil = with_baselines(truth_moments(pred, silver, "silver"), majority)
    low = with_baselines(truth_moments(pred, silver, "prompts_only"), majority)
    out: dict[str, object] = {
        "silver_clips": int(sil["clip_id"].nunique()),
        "silver": {c: accuracy(sil, c) for c in ("pred", "majority", "rules")},
        "silver_by_stratum": {str(k): accuracy(g, "pred") for k, g in sil.groupby("stratum")},
        "silver_per_state": per_state(sil["truth"], sil["pred"]),
        "silver_confusion": pd.crosstab(sil["truth"], sil["pred"]),
        "lower_tier_clips": int(low["clip_id"].nunique()),
        "lower_tier_family": {
            c: float((low[c].map(FAMILY) == low["truth_family"]).mean())
            for c in ("pred", "majority", "rules")
        },
        "rules_coverage_silver": float((sil["rules"] != "none").mean()),
        "label_free": label_free(pred),
        "predicted_share": pred["pred"].value_counts(normalize=True).round(3).to_dict(),
    }
    return out
