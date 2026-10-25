"""Fights on hold: kept on disk but left out of everything until cleared.

A fight goes on hold when we can't confirm it is an adult bout: its on-screen
graphics never show the division (older IMMAF events), and nothing else says
which division it was. Held fights are excluded from modeling, silver sets,
reports and new GPU batches, but not deleted, so they can come back if the
event's results show it had no junior or youth division. Ones that stay unclear
are deleted (DECISIONS.md, "Adult bouts, part 3").

The list is data/age_hold.csv (local, like fights.csv): fight_id, reason.
"""

from pathlib import Path

import pandas as pd

HOLD_CSV = Path("data/age_hold.csv")


def held_fights(path: Path = HOLD_CSV) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["fight_id"].astype(str))
