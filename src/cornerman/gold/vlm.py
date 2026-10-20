"""Labeller B: an open-weight video VLM labelling silver-set clips (Phase 2).

Each clip has the red-corner fighter in a red box and the blue-corner fighter in
a blue box (gold.cut). The VLM is asked about those two people with two
different prompts, and a clip goes into the silver set only where both prompts
and the rules labeller agree:

- direct: pick each fighter's state from the taxonomy, given short definitions;
- questions: answer plain questions about the clip (standing or ground, who is
  on top, clinched and who controls it, pinned on the fence, hurt). The states
  are worked out from the answers here, in code.

The two prompts reach a state by different routes, so agreement between them
means more than asking the same thing twice. Both may answer that the clip isn't
live fighting (walk-outs, close-ups, replays, the end of the fight), and such
clips are left out.

This module only builds prompts and reads answers; it never calls a hosted or
paid model (hard rule 9). The model runs in the private Kaggle job
(kaggle/vlm/job.py), which installs this package at the pushed commit.
"""

import json
import re
from typing import Literal

from cornerman.taxonomy import State, compatible

Corner = Literal["red", "blue"]
Labels = dict[str, str | None]
"""{"live": "yes"/"no", "red": state or None, "blue": state or None}."""

_CLIP = (
    "This is a <SECONDS>-second clip from an amateur MMA fight. One fighter is "
    "marked with a RED box and the other with a BLUE box (the boxes are drawn on the "
    "video; they are not the fighters' clothing). Judge the clip as a whole, by what "
    "happens for most of it."
)

STATE_DEFINITIONS = {
    State.STANDING_NEUTRAL: "on the feet, not clinched, not pinned on the fence",
    State.STANDING_CAGE_PRESSURING: "on the feet, pushing or holding the opponent against "
    "the fence (including clinching them against the fence)",
    State.STANDING_CAGE_PRESSURED: "on the feet, pinned against the fence by the opponent "
    "(including being clinched against the fence)",
    State.STANDING_OPPONENT_HURT: "on the feet, and the opponent is visibly hurt "
    "(stumbling, buckled legs, covering up after being hit)",
    State.CLINCH_CONTROLLING: "standing, upper bodies locked together away from the "
    "fence, and this fighter has the better grips or position",
    State.CLINCH_CONTROLLED: "standing, upper bodies locked together away from the "
    "fence, and the opponent has the better grips or position",
    State.GROUND_TOP: "on top of a downed opponent: inside their guard, side "
    "control, mount, on their back, or standing over an opponent who is down on the mat",
    State.GROUND_BOTTOM: "down on the mat with the opponent on top or standing over "
    "them (including holding guard)",
    State.GROUND_SCRAMBLING: "on the ground and nobody is on top: both are moving to "
    "get up or to get position",
}

PROMPTS: dict[str, str] = {
    "direct": _CLIP
    + "\n\nGive each boxed fighter's state, from these options:\n"
    + "\n".join(f"- {s.value}: {d}" for s, d in STATE_DEFINITIONS.items())
    + "\n\nIf the clip is not live fighting (walk-out, introduction, a close-up of one "
    "person, a replay, the referee stopping the fight, after the fight), or a boxed "
    'fighter can\'t be seen, answer "live": "no".\n'
    "Answer with JSON only, in this form:\n"
    '{"live": "yes" or "no", "red": "<state>", "blue": "<state>"}',
    "questions": _CLIP + "\n\nAnswer these questions about the two boxed fighters.\n"
    "1. live: is this live fighting between the two boxed fighters? (no for "
    "walk-outs, introductions, close-ups of one person, replays, the referee "
    "stopping the fight, after the fight) yes / no\n"
    "2. level: are both fighters on their feet, or is the fight on the ground (at "
    "least one fighter down on the mat, with the other on them or standing over "
    "them)? standing / ground\n"
    "3. top: if on the ground, which fighter is on top (or standing over the other)? "
    "red / blue / neither\n"
    "4. clinch: if standing, are their upper bodies locked together (grips on the "
    "head, neck, arms or body)? yes / no\n"
    "5. clinch_control: if clinched, who has the better grips or position? red / "
    "blue / even\n"
    "6. fence: if standing, is one fighter pinned against the fence by the other? "
    "red / blue / neither (name the fighter who is pinned)\n"
    "7. hurt: is one fighter visibly hurt (stumbling, buckled legs, covering up "
    "after being hit)? red / blue / neither\n"
    "Answer with JSON only, in this form:\n"
    '{"live": "...", "level": "...", "top": "...", "clinch": "...", '
    '"clinch_control": "...", "fence": "...", "hurt": "..."}',
}

_NOT_LIVE: Labels = {"live": "no", "red": None, "blue": None}


def prompt(name: str, seconds: float = 3.0) -> str:
    return PROMPTS[name].replace("<SECONDS>", f"{seconds:.0f}")


def extract_json(text: str) -> dict[str, str] | None:
    """The last {...} object in a model answer, lower-cased string values; None if none."""
    for match in reversed(re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)):
        try:
            data = json.loads(match)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return {str(k).strip().lower(): str(v).strip().lower() for k, v in data.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    return None


def _state(value: str | None) -> State | None:
    try:
        return State(value) if value is not None else None
    except ValueError:
        return None


def _pair(red: State | None, blue: State | None) -> Labels | None:
    """Labels for a valid, compatible pair, else None (the answer is unusable)."""
    if red is None or blue is None or not compatible(red, blue):
        return None
    return {"live": "yes", "red": red.value, "blue": blue.value}


def parse_direct(text: str) -> Labels | None:
    """Labels from a 'direct' answer, or None if it can't be read or is impossible."""
    a = extract_json(text)
    if a is None:
        return None
    if a.get("live") == "no":
        return dict(_NOT_LIVE)
    return _pair(_state(a.get("red")), _state(a.get("blue")))


def _other(c: str) -> Corner:
    return "blue" if c == "red" else "red"


def parse_questions(text: str) -> Labels | None:
    """Labels worked out from a 'questions' answer, or None if it can't be decided.

    A fighter pinned on the fence makes it cage pressure even in a clinch (the
    clinch states are for clinches away from the fence). An even clinch has no
    state in the taxonomy (one fighter must be controlling), so it gives None
    rather than a guess.
    """
    a = extract_json(text)
    if a is None:
        return None
    if a.get("live") == "no":
        return dict(_NOT_LIVE)
    s: dict[str, State] = {}
    if a.get("level") == "ground":
        top = a.get("top")
        if top in ("red", "blue"):
            s[top], s[_other(top)] = State.GROUND_TOP, State.GROUND_BOTTOM
        else:
            s["red"] = s["blue"] = State.GROUND_SCRAMBLING
    elif a.get("level") == "standing":
        control, pinned, hurt = a.get("clinch_control"), a.get("fence"), a.get("hurt")
        # The fence comes first: a clinch against the fence is cage pressure.
        if pinned in ("red", "blue"):
            s[pinned] = State.STANDING_CAGE_PRESSURED
            s[_other(pinned)] = (
                State.STANDING_OPPONENT_HURT if hurt == pinned else State.STANDING_CAGE_PRESSURING
            )
        elif a.get("clinch") == "yes":
            if control not in ("red", "blue"):
                return None
            s[control], s[_other(control)] = State.CLINCH_CONTROLLING, State.CLINCH_CONTROLLED
        elif hurt in ("red", "blue"):
            s[hurt], s[_other(hurt)] = State.STANDING_NEUTRAL, State.STANDING_OPPONENT_HURT
        else:
            s["red"] = s["blue"] = State.STANDING_NEUTRAL
    else:
        return None
    return _pair(s.get("red"), s.get("blue"))
