"""Labeller B: an open-weight video VLM labelling silver-set clips (Phase 2).

Each clip has the red-corner fighter in a red box and the blue-corner fighter in
a blue box (gold.cut). The VLM is asked about those two people with two
different prompts, and a clip goes into the silver set only where both prompts
and the rules labeller agree:

- direct: pick each fighter's state from the taxonomy, given short definitions;
- questions: answer plain questions about the clip (standing or ground, who is
  on top, clinched and who controls it, who is backing whom onto the fence,
  hurt). The states are worked out from the answers here, in code.

Taxonomy readings (Ansh, 2026-10-20): a clinch is a clinch wherever it happens,
against the fence too; cage pressure is only when the fighters are not locked
up, at striking range, with one fighter's back near the fence. Fence position is
measured separately from geometry later. A fighter standing over a downed
opponent is ground-top.

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
    State.STANDING_NEUTRAL: "on the feet, not clinched, and neither fighter has their "
    "back to the fence",
    State.STANDING_CAGE_PRESSURING: "on the feet and not clinched, at striking range, "
    "with the opponent backed up so their back is near the fence",
    State.STANDING_CAGE_PRESSURED: "on the feet and not clinched, at striking range, "
    "backed up with their own back near the fence",
    State.STANDING_OPPONENT_HURT: "on the feet, and the opponent is visibly hurt "
    "(stumbling, buckled legs, covering up after being hit)",
    State.CLINCH_CONTROLLING: "standing, upper bodies locked together (in open space "
    "or against the fence), and this fighter has the better grips or position",
    State.CLINCH_CONTROLLED: "standing, upper bodies locked together (in open space "
    "or against the fence), and the opponent has the better grips or position",
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
    "6. fence: if standing and not clinched, is one fighter backing the other up so "
    "that the other's back is near the fence? Name the fighter who is doing the "
    "backing up: red / blue / neither\n"
    "7. hurt: is one fighter visibly hurt (stumbling, buckled legs, covering up "
    "after being hit)? red / blue / neither\n"
    "Answer with JSON only, in this form:\n"
    '{"live": "...", "level": "...", "top": "...", "clinch": "...", '
    '"clinch_control": "...", "fence": "...", "hurt": "..."}',
}


def swap_colours(text: str) -> str:
    """Swap every red/RED with blue/BLUE. In the questions prompt each colour is
    named symmetrically ("a RED box and the other with a BLUE box", "red / blue /
    even"), so this only flips which corner comes first."""
    table = {"red": "blue", "blue": "red", "RED": "BLUE", "BLUE": "RED"}
    return re.sub(r"\b(red|blue|RED|BLUE)\b", lambda m: table[m.group(1)], text)


# The questions prompt with blue named first everywhere. The direct prompt said
# red was controlling in every clinch (114 to 0), so clinch control is only
# trusted where the questions answer survives this swap. Answers still name the
# corners by colour, so parse_questions reads both.
PROMPTS["questions_swapped"] = swap_colours(PROMPTS["questions"])

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

    A clinch is a clinch wherever it happens; cage pressure only applies when the
    fighters are not clinched. An even clinch has no state in the taxonomy (one
    fighter must be controlling), so it gives None rather than a guess.
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
        control, pusher, hurt = a.get("clinch_control"), a.get("fence"), a.get("hurt")
        # The clinch comes first: a clinch against the fence is still a clinch.
        if a.get("clinch") == "yes":
            if control not in ("red", "blue"):
                return None
            s[control], s[_other(control)] = State.CLINCH_CONTROLLING, State.CLINCH_CONTROLLED
        elif pusher in ("red", "blue"):
            pinned = _other(pusher)
            s[pinned] = State.STANDING_CAGE_PRESSURED
            s[pusher] = (
                State.STANDING_OPPONENT_HURT if hurt == pinned else State.STANDING_CAGE_PRESSURING
            )
        elif hurt in ("red", "blue"):
            s[hurt], s[_other(hurt)] = State.STANDING_NEUTRAL, State.STANDING_OPPONENT_HURT
        else:
            s["red"] = s["blue"] = State.STANDING_NEUTRAL
    else:
        return None
    return _pair(s.get("red"), s.get("blue"))


# Actions (Phase 3, Step 6). One question per action of the taxonomy, asked about
# each boxed fighter, with flat JSON keys (extract_json reads flat objects only).
# Strikes are counted; the rest are yes / no. "actions_swapped" names blue first
# everywhere, including the order of the keys, since the clinch questions showed the
# model leans to the fighter named first.
ACTION_QUESTIONS: dict[str, str] = {
    "strikes": "how many punches, kicks, knees or elbows does this fighter throw, "
    "landed or not? 0 / 1 / 2 / 3+",
    "takedown": "does this fighter shoot for or attempt a takedown (driving in to take "
    "the opponent down, a trip or a throw), whether it works or not? yes / no",
    "clinch": "does this fighter start a clinch (grabbing the opponent's head, neck or "
    "body from open space) or break out of one? yes / no",
    "pass": "on the ground and on top, does this fighter pass the guard or move to a "
    "better position (side control, mount, the back)? yes / no",
    "submission": "does this fighter attempt a submission (a choke or a joint lock)? yes / no",
    "sweep": "on the ground and underneath, does this fighter sweep or reverse the "
    "opponent and end up on top? yes / no",
    "stand_up": "does this fighter get back up to the feet from the ground? yes / no",
    "reset": "does this fighter move away or circle out to get off the fence or back "
    "to open space? yes / no",
}
ACTION_KEYS = tuple(ACTION_QUESTIONS)


def _action_prompt(first: Corner) -> str:
    second = _other(first)
    clip = _CLIP if first == "red" else swap_colours(_CLIP)
    lines = [
        clip,
        "\n\nFirst: is this live fighting between the two boxed fighters? (no for "
        "walk-outs, introductions, close-ups of one person, replays, the referee "
        "stopping the fight, after the fight) yes / no",
        f"\nThen answer these questions for the {first.upper()} fighter and for the "
        f"{second.upper()} fighter, about what each does during the clip:",
    ]
    lines += [f"- {k}: {q}" for k, q in ACTION_QUESTIONS.items()]
    keys = ", ".join(f'"{c}_{k}": "..."' for c in (first, second) for k in ACTION_KEYS)
    lines.append('\nAnswer with JSON only, in this form:\n{"live": "...", ' + keys + "}")
    return "\n".join(lines)


PROMPTS["actions"] = _action_prompt("red")
PROMPTS["actions_swapped"] = _action_prompt("blue")


def _yes(value: str | None) -> bool | None:
    return {"yes": True, "no": False}.get(value or "")


def _strikes(value: str | None) -> int | None:
    m = re.fullmatch(r"\s*(\d+)\s*\+?\s*", value or "")
    return min(int(m.group(1)), 3) if m else None


def parse_actions(text: str) -> Labels | None:
    """Labels from an actions answer: per corner, the actions it took as a string like
    "strikes=2;takedown;clinch" (strikes capped at 3, "3" meaning 3 or more), or None
    if the answer can't be read in full. Every key must be answered for both corners,
    since a half answer can't tell "no" from "not asked"."""
    a = extract_json(text)
    if a is None:
        return None
    if a.get("live") == "no":
        return dict(_NOT_LIVE)
    if a.get("live") != "yes":
        return None
    out: Labels = {"live": "yes"}
    for c in ("red", "blue"):
        n = _strikes(a.get(f"{c}_strikes"))
        flags = [_yes(a.get(f"{c}_{k}")) for k in ACTION_KEYS[1:]]
        if n is None or any(f is None for f in flags):
            return None
        done = [k for k, f in zip(ACTION_KEYS[1:], flags, strict=True) if f]
        out[c] = ";".join([f"strikes={n}", *done])
    return out


def actions_of(value: str | None) -> dict[str, int]:
    """The inverse of parse_actions for one corner: {"strikes": n, "takedown": 1, ...}
    with every action key present (0 where not taken)."""
    out = dict.fromkeys(ACTION_KEYS, 0)
    for part in (value or "").split(";"):
        if part.startswith("strikes="):
            out["strikes"] = int(part.split("=")[1])
        elif part in out:
            out[part] = 1
    return out
