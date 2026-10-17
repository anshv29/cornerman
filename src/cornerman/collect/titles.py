"""Parse fight video titles into fighters and event, and normalise names.

Handles titles like
    "First Last vs. Other Name 2 | FULL FIGHT | 2023 Some Championships"
where a trailing number after the second fighter marks a rematch.
"""

import re
import unicodedata
from dataclasses import dataclass

_TITLE = re.compile(
    r"^\s*(?P<a>.+?)\s+vs\.?\s+(?P<b>.+?)(?:\s+(?P<n>[2-9]))?"
    r"\s*\|\s*full fight\s*\|\s*(?P<event>.+?)\s*$",
    re.IGNORECASE,
)
# Letters NFKD doesn't split into a base letter plus an accent.
_LETTERS = str.maketrans(
    {
        "ł": "l",
        "Ł": "L",
        "ø": "o",
        "Ø": "O",
        "đ": "d",
        "Đ": "D",
        "ß": "ss",
        "æ": "ae",
        "Æ": "AE",
        "\u0131": "i",  # dotless i
        "œ": "oe",
    }
)
_YEAR = re.compile(r"^(?P<year>(?:19|20)\d{2})\s+(?P<rest>.+)$")


@dataclass(frozen=True)
class ParsedTitle:
    fighter_a: str
    fighter_b: str
    meeting: int
    """1 for a first meeting, 2 for a rematch titled "... 2", and so on."""
    event: str
    year: int | None


def parse_title(title: str) -> ParsedTitle | None:
    m = _TITLE.match(title)
    if m is None:
        return None
    event = m.group("event")
    y = _YEAR.match(event)
    return ParsedTitle(
        fighter_a=m.group("a").strip(),
        fighter_b=m.group("b").strip(),
        meeting=int(m.group("n") or 1),
        event=event,
        year=int(y.group("year")) if y else None,
    )


def name_key(name: str) -> str:
    """Accent-free, lower-case, punctuation-free form used to match name variants."""
    folded = unicodedata.normalize("NFKD", name.translate(_LETTERS))
    ascii_only = "".join(c for c in folded if not unicodedata.combining(c))
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", ascii_only.casefold())
    return " ".join(cleaned.split())
