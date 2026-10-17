"""Adult bouts only: flag any video that might show a fighter under 18.

The check reads the title and the description, because a bout can be filed
under an adult event name ("2024 World Championships") and only say "Junior
World Champion" in the description. It errs on the side of skipping:
- youth, junior and cadet divisions (junior is skipped even where it means
  18 to 20, since a title can't tell us which definition applies);
- age groups like U16, U-18, U21 or "under 18" (anything up to 21);
- words for children: kids, children, teens, schoolboys and so on;
- an explicit age under 18 ("16 years old", "aged 17").
See DECISIONS.md, "Adult bouts only".
"""

import re

MAX_AGE_GROUP = 21
"""Age groups up to this (U21 and below) may include minors, so they are skipped."""

_WORDS = re.compile(
    r"\b(junior|juniors|youth|cadet|cadets|kids?|children|child|teens?|teenagers?"
    r"|schoolboys?|schoolgirls?|boys|girls|minors?)\b",
    re.IGNORECASE,
)
_AGE_GROUP = re.compile(r"\b(?:u|under)[\s-]?(\d{1,2})s?\b", re.IGNORECASE)
_AGE = re.compile(
    r"\b(\d{1,2})[\s-]?(?:years?|yrs?)[\s-]?old\b|\baged?\s+(\d{1,2})\b", re.IGNORECASE
)


def minor_reasons(text: str) -> list[str]:
    """Everything in the text that suggests a fighter under 18. Empty means adult."""
    reasons = [f"word: {m.group(1).lower()}" for m in _WORDS.finditer(text)]
    for m in _AGE_GROUP.finditer(text):
        if int(m.group(1)) <= MAX_AGE_GROUP:
            reasons.append(f"age group: {m.group(0)}")
    for m in _AGE.finditer(text):
        age = int(m.group(1) or m.group(2))
        if age < 18:
            reasons.append(f"age: {m.group(0)}")
    return sorted(set(reasons))
