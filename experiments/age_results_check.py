"""Check fights on age hold against IMMAF's published results (2026-10-25).

Fights whose score bar never shows a division are on hold (contracts.hold). For
each one we look for both fighters in IMMAF's results pages:
- adult: both fighters in the same bracket or bout whose heading is not a junior
  or youth division, and neither fighter in a junior or youth section that year;
- junior: both fighters together in a junior or youth section that year;
- unclear: anything else (a name not found, only one fighter found, ...).

"That year" means pages of the fight's year or the year before: YouTube event
names don't match IMMAF's (the "2021" World Championships were held in January
2022). Junior bouts in other years don't count against a fight; a fighter who
was a junior in 2019 can fight as an adult in 2022.

A page is junior or youth as a whole only if its key says so (2019je, 2019yw);
on combined pages (the 2021 Senior & Junior European Open) each bout's heading
decides ("Male Jr Flyweight ..."). A name matches only within one line.

Pages are fetched once by hand at IMMAF's robots.txt crawl delay and kept outside
the repo, because they hold real names. Usage:

    uv run python experiments/age_results_check.py <pages_dir> <pages.txt> <out.csv>

pages.txt lists "<key> <url path>"; keys start with the year (2019e, 2019je).
Writes fight_id, verdict, detail (no names) to out.csv.
"""

import html
import re
import sys
from pathlib import Path

import pandas as pd

from cornerman.collect.titles import name_key
from cornerman.contracts.hold import held_fights

HEADING = re.compile(r"\d+(\.\d+)?\s*kg", re.IGNORECASE)
YOUNG = re.compile(r"\b(jr|junior|juniors|youth|cadet)\b", re.IGNORECASE)


def page_lines(path: Path) -> list[str]:
    s = path.read_text(encoding="utf-8", errors="ignore")
    s = re.sub(r"<script.*?</script>|<style.*?</style>", "", s, flags=re.DOTALL)
    s = html.unescape(re.sub(r"<[^>]+>", "\n", s))
    return [line.strip() for line in s.splitlines() if line.strip()]


def sections(lines: list[str], young_page: bool) -> list[tuple[bool, list[str]]]:
    """(is_young, the section's lines as name keys) per bracket or bout.

    A section starts at a line naming a weight in kg (a bracket heading such as
    "Male Jr Flyweight 56.7 kg ..." or a bout line "Bout 3: Male Flyweight ...").
    """
    out: list[tuple[bool, list[str]]] = []
    for line in lines:
        if HEADING.search(line):
            out.append((young_page or bool(YOUNG.search(line)), [line]))
        elif out:
            out[-1][1].append(line)
    return [(young, [name_key(x) for x in text]) for young, text in out]


def found(key: str, lines: list[str]) -> bool:
    """The name on one line: as a whole, or all its words (other order, extra names)."""
    words = set(key.split())
    return any(f" {key} " in f" {line} " or words <= set(line.split()) for line in lines)


Section = tuple[str, int, bool, list[str]]
"""(page key, page year, is junior or youth, the section's lines as name keys)."""


def load_sections(pages_dir: Path, pages_txt: Path) -> list[Section]:
    secs: list[Section] = []
    for line in pages_txt.read_text().splitlines():
        key, _ = line.split()
        young_page = bool(re.fullmatch(r"\d{4}(j|y)\w*", key))
        for young, text in sections(page_lines(pages_dir / f"{key}.html"), young_page):
            secs.append((key, int(key[:4]), young, text))
    return secs


def judge(name_a: str, name_b: str, year: int, secs: list[Section]) -> tuple[str, str]:
    """(verdict, detail) for one bout; see the module docstring."""
    a, b = name_key(name_a), name_key(name_b)
    near = [(k, y, t) for k, yr, y, t in secs if yr in (year - 1, year)]
    together = [(k, y) for k, y, t in near if found(a, t) and found(b, t)]
    young_any = sorted({k for k, y, t in near if y and (found(a, t) or found(b, t))})
    adult_together = sorted({k for k, y in together if not y})
    if any(y for _, y in together):
        verdict = "junior"
    elif adult_together and not young_any:
        verdict = "adult"
    else:
        verdict = "unclear"
    detail = (
        f"together in adult sections of {adult_together}; "
        f"a fighter in young sections of {young_any}; "
        f"a found: {any(found(a, t) for _, _, t in near)}, "
        f"b found: {any(found(b, t) for _, _, t in near)}"
    )
    return verdict, detail


def main(pages_dir: Path, pages_txt: Path, out_csv: Path) -> None:
    secs = load_sections(pages_dir, pages_txt)
    fights = pd.read_csv("data/fights.csv").set_index("fight_id")
    rows: list[dict[str, str]] = []
    for fid in sorted(held_fights()):
        row = fights.loc[fid]
        verdict, detail = judge(
            str(row["fighter_a_name"]), str(row["fighter_b_name"]), int(row["year"]), secs
        )
        rows.append({"fight_id": fid, "verdict": verdict, "detail": detail})
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(out["verdict"].value_counts().to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
