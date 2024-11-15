import re
import sys
from pathlib import Path
from typing import Iterable

HYPERLINK = re.compile(r"!?\[([^\]]*)\]\([^\)]*\)")
ENCLOSED_DIGIT = re.compile(r"\[\d\]")
NUMBERED_LIST = re.compile(r"^\d+[.)]\s")
COLOR_CODE = re.compile(r"{color{.{6}}([^}]*)")
MULTISPACE = re.compile(r" {2,}")
DISCORD_TAG = re.compile(r"\s?-?.[^\s]*\\?#[0-9]{4}")


def find_closing_bracket(s: str, start: int) -> int:
    count = 0
    for i, c in enumerate(s[start:], start):
        if c == "[":
            count += 1
        elif c == "]":
            count -= 1
            if count == 0:
                return i
    return -1


def find_toplevel_bracket_pairs(s: str) -> list[tuple[int, int]]:
    brackets = []
    start = -1
    while (start := s.find("[", start + 1)) != -1:
        end = find_closing_bracket(s, start)
        if end != -1:
            brackets.append((start, end))
            start = end
        else:
            break
    return brackets


def clear_nested_brackets(s: str) -> str:
    brackets = find_toplevel_bracket_pairs(s)
    orig_len = len(s)
    for start, end in brackets:
        diff = orig_len - len(s)
        start -= diff
        end -= diff
        s = s[:start + 1] + s[start + 1:end].replace("[", "").replace("]", "") + s[end:]
    return s


def clean_line(line: str) -> str:
    if "/>" in line or "$" in line:
        return ""

    line = line.strip("#>*:- ")
    if any(map(line.casefold().startswith, (
        "<", "`", "|",  ":", "import",
        "by:", "added:", "last tested:", "last updated:",
        "original ticket:", "search:", "sidebar_position:",
    ))):
        return ""
    
    line = "".join(c for c in line if ord(c) <= ord('°'))

    line = line \
        .replace("*", "") \
        .replace("\\", "") \
        .replace("_", "") \
        .replace("`", "") \
        .replace("description: ", "") \
        .replace("&gt;", ">") \
        .replace("&lt;", "<") \
        .replace("&amp;", "&") \
        .replace("<br/>", "")

    line = clear_nested_brackets(line)
    line = re.sub(HYPERLINK, lambda m: m.group(1), line) # keep link names
    line = re.sub(ENCLOSED_DIGIT, "", line)
    line = re.sub(NUMBERED_LIST, "", line)
    line = re.sub(COLOR_CODE, lambda m: m.group(1), line) # keep colored text
    line = re.sub(MULTISPACE, " ", line)
    line = re.sub(DISCORD_TAG, "", line)
    line = line.strip() + "\n"

    if len(line.split(" ")) < 3:
        return ""

    return line


def clean_file(file: Path) -> Iterable[str]:
    with open(file, "r", encoding="utf-8") as f:
        in_code_or_latex = False
        for line in f:
            if line.startswith("```") or line.startswith("$$"):
                in_code_or_latex = not in_code_or_latex
            if not in_code_or_latex:
                yield clean_line(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean.py <directory>")
        sys.exit(1)

    files = Path(sys.argv[1]).iterdir()
    with open("material.txt", "w", encoding="utf-8") as out:
        for file in files:
            out.writelines(clean_file(file))
