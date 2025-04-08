import sys
import time
import multiprocessing
from multiprocessing.pool import ThreadPool
from pathlib import Path
from itertools import pairwise
from typing import Generator


def extract_prose(src: str, word_first: bool = False) -> Generator[str]:
    pos = 1 if word_first else 0
    word = 1 - pos
    tokens = (
        s[word] for line in src.splitlines()
        if len(s := line.split()) >= 2 and s[pos].isdigit()
    )
    quotes = "\"'”’"
    attaches_to_next = {"(", "[", "{"}
    attaches_to_prev = {".", ",", ";", ":", "?", "!", ")", "]", "}"}
    for tok, tok_next in pairwise(tokens):
        if tok in quotes:
            continue
        if tok not in attaches_to_next and tok_next not in attaches_to_prev:
            tok += " "
        yield tok.strip(quotes)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {__file__}.py <directory>")
        sys.exit(1)

    directory = Path(sys.argv[1])
    pool = ThreadPool(processes=multiprocessing.cpu_count())
    start_t = time.perf_counter()
    with open(f"{directory.name}.txt", "w", encoding="utf-8") as out:
        prose_from_file = lambda file: "".join(extract_prose(file.read_text(encoding="utf-8"), True))
        for prose in pool.map(prose_from_file, directory.iterdir()):
            out.write(prose)

    total_t = time.perf_counter() - start_t
    n_files = len(list(directory.iterdir()))
    print(f"took {total_t:.1f} s for {n_files} files ({float(f"{total_t / n_files:.2g}")} s per file)")


if __name__ == "__main__":
    main()
