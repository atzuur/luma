import sys
import time
import multiprocessing
import unicodedata
from multiprocessing.pool import ThreadPool
from pathlib import Path


def clean_unicode(src: str):
    keep = ("ä", "å", "ö", "Ä", "Å", "Ö")
    src = unicodedata.normalize("NFKD", src)
    for k in keep:
        src = src.replace(unicodedata.normalize("NFKD", k), k)
    return "".join(c for c in src if c in keep or c.isascii())


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {__file__}.py <directory>")
        sys.exit(1)

    directory = Path(sys.argv[1])
    n_files = len(list(directory.iterdir()))
    pool = ThreadPool(processes=multiprocessing.cpu_count())
    start_t = time.perf_counter()

    clean = lambda file: (file, clean_unicode(file.read_text(encoding="utf-8")))
    for file, cleaned in pool.map(clean, directory.iterdir()):
        file.write_text(cleaned, encoding="utf-8")

    total_t = time.perf_counter() - start_t
    print(f"took {total_t:.1f} s for {n_files} files ({float(f"{total_t / n_files:.2g}")} s per file)")


if __name__ == "__main__":
    main()
