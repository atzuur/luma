import sys
import time
import multiprocessing
from multiprocessing.pool import ThreadPool
from pathlib import Path


def is_valid(content: str) -> bool:
    words = content.split()
    return len(words) >= 10 and all(w not in {"i", "och", "som", "till"} for w in words)


def extract_prose(src: str) -> str:
    prose = ""
    speech_end = 0
    while (speech_start := src.find("<vsk:PuheenvuoroOsa", speech_end)) != -1:
        speech_end = src.find("</vsk:PuheenvuoroOsa", speech_start)
        assert speech_end != -1
        section_end = speech_start
        while (section_start := src.find("<sis:KappaleKooste>", section_end, speech_end)) != -1:
            section_end = src.find("</sis:KappaleKooste>", section_start)
            assert section_end != -1
            content = src[section_start:section_end].removeprefix("<sis:KappaleKooste>")
            if is_valid(content):
                prose += content + " "
    return prose


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {__file__}.py <directory>")
        sys.exit(1)

    directory = Path(sys.argv[1])
    pool = ThreadPool(processes=multiprocessing.cpu_count())
    start_t = time.perf_counter()
    
    with open(f"{directory.name}.txt", "w", encoding="utf-8") as out:
        prose_from_file = lambda file: extract_prose(file.read_text(encoding="utf-8"))
        for prose in pool.map(prose_from_file, directory.iterdir()):
            out.write(prose)

    total_t = time.perf_counter() - start_t
    n_files = len(list(directory.iterdir()))
    print(f"took {total_t:.1f} s for {n_files} files ({float(f"{total_t / n_files:.2g}")} s per file)")


if __name__ == "__main__":
    main()
