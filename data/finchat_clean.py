import sys
from pathlib import Path

def clean(src: str):
    out = ""
    for line in src.splitlines():
        start = 0
        for i in range(len(line) - 1):
            if line[i].isnumeric() and line[i + 1] == ",":
                start = i + 2
        line = line[start:].strip("\"").replace("\"\"", "\"")
        out += line + "\n"
    return out


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {__file__}.py <file>")
        sys.exit(1)

    src_file = Path(sys.argv[1])
    cleaned = clean(src_file.read_text(encoding="utf-8"))
    src_file.with_suffix(".txt").write_text(cleaned, encoding="utf-8")


if __name__ == "__main__":
    main()
