import sys
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"usage: python {__file__}.py <directory> <approx_bytes> [<split_str>]")
        exit(1)

    files = list(Path(sys.argv[1]).iterdir())
    approx_bytes = int(sys.argv[2])
    split_str = sys.argv[3] if len(sys.argv) == 4 else "\n"

    for file in files:
        with open(file, encoding="utf-8") as f:
            i = 0
            prev_surplus = ""
            while data := prev_surplus + f.read(approx_bytes):
                cut_pos = 0
                while chunk := f.read(approx_bytes // 100):
                    if (cut_pos := chunk.find(split_str)) == -1:
                        data += chunk
                        continue
                    cut_pos += len(split_str)
                    data += chunk[:cut_pos]
                    break
                prev_surplus = chunk[cut_pos:]

                Path(f"{file}.{i}").write_text(data, encoding="utf-8")
                i += 1
