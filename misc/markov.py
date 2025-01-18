import random
from pathlib import Path
import sys

SENTENCE_ENDS = {".", "!", "?", "\n"}
PUNCTUATION = ",;:()\"'"


def find_continuation_word(words: list[str], idx: int, start: int) -> int:
    word = words[idx]
    if any(c.isdigit() for c in word) or len(word) == 1 and not word.isalpha():
        return idx + 1
    for i, w in enumerate(words[start:], start):
        if w.strip(PUNCTUATION) == word:
            return i
    return -1


def split_words_with_nl(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    words = []
    for line in lines:
        words += line.split(" ")
    return words


def random_first_word(words: list[str]) -> int:
    sentence_starts = [0] + [
        i for i, w in enumerate(words[1:], 1) 
        if w[0].isupper()
        and words[i - 1][-1] in SENTENCE_ENDS
        and w[-1] not in SENTENCE_ENDS
    ]
    return random.choice(sentence_starts)


def gen_sentence(text: str, max_len: int = 15) -> str:
    words = split_words_with_nl(text)
    word_idx = random_first_word(words)
    sentence = [words[word_idx]]
    next_start = lambda: 0 if word_idx == len(words) - 1 else word_idx + 1

    while (word_idx := find_continuation_word(
        words, word_idx, next_start())
    ) != -1 and len(sentence) < max_len:
        word_idx += 1
        next_word = words[word_idx].strip(PUNCTUATION)
        sentence.append(next_word)
        if next_word[-1] in SENTENCE_ENDS:
            break

    sentence = " ".join(sentence).strip().replace("\n", "")
    if sentence[-1] not in SENTENCE_ENDS:
        sentence += "."

    return sentence


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gen.py <words_file> <num_sentences>")
        sys.exit(1)

    words_file = Path(sys.argv[1])
    num_sentences = int(sys.argv[2])
    for _ in range(num_sentences):
        print(gen_sentence(words_file.read_text(encoding="utf-8")))
