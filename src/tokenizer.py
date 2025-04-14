import time
from collections import defaultdict, Counter, OrderedDict
from itertools import groupby, pairwise
from typing import Callable, Generator, Iterable


class Tokenizer:
    def __init__(self, src_gen: Callable[[None], Generator[str]], vocab_size: int):
        t0 = time.perf_counter()

        chars = sorted(set(src_gen()))
        n_chars = len(chars)
        assert n_chars <= vocab_size
        n_merges = vocab_size - n_chars

        self.str_to_tok = OrderedDict((c, i) for c, i in zip(chars, range(n_chars)))
        self.merges = [0] * 2 * n_merges
        word_freqs = Counter(" ".join(g) for k, g in groupby(src_gen(), key=str.isalnum) if k)

        t = time.perf_counter() - t0
        print(f"tok: counted {n_chars} unique chars and {len(word_freqs)} words in {t:.2f} s")

        t0 = time.perf_counter()

        for m in range(n_merges):
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                for pair in pairwise(word.split()):
                    pairs[pair] += freq
            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            self.str_to_tok["".join(best)] = len(self.str_to_tok)
            self.merges[2 * m] = self.str_to_tok[best[0]]
            self.merges[2 * m + 1] = self.str_to_tok[best[1]]

            for word in list(word_freqs):
                symbols = word.split()
                i = 0
                while i < len(symbols) - 1:
                    pair = (symbols[i], symbols[i + 1])
                    if pair == best:
                        symbols[i] += symbols.pop(i + 1)
                        i += 2
                    else:
                        i += 1
                merged_word = " ".join(symbols)
                if merged_word != word:
                    word_freqs[merged_word] = word_freqs.pop(word)

        self.tok_to_str = list(self.str_to_tok)

        t = time.perf_counter() - t0
        print(f"tok: found {n_merges} merges in {t:.2f} s ({t / n_merges:.2f} s/merge)")

    def encode(self, src: str) -> list:
        first_merge_idx = next(
            (i for i, t in enumerate(self.str_to_tok.keys()) if len(t) > 1),
            -1
        )
        tokens = [self.str_to_tok[c] for c in src]
        merged_tokens = [0] * len(tokens)

        for m in range(0, len(self.merges), 2):
            i = 0
            j = 0
            while i < len(tokens):
                can_merge = (
                    i != len(tokens) - 1 and
                    tokens[i] == self.merges[m] and
                    tokens[i + 1] == self.merges[m + 1]
                )
                if can_merge:
                    merged_tokens[j] = first_merge_idx + m // 2
                    i += 2
                else:
                    merged_tokens[j] = tokens[i]
                    i += 1
                j += 1
            tmp = tokens
            tokens = merged_tokens[:j]
            merged_tokens = tmp

        return merged_tokens

    def decode(self, tokens: Iterable) -> str:
        return "".join(self.tok_to_str[t] for t in tokens)

