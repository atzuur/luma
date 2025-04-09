import time
import math
from collections import defaultdict, Counter, OrderedDict
from datetime import timedelta
from itertools import groupby, pairwise
from typing import Callable, Generator, Iterable
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


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
        t0 = time.perf_counter()
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

        t = time.perf_counter() - t0
        print(f"tok: encoded {len(src) / 1e6:.1f} M chars in {t:.2f} s")

        return merged_tokens

    def decode(self, tokens: Iterable) -> str:
        return "".join(self.tok_to_str[t] for t in tokens)


class Dataset:
    def __init__(self, name: str, files: list[Path], vocab_size: int):
        self.files = files
        self.name = name

        tokenizer_file = Path(f"{self.name}-tokenizer.bin")
        if tokenizer_file.exists():
            self.tok = torch.load(tokenizer_file, weights_only=False)
        else:
            self.tok = Tokenizer(self.gen_contents, vocab_size)
            torch.save(self.tok, tokenizer_file)

        for f in self.files:
            token_file = self.get_token_file(f)
            if token_file.exists():
                continue
            print(f"encoding {f.name}")
            tokens = torch.tensor(self.tok.encode(f.read_text(encoding="utf-8")))
            torch.save(tokens, token_file)

    def get_token_file(self, file: Path):
        return file.with_name(f"{file.stem}-{self.name}-tokens.pt")

    def gen_contents(self):
        for f in self.files:
            text = f.read_text(encoding="utf-8")
            yield from text

    def gen_batch(self, batch_size: int, ctx_size: int, device: torch.device):
        while True:
            file_idx = torch.randint(0, len(self.files), ()).item()
            tokens = torch.load(self.get_token_file(self.files[file_idx]))
            n_batches = int(len(tokens) / (batch_size * ctx_size) + 1)

            for _ in range(n_batches):
                start_idxs = torch.randint(len(tokens) - ctx_size, size=(batch_size,))
                x = torch.stack([tokens[i:i + ctx_size] for i in start_idxs])
                y = torch.stack([tokens[i + 1:i + ctx_size + 1] for i in start_idxs])
                yield x.to(device), y.to(device)


class CausalSelfAttn(nn.Module):
    def __init__(self, d_emb: int, n_heads: int):
        super().__init__()
        assert d_emb % n_heads == 0
        self.n_heads = n_heads
        self.qkv_proj = nn.Linear(d_emb, 3 * d_emb, bias=False)
        self.o_proj = nn.Linear(d_emb, d_emb, bias=False)

    def forward(self, x: torch.Tensor):
        b, t, c = x.shape
        qkv = self.qkv_proj(x).view(b, t, 3 * self.n_heads, c // self.n_heads)
        q, k, v = qkv.transpose(1, 2).split(self.n_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.o_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, d_emb: int, n_heads: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_emb)
        self.attn = CausalSelfAttn(d_emb, n_heads)
        self.ln_2 = nn.LayerNorm(d_emb)
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, 4 * d_emb),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * d_emb, d_emb)
        )

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ctx_size: int,
        d_emb: int,
        n_layers: int,
        n_heads: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ctx_size = ctx_size
        self.d_emb = d_emb
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.wte = nn.Embedding(vocab_size, d_emb)
        self.wpe = nn.Embedding(ctx_size, d_emb)
        self.blocks = nn.Sequential(*(Block(d_emb, n_heads) for _ in range(n_layers)))
        self.ln_f = nn.LayerNorm(d_emb)

    def forward(
        self,
        tokens: torch.IntTensor,
        targets: torch.IntTensor = None,
        return_logits: bool = True
    ) -> tuple[torch.Tensor, float]:
        t = tokens.size(1)
        assert t <= self.ctx_size
        pos = torch.arange(t, dtype=torch.int32, device=tokens.device)

        x = self.wte(tokens) + self.wpe(pos)
        x = self.blocks(x)
        x = self.ln_f(x)

        if targets is None:
            logits = x[:, [-1], :] @ self.wte.weight.T
            loss = None
        else:
            logits = x @ self.wte.weight.T
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits if return_logits else None, loss

    @property
    def device(self):
        return next(self.parameters()).device

    def get_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
    ):
        params = [p for p in self.parameters() if p.requires_grad]
        optim_groups = [
            {'params': [p for p in params if p.dim() >= 2], 'weight_decay': weight_decay},
            {'params': [p for p in params if p.dim() < 2], 'weight_decay': 0.0}
        ]
        use_fused = self.device.type == "cuda"
        return torch.optim.AdamW(optim_groups, learning_rate, betas, fused=use_fused)

    @torch.no_grad()
    def estimate_loss(self, steps: int, batch_gen: Generator[tuple[torch.Tensor, torch.Tensor]]):
        total = 0
        for k in range(steps):
            x, y = next(batch_gen)
            _, loss = self(x, y)
            total += loss
        return total / steps

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int, top_k: int):
        for _ in range(max_new_tokens):
            tokens_ctx = tokens[-self.ctx_size:]
            logits = self(tokens_ctx.unsqueeze(-2))[0][0, :]
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')

            probs = torch.softmax(logits[-1], dim=0)
            tokens_next = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, tokens_next), dim=0)
        return tokens


def get_lr(k: int, n_steps: int, lr: float, warmup_steps: int):
    if k < warmup_steps:
        return lr * (k + 1) / (warmup_steps + 1)
    decay_ratio = (k - warmup_steps) / (n_steps - warmup_steps)
    coeff = 0.5 + math.cos(math.pi * decay_ratio) / 2
    min_lr = lr / 10
    return min_lr + coeff * (lr - min_lr)


def train(
    model: GPT,
    dataset: Dataset,
    batch_size: int,
    n_steps: int,
    n_eval_steps: int,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    warmup_steps: int
):
    print(
        "training gpt with "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters"
    )

    batch_gen = dataset.gen_batch(batch_size, model.ctx_size, model.device)
    optimizer = model.get_optimizer(weight_decay, learning_rate, betas)
    start_t = time.perf_counter()

    for k in range(1, n_steps + 1):
        try:
            lr = get_lr(k, n_steps, learning_rate, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            log_interval = n_steps // 100
            if k % log_interval == 0:
                end_t = time.perf_counter()
                avg_t = (end_t - start_t) / log_interval
                eta = int((n_steps - k) * avg_t)
                print(
                    f"step {k}/{n_steps}: {model.estimate_loss(n_eval_steps, batch_gen):.4f}, "
                    f"step time: {avg_t:.2f} s, eta: {timedelta(seconds=eta)}"
                )
                start_t = time.perf_counter()

            x, y = next(batch_gen)
            _, loss = model(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        except KeyboardInterrupt:
            print(f"training aborted on step {k}")
            return


def main():
    vocab_size = 1024
    ctx_size = 512
    batch_size = 12
    n_layers = 12
    n_heads = 12
    d_emb = 768
    n_steps = 100000
    n_eval_steps = 20
    learning_rate = 6e-4
    weight_decay = 1e-1
    betas = (0.9, 0.95)
    warmup_steps = 500

    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = [f for f in Path("data/ylilauta").iterdir() if f.suffix == ".txt"]
    dataset = Dataset(f"y{vocab_size}", files, vocab_size)
    model = GPT(vocab_size, ctx_size, d_emb, n_layers, n_heads).to(device)

    train(
        model,
        dataset,
        batch_size,
        n_steps,
        n_eval_steps,
        learning_rate,
        weight_decay,
        betas,
        warmup_steps
    )
    torch.save(model, "gpt-ylilauta.pt")

    prompt = torch.tensor(dataset.tok.encode("vituttaa ku"), device=model.device)
    print("sample output:")
    print(dataset.tok.decode(model.generate(prompt, 256, 30)))


if __name__ == "__main__":
    main()
