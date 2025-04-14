import time
import math
from pathlib import Path
from datetime import timedelta

import torch

from tokenizer import Tokenizer
from model import GPT


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
            t0 = time.perf_counter()
            src = f.read_text(encoding="utf-8")
            tokens = torch.tensor(self.tok.encode(src))
            t = time.perf_counter() - t0
            print(f"tok: encoded {len(src) / 1e6:.1f} M chars in {t:.2f} s")

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
    log_interval: int,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    warmup_steps: int,
    decay_lr: bool = True,
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
            lr = get_lr(k, n_steps, learning_rate, warmup_steps) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if k % log_interval == 0:
                end_t = time.perf_counter()
                avg_t = (end_t - start_t) / log_interval
                eta = int((n_steps - k) * avg_t)
                avg_loss = ""
                if n_eval_steps > 0:
                    avg_loss = f": {model.estimate_loss(n_eval_steps, batch_gen):.4f}"
                print(
                    f"step {k}/{n_steps}{avg_loss}, "
                    f"step time: {avg_t:.2f} s, eta: {timedelta(seconds=eta)}"
                )
                start_t = time.perf_counter()

            x, y = next(batch_gen)
            _, loss = model(x, y, return_logits=False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    log_interval = n_steps // 200
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
        log_interval,
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
