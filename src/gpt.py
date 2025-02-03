from collections import defaultdict
from functools import cache
from time import perf_counter

from modules import *

torch.manual_seed(42)

corpus = "data/shsp"
with open(f"{corpus}.txt", "r", encoding="utf-8") as f:
    text = f.read()

# bpe method; see Sennrich et al., 2016 (https://paperswithcode.com/method/bpe)
def tokenize(s: str, max_vocab_size = 256) -> tuple[list, torch.Tensor]:
    word_freqs = defaultdict(int)
    start = 0
    for i, c in enumerate(s):
        if c.isalpha():
            continue
        if 2 <= i - start:
            word = " ".join(s[start:i])
            word_freqs[word] += 1
        start = i + 1

    vocab_size = len(set(s))
    assert vocab_size <= max_vocab_size

    merges = []
    t1 = perf_counter()
    for _ in range(max_vocab_size - vocab_size):
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq

        best = max(pairs, key=pairs.get)
        merges.append(best)

        for word in list(word_freqs):
            symbols = word.split()
            merged_word = word
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair == best:
                    merged_word = " ".join((*symbols[:i], "".join(pair), *symbols[i + 2:]))
            word_freqs[merged_word] = word_freqs.pop(word)

    t = perf_counter() - t1
    print(f"tokenizer found {len(merges)} merges in {t:.2f} s ({t / len(merges):.2f} s/merge)")

    t1 = perf_counter()
    symbols = list(s)
    for pair in merges:
        i = 0
        while i < len(symbols) - 1:
            if (symbols[i], symbols[i + 1]) == pair:
                symbols[i] += symbols.pop(i + 1)
            else:
                i += 1

    t = perf_counter() - t1
    print(f"tokenizer made {len(merges)} merges in {t:.2f} s ({t / len(merges):.2f} s/merge)")

    vocab = list(set(symbols))
    vocab.sort()
    stoi = {s: i for i, s in enumerate(vocab)}
    tokens = [stoi[s] for s in symbols]
    return vocab, torch.as_tensor(tokens)


# tok_path = f"{corpus}-tokens.pt"
# vocab_path = f"{corpus}-vocab.pt"
# try:
#     tokens = torch.load(tok_path)
#     vocab = torch.load(vocab_path)
# except FileNotFoundError:
#     vocab, tokens = tokenize(text)
#     torch.save(tokens, tok_path)
#     torch.save(vocab, vocab_path)

# assert isinstance(vocab, list)
# vocab_size = len(vocab)

# def decode(a: torch.Tensor) -> str:
#     return "".join(vocab[i] for i in a)

chars = sorted(list(set(text)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i.item()] for i in l])

tokens = torch.tensor(encode(text), dtype=torch.long)
vocab_size = len(chars)


ctx_size = 64
n_batches = 12
n_layers = 4
d_emb = 128
n_epochs = 2000

# @cache
# def pos_emb_fn(max_t: int) -> torch.Tensor:
#     idxs = torch.arange(d_emb).unsqueeze(0)
#     ts = torch.arange(max_t).unsqueeze(1)
#     return torch.where(
#         idxs % 2 == 1,
#         torch.sin(ts / max_t ** ((idxs + 1) / d_emb)),
#         torch.cos(ts / max_t ** (idxs / d_emb))
#     )


def get_batch():
    start_idxs = torch.randint(len(tokens) - ctx_size, size=(n_batches,))
    x = torch.stack([tokens[i:i + ctx_size] for i in start_idxs])
    y = torch.stack([tokens[i + 1:i + ctx_size + 1] for i in start_idxs])
    return x, y


class DTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = Embedding(vocab_size, d_emb)
        self.pos_emb = Embedding(ctx_size, d_emb)

        self.ln1 = LayerNorm(d_emb)
        self.mh_att = MHAttention(d_emb, 4)

        self.ln2 = LayerNorm(d_emb)
        self.lin1 = Linear(d_emb, 4 * d_emb)
        self.gelu = GELU()
        self.lin2 = Linear(4 * d_emb, d_emb)
        self.add = Add()

        self.lnf = LayerNorm(d_emb)
        self.vocab_proj = Linear(d_emb, vocab_size)
        self.cross_entropy = CrossEntropy()

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):
        x = self.tok_emb(tokens) + self.pos_emb(torch.arange(tokens.size(-1)))
        for _ in range(n_layers):
            x_tild = self.ln1(x)
            x_tild = self.mh_att(x_tild)
            x = self.add(x, x_tild)
            x_tild = self.ln2(x)
            x_tild = self.lin2(self.gelu(self.lin1(x_tild)))
            x = self.add(x, x_tild)

        x_hat = self.lnf(x)
        logits = self.vocab_proj(x_hat)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = self.cross_entropy(logits.view(B*T, V), targets.view(B*T))
        return logits, loss

    def estimate_loss(self):
        total = 0
        for _ in range(20):
            logits, targets = get_batch()
            _, loss = self(logits, targets)
            total += loss
        return total / 20

    def generate(self, tokens: torch.Tensor, max_new_tokens: int, top_k: int | None = None):
        for _ in range(max_new_tokens):
            tokens_ctx = tokens[-ctx_size:]
            logits, _ = self(tokens_ctx.unsqueeze(-2))
            logits = logits[0]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = torch.softmax(logits[-1], dim=0)
            tokens_next = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, tokens_next), dim=0)
        return tokens


model = DTransformer()
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
try:
    for e in range(n_epochs):
        if e % 200 == 0 or e == n_epochs - 1:
            print(f"epoch {e}: {model.estimate_loss():.4f}")

        x, y = get_batch()
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
except KeyboardInterrupt:
    pass

print(decode(model.generate(torch.zeros(1, dtype=torch.long), 500)))
