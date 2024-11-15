import numpy as np
from collections import defaultdict
from time import perf_counter

corpus = "shsp"
with open(f"{corpus}.txt", "r", encoding="utf-8") as f:
    text = f.read()

# bpe method; see Sennrich et al., 2016 (https://paperswithcode.com/method/bpe)
def tokenize(s: str, max_vocab_size = 256) -> tuple[list, np.ndarray]:
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
    return vocab, np.fromiter(map(stoi.get, s), dtype=np.int64)

tok_path = f"{corpus}-tokens.npy"
vocab_path = f"{corpus}-vocab.npy"
try:
    tokens = np.load(tok_path)
    vocab = list(np.load(vocab_path))
except FileNotFoundError:
    vocab, tokens = tokenize(text)
    np.save(tok_path, tokens)
    np.save(vocab_path, np.array(vocab))

vocab_size = len(vocab)

def decode(a: np.ndarray) -> str:
    return "".join(vocab[i] for i in a)

def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    ps = np.exp(logits)
    B, T = targets.shape
    ps_y = ps[np.arange(B)[:, np.newaxis], np.arange(T), y]
    return -np.mean(np.log(ps_y / ps.sum(2)))

def cross_entropy_dx(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    _, T, C = logits.shape
    ps = np.exp(logits)
    s = ps.sum(2, keepdims=True)
    grad = 1 - (s - ps) / s - np.eye(C)[targets]
    return grad / T

def softmax2d(mat: np.ndarray) -> np.ndarray:
    ps = np.exp(mat)
    return ps / np.sum(ps, 1)[:, np.newaxis]

ctx_size = 32
n_batches = 8
d_emb = 64

rng = np.random.default_rng()
tok_emb = rng.standard_normal((vocab_size, d_emb))

def pos_emb_fn(max_t: int) -> np.ndarray:
    idxs = np.arange(d_emb)
    return np.vstack([
        np.piecewise(
            idxs, [idxs % 2],
            [lambda i: np.sin(t / max_t ** ((i + 1) / d_emb)),
            lambda i: np.cos(t / max_t ** (i / d_emb))]
        ) for t in np.arange(max_t)
    ])
pos_emb = pos_emb_fn(ctx_size)

unembed_w = rng.standard_normal((d_emb, vocab_size))
unembed_b = rng.standard_normal(vocab_size)

def unembed(logits: np.ndarray) -> np.ndarray:
    return logits @ unembed_w + unembed_b

def unembed_dw(x: np.ndarray, d_loss: np.ndarray) -> np.ndarray:
    return x.transpose((0, 2, 1)) @ d_loss

def unembed_dx(d_loss: np.ndarray) -> np.ndarray:
    return d_loss @ unembed_w.T

def get_batch():
    start_idxs = rng.integers(len(tokens) - ctx_size - 1, size=n_batches)
    x = np.stack([tokens[i:i+ctx_size] for i in start_idxs])
    y = np.stack([tokens[i+1:i+ctx_size+1] for i in start_idxs])
    return x, y

def fwd(x: np.ndarray) -> np.ndarray:
    emb = tok_emb[x] + pos_emb_fn(x.shape[1])
    logits = unembed(emb)
    return logits

def estimate_loss():
    total = 0
    for _ in range(200):
        x, y = get_batch()
        logits = fwd(x)
        total += cross_entropy(logits, y)
    return total / 200

def generate(idx: np.ndarray, max_new_tokens: int):
    for _ in range(max_new_tokens):
        logits = fwd(idx)[:, -1, :]
        probs = softmax2d(logits)
        idx_next = rng.multinomial(1, probs).argmax(axis=1)[:, np.newaxis]
        idx = np.concatenate((idx, idx_next), axis=1)
    return idx

for e in range(1001):
    x, y = get_batch()
    emb = tok_emb[x] + pos_emb
    logits = unembed(emb)

    d_loss = cross_entropy_dx(logits, y)
    d_unembed_w = unembed_dw(emb, d_loss)
    d_unembed_x = unembed_dx(d_loss)
    
    unembed_w -= d_unembed_w.mean(0)
    unembed_b -= d_loss.mean((0, 1))
    tok_emb[x] -= d_unembed_x.mean(0)

    if e % 200 == 0:
        print(f"epoch {e}: {estimate_loss()}")

print(decode(generate(np.zeros((1, 1), dtype=np.int64), 500)[0]))
