from typing import Generator

import torch
from torch import nn
from torch.nn import functional as F

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
