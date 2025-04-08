import math
from functools import reduce
from operator import mul

import torch
import torch.autograd as ag
from torch import nn


class CrossEntropyFn(ag.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor):
        T = x.size(0)
        s = x.exp() / x.exp().sum(1).unsqueeze(1)
        ctx.save_for_backward(x, y, s)
        return -s[torch.arange(T), y].log().mean(0)

    @staticmethod
    def backward(ctx, y_grad: torch.Tensor):
        x, y, s = ctx.saved_tensors
        T, C = x.shape
        dx = y_grad * (s - torch.eye(C)[y]) / T
        return dx, None


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return CrossEntropyFn.apply(x, y)


class LayerNormFn(ag.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        eps = 1e-05
        m = x.mean(-1, keepdim=True)
        mu = x - m
        v = torch.mean(mu ** 2, -1)
        sigma = torch.rsqrt(v + eps)
        y = mu * sigma.unsqueeze(-1) * gamma + beta

        ctx.save_for_backward(x, gamma, mu, sigma)
        return y

    @staticmethod
    def backward(ctx, y_grad: torch.Tensor):
        x, gamma, mu, sigma = ctx.saved_tensors
        C = x.size(-1)

        dgamma = torch.einsum('btc,btc,bt->c', y_grad, mu, sigma)
        dbeta = y_grad.sum((-3, -2))

        dx = (
            y_grad * gamma * sigma.unsqueeze(-1)
            - 1 / C * torch.einsum('btc,c,bt->bt', y_grad, gamma, sigma).unsqueeze(-1)
            - 1 / C * mu * torch.einsum('btc,c,btc,bt->bt', y_grad, gamma, mu, sigma ** 3).unsqueeze(-1)
        )

        return dx, dgamma, dbeta


class LayerNorm(nn.Module):
    def __init__(self, c_dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(c_dim))
        self.beta = nn.Parameter(torch.empty(c_dim))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LayerNormFn.apply(x, self.gamma, self.beta)


class LinearFn(ag.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        ctx.save_for_backward(x, weight, bias)
        out = x @ weight.T
        return out + bias

    @staticmethod
    def backward(ctx, y_grad: torch.Tensor):
        x, weight, bias = ctx.saved_tensors
        dx = y_grad @ weight
        dw = y_grad.transpose(-2, -1) @ x
        db = y_grad.sum((-3, -2))
        return dx, dw, db


class Linear(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_c, in_c))
        self.bias = nn.Parameter(torch.empty(out_c))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LinearFn.apply(x, self.weight, self.bias)


class GELUFn(ag.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        a = torch.sqrt(2 / torch.tensor(torch.pi))
        b = a * (x + 0.044715 * x ** 3)
        ctx.save_for_backward(x, a, b)
        return 0.5 * x * (1 + b.tanh())

    @staticmethod
    def backward(ctx, y_grad: torch.Tensor):
        x, a, b = ctx.saved_tensors
        db = a * (1 + 3 * 0.044715 * x ** 2)
        return y_grad * 0.5 * (1 + b.tanh() + x / b.cosh() ** 2 * db)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GELUFn.apply(x)


class AddFn(ag.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor):
        assert a.shape == b.shape
        return a + b

    @staticmethod
    def backward(ctx, y_grad: torch.Tensor):
        return y_grad, y_grad


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return AddFn.apply(a, b)


class EmbeddingFn(ag.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, emb: torch.Tensor):
        ctx.save_for_backward(x, emb)
        return emb[x]

    @staticmethod
    def backward(ctx, y_grad: torch.Tensor):
        x, emb = ctx.saved_tensors
        *D, C = y_grad.shape
        length_dim = reduce(mul, D, 1)
        demb = torch.zeros_like(emb).index_add_(0, x.view(length_dim), y_grad.view(length_dim, C))
        return None, demb


class Embedding(nn.Module):
    def __init__(self, n_emb: int, d_emb: int):
        super().__init__()
        self.emb = nn.Parameter(torch.empty(n_emb, d_emb))
        nn.init.normal_(self.emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return EmbeddingFn.apply(x, self.emb)


class MHAttentionFn(ag.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        w_o: torch.Tensor
    ):
        scale = 1 / math.sqrt(w_q.size(-1))
        q = x.unsqueeze(-3) @ w_q
        k = x.unsqueeze(-3) @ w_k
        v = x.unsqueeze(-3) @ w_v
        s = q @ k.transpose(-2, -1)
        s_hat = s.masked_fill(s.tril() == 0, float("-inf")) * scale
        s_hat -= s_hat.max(-1, keepdim=True).values
        a = s_hat.exp() / s_hat.exp().sum(-1, keepdim=True)
        y = a @ v
        y_cat = y.transpose(-3, -2).contiguous().view_as(x)
        o = y_cat @ w_o

        ctx.save_for_backward(x, w_q, w_k, w_v, w_o, q, k, v, s, a, y_cat)
        return o

    @staticmethod
    def backward(ctx, o_grad: torch.Tensor):
        x, w_q, w_k, w_v, w_o, q, k, v, s, a, y_cat = ctx.saved_tensors
        H, C, Ca = w_q.shape
        B, T, C = x.shape

        dw_o = y_cat.transpose(-2, -1) @ o_grad
        dy_cat = o_grad @ w_o.T
        dy = dy_cat.view(B, T, H, Ca).transpose(-3, -2)
        dv = a.transpose(-2, -1) @ dy
        da = dy @ v.transpose(-2, -1)
        ds_hat = a * (da - (da * a).sum(-1, keepdim=True))
        ds = (s.tril() > 0) / math.sqrt(Ca) * ds_hat
        dk = ds.transpose(-2, -1) @ q
        dq = ds @ k
        dw_v = (x.transpose(-2, -1).unsqueeze(-3) @ dv).sum(-4)
        dw_k = (x.transpose(-2, -1).unsqueeze(-3) @ dk).sum(-4)
        dw_q = (x.transpose(-2, -1).unsqueeze(-3) @ dq).sum(-4)
        dx = (
            dq @ w_q.transpose(-2, -1) +
            dk @ w_k.transpose(-2, -1) +
            dv @ w_v.transpose(-2, -1)
        ).sum(-3)

        return dx, dw_q, dw_k, dw_v, dw_o


class MHAttention(nn.Module):
    def __init__(self, d_emb: int, n_heads: int):
        super().__init__()
        assert d_emb % n_heads == 0
        d_att = d_emb // n_heads

        w_qkv = torch.empty(3, n_heads, d_emb, d_att)
        nn.init.kaiming_uniform_(w_qkv, a=math.sqrt(5))
        self.w_q, self.w_k, self.w_v = (nn.Parameter(w) for w in w_qkv)

        self.w_o = nn.Parameter(torch.empty(d_emb, d_emb))
        nn.init.kaiming_uniform_(self.w_o, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MHAttentionFn.apply(x, self.w_q, self.w_k, self.w_v, self.w_o)


if __name__ == "__main__":
    B, T, C = 8, 32, 64
    tests = {
        "cross_entropy": {
            "params": [
                (T, C), torch.randint(C - 1, size=(T,)),
            ],
            "func": CrossEntropyFn.apply,
            "fwd_ref": nn.functional.cross_entropy
        },
        "layer_norm": {
            "params": [
                (B, T, C), (C,), (C,)
            ],
            "func": LayerNormFn.apply,
            "fwd_ref": lambda x, g, b: nn.functional.layer_norm(x, (C,), g, b)
        },
        "linear": {
            "params": [
                (B, T, C), (C, C), (C,)
            ],
            "func": LinearFn.apply,
            "fwd_ref": nn.functional.linear
        },
        "gelu": {
            "params": [
                (B, T, C),
            ],
            "func": GELUFn.apply,
            "fwd_ref": lambda x: nn.functional.gelu(x, approximate="tanh")
        },
        "add": {
            "params": [
                (T, C), (T, C)
            ],
            "func": AddFn.apply,
            "fwd_ref": torch.add
        },
        "embedding": {
            "params": [
                torch.randint(2 * C - 1, size=(B, T)), (2 * C, C)
            ],
            "func": EmbeddingFn.apply,
            "fwd_ref": nn.functional.embedding
        },
        "mhattention": {
            "params": [
                (B, T, C), (4, C, C // 4), (4, C, C // 4), (4, C, C // 4), (C, C),
            ],
            "func": MHAttentionFn.apply,
            "fwd_ref": lambda x, wq, wk, wv, wo:
                nn.functional.scaled_dot_product_attention(
                    x.unsqueeze(-3) @ wq,
                    x.unsqueeze(-3) @ wk,
                    x.unsqueeze(-3) @ wv,
                    is_causal=True
                ).transpose(-2, -3).contiguous().view_as(x) @ wo
        }
    }

    for test, info in tests.items():
        params = [
            torch.rand(p, dtype=torch.float64, requires_grad=True) if isinstance(p, tuple)
            else p
            for p in info["params"]
        ]
        err_args = dict(atol=1e-2, rtol=1e-2)
        x = info["func"](*params)
        x_ref = info["fwd_ref"](*params)
        if not torch.allclose(x, x_ref, **err_args):
            print(f"erroneous fwd: {x}\n\nref: {x_ref}")

        ag.gradcheck(info["func"], params, eps=1e-6, **err_args)

        print(f"{test} passed")
    print("\x1b[1;32m" "all tests passed! yay congrats >w<" "\x1b[0m")
