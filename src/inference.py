import sys
from model import *
from tokenizer import *


@torch.no_grad()
def gen_until(model: GPT, tokens: torch.Tensor, stop_token: int, top_k: int):
    prompt_len = len(tokens)
    next_tok = None
    while next_tok != stop_token:
        tokens_ctx = tokens[-model.ctx_size:]
        logits = model(tokens_ctx.unsqueeze(-2))[0][0, :]
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('inf')

        probs = torch.softmax(logits[-1], dim=0)
        next_tok = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat((tokens, next_tok))
    return tokens[prompt_len:]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("gpt-ylilauta2-ft.pt", weights_only=False, map_location=device)
    tok = torch.load("y1024-tokenizer.bin", weights_only=False)

    context = torch.tensor([], dtype=torch.long)
    while True:
        prompt = torch.tensor(tok.encode(input("Prompt: ") + "\n"), device=device)
        context = torch.cat((context, prompt))[-model.ctx_size:]
        logits = gen_until(model, context, tok.str_to_tok["\n"], 30)
        context = torch.cat((context, logits))
        print(tok.decode(logits), end="")


if __name__ == "__main__":
    main()
