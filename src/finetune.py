from pathlib import Path

import torch

from model import *
from tokenizer import *
from train import train, Dataset


def main():
    batch_size = 32
    n_steps = 10
    n_eval_steps = 0
    log_interval = 2
    learning_rate = 3e-5
    weight_decay = 1e-1
    betas = (0.9, 0.95)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = [f for f in Path("data/finchat").iterdir() if f.suffix == ".txt"]
    ft_dataset = Dataset("y1024", files, 1024)
    model = torch.load("gpt-ylilauta2.pt", weights_only=False, map_location=device)

    train(
        model,
        ft_dataset,
        batch_size,
        n_steps,
        n_eval_steps,
        log_interval,
        learning_rate,
        weight_decay,
        betas,
        warmup_steps=0,
        decay_lr=False
    )
    torch.save(model, "gpt-ylilauta2-ft.pt")

if __name__ == "__main__":
    main()
