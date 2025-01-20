# luma
> [!WARNING]
> This project is still a work in progress.

GPT-2 [[1]](#1) implementation with manually computed gradients, inspired by [karpathy/llm.c](https://github.com/karpathy/llm.c) and [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). Also features a BPE tokenizer [[2]](#2). The plan is to eventually rewrite this in C++ with hand-optimized CPU kernels for small, hardware-restricted LM purposes.

## Usage
Requires [PyTorch](https://pytorch.org) >= 2.6.0. Run `python src/gpt.py` to train using the [Shakespeare dataset](data/shsp.txt) and display a sample of inference.

## References
- <a id="1">[1]</a>
Phuong, Mary, and Marcus Hutter. **‘Formal Algorithms for Transformers’.** *arXiv [Cs.LG]*, 2022, http://arxiv.org/abs/2207.09238. [![arXiv](https://img.shields.io/badge/arXiv-2207.09238-b31b1b.svg)](https://arxiv.org/abs/2207.09238)
- <a id="1">[2]</a> 
Sennrich, Rico, et al. **‘Neural Machine Translation of Rare Words with Subword Units’.** *arXiv [Cs.CL]*, 2016, http://arxiv.org/abs/1508.07909. [![arXiv](https://img.shields.io/badge/arXiv-1508.07909-b31b1b.svg)](https://arxiv.org/abs/1508.07909)
