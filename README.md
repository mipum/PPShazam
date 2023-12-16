![# PPShazam](assets/logo.png)

# PPShazam
A privacy preserving version of Shazam using Concrete ML

## Table of Contents

- [Benchmark](#benchmark)

## Benchmark

Compared to ChatGLM's [P-Tuning](https://github.com/THUDM/ChatGLM2-6B/tree/main/ptuning), LLaMA-Factory's LoRA tuning offers up to **3.7 times faster** training speed with a better Rouge score on the advertising text generation task. By leveraging 4-bit quantization technique, LLaMA-Factory's QLoRA further improves the efficiency regarding the GPU memory.

![benchmark](assets/benchmark.svg)

<details><summary>Definitions</summary>

- **Training Speed**: the number of training samples processed per second during the training. (bs=4, cutoff_len=1024)
- **Rouge Score**: Rouge-2 score on the development set of the [advertising text generation](https://aclanthology.org/D19-1321.pdf) task. (bs=4, cutoff_len=1024)
- **GPU Memory**: Peak GPU memory usage in 4-bit quantized training. (bs=1, cutoff_len=1024)
- We adopt `pre_seq_len=128` for ChatGLM's P-Tuning and `lora_rank=32` for LLaMA-Factory's LoRA tuning.

</details>

