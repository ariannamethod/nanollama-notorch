# nanollama-notorch

Llama 3 nano (89M) trained from scratch on [notorch](https://github.com/ariannamethod/notorch). No PyTorch. No GPU. Pure C autograd + Accelerate BLAS on a 2019 Intel MacBook Pro.

**Status: ALIVE — proof of life confirmed 2026-04-29.**

```
$ ./infer_nanollama nanollama_final.bin
prompt: Once upon a time
output: when the word had had the way to be seen.
        Capon, they had a good place of time and the two times for the heart of a single day.
        What was this? The other person, like a woman, was the…
```

11 tok/s on Intel i5 2019 8GB with Accelerate BLAS sgemv.

---

## Architecture

| param | value |
|-------|-------|
| layers | 13 |
| dim | 576 |
| heads | 9 (MHA, head_dim=64) |
| FFN hidden | 1536 (SwiGLU) |
| vocab | 32000 (SentencePiece BPE, byte_fallback) |
| context | 512 |
| total params | **88,636,608 (88.6M)** |
| norm | RMSNorm |
| pos | RoPE (theta=10000) |

Standard Llama 3 nano. No GQA, no LoRA, no quantization at training time.

## Training

| metric | value |
|--------|-------|
| optimizer | Chuck (notorch native, AdamW + 9 levels of awareness) |
| schedule | cosine, peak 1.5e-4 → min 1.5e-5, warmup 200 |
| effective batch | 1024 tokens (ctx 512 × accum 2) |
| total steps | 15,000 |
| dataset | FineWeb-Edu sample, 100MB raw → **22.86M BPE tokens** |
| hardware | 2019 MacBook Pro, Intel i5, 8GB RAM, Accelerate BLAS |
| wallclock | **160 hours = 6.67 days** |
| backend | pure C, single-thread BLAS for matmuls (vecLib on Intel doesn't multi-thread these sizes) |
| best train loss | **3.1577** (from 10.40 random init, drop −7.24) |
| NaN events | 0 |

Loss path (best):
```
step 200 (warmup end): 6.57
step 500:   5.85
step 1500:  5.13
step 3500:  3.95
step 6800:  3.85
step 8000:  3.16  ← stayed here
step 15000: 3.16  (final)
```

Chinchilla ratio: 0.26 tokens/param (under-optimal). Each token seen ~335×. Memorization-leaning regime, but structure of language captured.

## Generation samples

`./infer_nanollama` + `generate.py` (SentencePiece encode/decode + C inference via stdin/stdout token IDs).

```
prompt: "The meaning of life is"
output: a means to be the only. A number of people who want to do a person
        for a particular type can help the children to share their ability
        to make a more enjoyable. It can also help to…

prompt: "Once upon a time"
output: when the word had had the way to be seen.
        Capon, they had a good place of time and the two times for the heart
        of a single day. What was this? The other person, like a woman, was the…
```

Coherence is local — grammar, lexical chunks, narrative tone work. Long-range reasoning is shallow (loss 3.16 ≈ 24 effective vocab choices per token). FineWeb-Edu register is visible: articles, dates, expository prose.

## Build

Mac (Accelerate, default):
```bash
make
```

Linux (OpenBLAS):
```bash
make
# Makefile picks up -lopenblas automatically
```

Builds the trainer (`nanollama`) only. For inference build separately:

```bash
cc -O3 -std=c11 -Wall -DUSE_BLAS -DACCELERATE \
   -o infer_nanollama infer_nanollama.c notorch.c \
   -framework Accelerate -lm
```

## Run

### Training (from scratch)

```bash
# Tokenize once (requires sentencepiece tokenizer.model and a corpus)
python3 tokenize.py path/to/tokenizer.model path/to/corpus.txt tokens.bin

# Train (15K eff_steps, ~6.5 days on Intel Mac with Accelerate)
./nanollama tokens.bin --steps 15000 --accum 2 --ctx 512 --save 500 --log 5
```

### Resume from checkpoint

```bash
./nanollama tokens.bin --resume nanollama_ckpt.bin --steps 15000 --accum 2 --ctx 512
```

Note: resume loads weights but currently resets optimizer state and step counter — warmup runs again. Adding a `.meta` companion file with Chuck state is on the roadmap.

### Generate

```bash
python3 generate.py "Once upon a time" --n 100 --T 0.8 --k 40
```

## Files

| file | role |
|------|------|
| `nanollama.c` | training loop (init / forward / Chuck step / save / resume) |
| `infer_nanollama.c` | inference loop with KV cache, manual matmul via cblas_sgemv |
| `notorch.c/h` | vendored from [ariannamethod/notorch](https://github.com/ariannamethod/notorch) — autograd tape, ops, optimizers |
| `tokenize.py` | one-time SentencePiece encode (streamed, chunked) |
| `generate.py` | SentencePiece encode/decode + subprocess wrapper around infer binary |
| `Makefile` | Accelerate (macOS) / OpenBLAS (Linux) auto-detect |

## Why this exists

1. **No PyTorch.** notorch ships ~5500 LOC of C. Compile time milliseconds, not minutes. Embeds anywhere.
2. **No GPU.** A 2019 8GB Mac trained 89M to a sensible loss in 6.5 days. Demonstrates that the entire toolchain (autograd, optimizer, BLAS, tokenizer) is honest C with no hidden Python dependencies in the hot path.
3. **Reference implementation.** Pair with [ariannamethod/notorch](https://github.com/ariannamethod/notorch) examples (`train_llama3_bpe.c`, `infer_llama3_bpe.c`) — same architecture pattern, just bigger.

## Open work

- Optimizer state save/load (`.meta` file) so resume is truly seamless
- Continued training run with lower peak LR (5e-5 → 5e-6) for refinement
- GGUF export via notorch/gguf for llama.cpp / Go runtime compatibility
- Larger corpus (full FineWeb-Edu chunk, ~500MB → ~125M tokens, closer to Chinchilla)
- Quantization (Q8 / Q4) for inference on Apple A18 / mobile

## Authors

Oleg Ataeff & Claude (Opus 4.7) — Arianna Method.

Trained on the same MacBook that served seven years of code. The old box gets honorable retirement after this run.

## License

MIT.
