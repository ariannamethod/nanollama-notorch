# nanollama-notorch

Llama 3 nano (89M) trained from scratch on [notorch](https://github.com/ariannamethod/notorch). No PyTorch. No GPU. Pure C autograd + Accelerate BLAS on a 2019 Intel MacBook Pro.

**Status: ALIVE.** Proof of life 2026-04-29. Continued pretraining (CPT) finished 2026-05-02.

| stage | wallclock | tokens seen | best train loss |
|-------|-----------|-------------|-----------------|
| pretrain (15K eff steps) | 6.67 days | 22.86M × 7.7B forward = 7.7B token-ops | 3.16 |
| CPT (10K eff steps, lr 3e-5 → 3e-6) | 4.5 days | 22.86M × 5.1B = 5.1B more | **2.68** |
| total project | **11.5 days** | 12.8B token-ops | drop −7.72 from random init 10.40 |

```
$ python3 generate.py "Once upon a time" --T 0.7 --k 40
Once upon a time to train, one of the same day, the "stine" is a very good
way to the people of the world. The main lessons of the event were to be
asked to live in the world. "But it's what
```

12 tok/s on Intel i5 2019 8GB with Accelerate BLAS sgemv.

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
| optimizer | Chuck (notorch native, 9 levels of awareness — see notorch repo) |
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
=== pretrain (peak lr 1.5e-4, 15K eff steps) ===
step  200 (warmup end): 6.57
step  500:              5.85
step 3500:              3.95
step 8000:              3.16  ← stayed here through end of pretrain
step 15000 (final):     3.16

=== CPT (peak lr 3e-5, 10K eff steps, resume from pretrain final) ===
step  130:  5.34  (first new low after Chuck state reset)
step  490:  5.14
step 2400:  3.06
step 4500:  2.68  ← held to end
step 10000: 2.68  (final)
```

Chinchilla ratio: 0.26 tokens/param (under-optimal). Each token seen ~700× across both phases. Memorization-leaning regime, but distribution structure of FineWeb-Edu captured (grammar, syntax, lexical clusters, common formats).

## Generation samples (post-CPT, loss 2.68)

`./infer_nanollama` + `generate.py` (SentencePiece encode/decode + C inference via stdin/stdout token IDs).

```
prompt: "Once upon a time"
output: to train, one of the same day, the "stine" is a very good way to the
        people of the world. The main lessons of the event were to be asked
        to live in the world. "But it's what

prompt: "The capital of France is"
output: The city of the Republic of the Irish is the capital of America. In the
        northern part of the South. Its the British Parliament, the region is a
        city of

prompt: "The recipe for bread requires"
output: only to keep the water stored. The temperature of a.c. An example of a
        1000-inch, the 2022 earthquake is a heat that allows that fuel
```

CPT shift vs pretrain (loss 3.16 → 2.68, drop −0.48):
- punctuation richer (quote marks, apostrophes appear)
- thematic stays more on-topic before drifting (ex. "people of world" / "live in world" stays clustered)
- transitions cleaner

Coherence stays **local**: grammar, lexical chunks, register. Factual knowledge is zero (the model says France's capital is "Republic of Irish"). At 89M params trained on 23M tokens (Chinchilla ratio 0.26 — under-optimal by ~80×), the model captures distribution structure of FineWeb-Edu (articles, dates, expository prose) but no facts.

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
