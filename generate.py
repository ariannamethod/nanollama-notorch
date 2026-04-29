"""Generate from nanollama-notorch using SentencePiece tokenizer + infer_nanollama binary.

Usage: python3 generate.py "Your prompt here" [--n 100] [--T 0.8] [--k 40] [--seed 0]
"""
import argparse, os, subprocess, sys
import sentencepiece as spm

HOME = os.path.expanduser("~")
DEFAULT_TOKENIZER = f"{HOME}/Downloads/nanollama/weights/tokenizer.model"
DEFAULT_WEIGHTS   = f"{HOME}/nanollama-notorch/nanollama_final.bin"
DEFAULT_BIN       = f"{HOME}/nanollama-notorch/infer_nanollama"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", help="prompt text")
    ap.add_argument("--n", type=int, default=100, help="tokens to generate")
    ap.add_argument("--T", type=float, default=0.8, help="temperature (0=greedy)")
    ap.add_argument("--k", type=int, default=40, help="top-k (0=disabled)")
    ap.add_argument("--seed", type=int, default=0, help="0=time-based")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--weights",   default=DEFAULT_WEIGHTS)
    ap.add_argument("--bin",       default=DEFAULT_BIN)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    prompt_ids = sp.Encode(args.prompt)
    print(f"prompt: {len(prompt_ids)} tokens", file=sys.stderr)

    header = f"{len(prompt_ids)} {args.n} {args.T} {args.k} {args.seed}\n"
    body   = " ".join(map(str, prompt_ids)) + "\n"
    proc = subprocess.Popen(
        [args.bin, args.weights],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
        text=True,
    )
    out, _ = proc.communicate(header + body)
    if proc.returncode != 0:
        print(f"infer exited {proc.returncode}", file=sys.stderr); sys.exit(1)

    tokens = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line == "DONE":
            continue
        tokens.extend(int(t) for t in line.split())

    text = sp.Decode(prompt_ids + tokens)
    print(text)

if __name__ == "__main__":
    main()
