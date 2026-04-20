"""One-time tokenization: fineweb.txt -> tokens.bin using SentencePiece.

Streams text in line-chunks and writes int32 stream so 100MB raw input
does not balloon memory through a 25M-item argument list.
"""
import array, os, struct, sys, sentencepiece as spm

HOME = os.path.expanduser("~")
model_path = sys.argv[1] if len(sys.argv) > 1 else f"{HOME}/Downloads/nanollama/weights/tokenizer.model"
text_path  = sys.argv[2] if len(sys.argv) > 2 else f"{HOME}/Downloads/fineweb_edu_100m.txt"
out_path   = sys.argv[3] if len(sys.argv) > 3 else f"{HOME}/nanollama-notorch/tokens.bin"
chunk_lines = int(sys.argv[4]) if len(sys.argv) > 4 else 2000

sp = spm.SentencePieceProcessor(model_file=model_path)
print(f"tokenizer: {sp.GetPieceSize()} vocab from {model_path}")
print(f"input    : {text_path} ({os.path.getsize(text_path)/1e6:.1f} MB)")
print(f"output   : {out_path}")

total = 0
buf = []
with open(text_path, 'r', encoding='utf-8', errors='replace') as fin, \
     open(out_path, 'wb') as fout:
    fout.write(struct.pack('i', 0))  # placeholder for count
    for line in fin:
        buf.append(line)
        if len(buf) >= chunk_lines:
            ids = sp.Encode("".join(buf))
            array.array('i', ids).tofile(fout)
            total += len(ids)
            buf = []
            if total % (1 << 20) < len(ids):
                print(f"  {total/1e6:.2f}M tokens")
    if buf:
        ids = sp.Encode("".join(buf))
        array.array('i', ids).tofile(fout)
        total += len(ids)
    fout.seek(0)
    fout.write(struct.pack('i', total))

size_mb = os.path.getsize(out_path) / 1e6
print(f"done: {total:,} tokens, {size_mb:.1f} MB -> {out_path}")
