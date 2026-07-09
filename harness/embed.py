"""Shared GENOS embedding extractor — the reusable core of every Phase-4 benchmark.

Input:  a JSONL/CSV with columns {id, sequence[, label]} (DNA strings, A/C/G/T/N, uppercased).
Output: <out>.npy  (N x 4096 float32 mean-pooled embeddings) + <out>.labels.npy (if labels present)
        + <out>.ids.json.

Runs on the pod (bgigenos/vllm:v1) against the cached weights at $GENOS_PATH.
    python embed.py --input data/clinvar.jsonl --out emb/clinvar --batch 8 --maxlen 8192
"""
import argparse, json, os, sys, time

import numpy as np


def read_rows(path):
    rows = []
    if path.endswith(".jsonl"):
        for line in open(path):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    elif path.endswith(".csv"):
        import csv
        rows = list(csv.DictReader(open(path)))
    else:
        sys.exit(f"unsupported input: {path}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("GENOS_PATH", "/workspace/genos-10b-v2"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=8192, help="truncate/pad token length (bp)")
    ap.add_argument("--pool", default="mean", choices=["mean", "last", "max"])
    ap.add_argument("--seq_field", default="sequence")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    rows = read_rows(args.input)
    seqs = [str(r[args.seq_field]).upper() for r in rows]
    ids = [r.get("id", i) for i, r in enumerate(rows)]
    has_labels = all("label" in r for r in rows)
    print(f"{len(seqs)} sequences, labels={has_labels}, model={args.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa",
    ).cuda().eval()

    embs = []
    t0 = time.time()
    for i in range(0, len(seqs), args.batch):
        chunk = seqs[i:i + args.batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.maxlen, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            hs = model(**enc).last_hidden_state          # B x L x 4096
        mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
        if args.pool == "mean":
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
        elif args.pool == "max":
            pooled = (hs.masked_fill(mask == 0, -1e9)).max(1).values
        else:  # last non-pad token
            idx = enc["attention_mask"].sum(1) - 1
            pooled = hs[torch.arange(hs.size(0)), idx]
        embs.append(pooled.float().cpu().numpy())
        if (i // args.batch) % 20 == 0:
            print(f"  {i+len(chunk)}/{len(seqs)}  {time.time()-t0:.0f}s", flush=True)

    X = np.concatenate(embs, axis=0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out + ".npy", X)
    json.dump(ids, open(args.out + ".ids.json", "w"))
    if has_labels:
        y = np.array([r["label"] for r in rows])
        np.save(args.out + ".labels.npy", y)
    print(f"wrote {args.out}.npy  shape={X.shape}  in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
