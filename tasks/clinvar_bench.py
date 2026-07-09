"""ClinVar pathogenic-variant benchmark on GENOS (headline: ROC-AUC 0.9326, 8192bp).

Genomics Long-Range Benchmark task variant_effect_pathogenic_clinvar: each variant ships a
ref and alt 8192bp window centered on the SNP. Protocol: embed both windows (all 13 layers,
mean-pooled 4096-d), per layer build a feature = concat(ref, alt, |ref-alt|), sweep a head,
report best layer's test AUC (positive class = Pathogenic). Embeddings cached to the volume.

    python clinvar_bench.py --seqlen 8192 --max_train 10000 --head both
"""
import argparse, json, os, time, traceback

import numpy as np

TARGET = 0.9326


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("GENOS_PATH", "/workspace/genos-10b-v2"))
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--max_train", type=int, default=10000, help="subsample train for a first-pass (0=all)")
    ap.add_argument("--max_test", type=int, default=0, help="0 = full test split")
    ap.add_argument("--head", default="both", choices=["logreg", "mlp", "both"])
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    out = "/workspace/bench_clinvar_result.json"
    res = {"task": "variant_effect_pathogenic_clinvar", "target_auc": TARGET, "seqlen": args.seqlen,
           "head": args.head, "started": time.strftime("%F %T")}

    def save():
        json.dump(res, open(out, "w"), indent=2)

    import torch
    from sklearn.metrics import roc_auc_score
    cache = f"/workspace/emb_cache/clinvar_{args.seqlen}_{args.max_train}_{args.max_test}.npz"

    if os.path.exists(cache):
        z = np.load(cache)
        n_layers = int(z["n_layers"])
        RTr = [z[f"rtr{L}"] for L in range(n_layers)]; ATr = [z[f"atr{L}"] for L in range(n_layers)]
        RTe = [z[f"rte{L}"] for L in range(n_layers)]; ATe = [z[f"ate{L}"] for L in range(n_layers)]
        ytr, yte = z["ytr"], z["yte"]
        res["emb_cache"] = "hit"; res["n_hidden_layers"] = n_layers
        print("emb cache HIT", flush=True)
    else:
        from datasets import load_dataset
        from transformers import AutoModel, AutoTokenizer
        ds = None
        for kw in ({"trust_remote_code": True}, {}):
            try:
                ds = load_dataset("InstaDeepAI/genomics-long-range-benchmark",
                                  task_name="variant_effect_pathogenic_clinvar",
                                  sequence_length=args.seqlen, **kw)
                break
            except Exception as e:
                res["load_err"] = str(e)[:250]
        if ds is None:
            save(); print("clinvar load failed:", res.get("load_err"), flush=True); return
        cols = ds["train"].column_names
        res["columns"] = cols
        refk = next(c for c in cols if "ref" in c.lower() and "seq" in c.lower())
        altk = next(c for c in cols if "alt" in c.lower() and "seq" in c.lower())
        labk = "label" if "label" in cols else next(c for c in cols if "label" in c.lower())
        tr, te = ds["train"], ds["test"]
        if args.max_train and len(tr) > args.max_train:
            tr = tr.shuffle(seed=42).select(range(args.max_train))
        if args.max_test and len(te) > args.max_test:
            te = te.shuffle(seed=42).select(range(args.max_test))
        res["n_train"], res["n_test"] = len(tr), len(te)
        ytr = np.array([int(x) for x in tr[labk]]); yte = np.array([int(x) for x in te[labk]])
        print(f"clinvar: train={len(tr)} test={len(te)} refk={refk} altk={altk}", flush=True); save()

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="sdpa", output_hidden_states=True).cuda().eval()
        n_layers = model.config.num_hidden_layers + 1
        res["n_hidden_layers"] = n_layers; save()

        def embed_all(seqs):
            acc = [[] for _ in range(n_layers)]
            t0 = time.time()
            for i in range(0, len(seqs), args.batch):
                chunk = [s.upper() for s in seqs[i:i + args.batch]]
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                          max_length=args.seqlen, add_special_tokens=False).to("cuda")
                m = enc["attention_mask"].unsqueeze(-1)
                with torch.no_grad():
                    hs = model(**enc).hidden_states
                for L in range(n_layers):
                    mm = m.to(hs[L].dtype)
                    acc[L].append(((hs[L] * mm).sum(1) / mm.sum(1).clamp(min=1)).float().cpu().numpy())
                if (i // args.batch) % 50 == 0:
                    print(f"  embed {i+len(chunk)}/{len(seqs)} {time.time()-t0:.0f}s", flush=True); save()
            return [np.concatenate(a, 0) for a in acc]

        RTr = embed_all(tr[refk]); res["ref_train_done"] = time.strftime("%T"); save()
        ATr = embed_all(tr[altk]); res["alt_train_done"] = time.strftime("%T"); save()
        RTe = embed_all(te[refk]); ATe = embed_all(te[altk]); res["embed_done"] = time.strftime("%T"); save()
        os.makedirs("/workspace/emb_cache", exist_ok=True)
        np.savez(cache, n_layers=n_layers, ytr=ytr, yte=yte,
                 **{f"rtr{L}": RTr[L] for L in range(n_layers)}, **{f"atr{L}": ATr[L] for L in range(n_layers)},
                 **{f"rte{L}": RTe[L] for L in range(n_layers)}, **{f"ate{L}": ATe[L] for L in range(n_layers)})
        res["emb_cache"] = "saved"; save()

    def feat(r, a):
        return np.concatenate([r, a, np.abs(r - a)], axis=1)   # concat(ref, alt, |ref-alt|)

    Xtr_L = [feat(RTr[L], ATr[L]) for L in range(n_layers)]
    Xte_L = [feat(RTe[L], ATe[L]) for L in range(n_layers)]

    if args.head in ("logreg", "both"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sub = np.random.default_rng(42).permutation(len(ytr))[:min(6000, len(ytr))]  # cheap linear baseline
        pl = []
        for L in range(n_layers):
            sc = StandardScaler().fit(Xtr_L[L][sub])
            clf = LogisticRegression(max_iter=500, C=1.0).fit(sc.transform(Xtr_L[L][sub]), ytr[sub])
            auc = roc_auc_score(yte, clf.predict_proba(sc.transform(Xte_L[L]))[:, 1])
            pl.append({"layer": L, "auc": round(float(auc), 4)}); res["logreg_per_layer"] = pl; save()
        b = max(pl, key=lambda d: d["auc"]); res["logreg_best"] = b
        print(f"logreg best L{b['layer']} AUC={b['auc']}", flush=True)

    if args.head in ("mlp", "both"):
        import torch.nn as nn
        torch.manual_seed(42)

        def train_mlp(Xtr, Xte):
            dev = "cuda"; xtr = torch.tensor(Xtr, dtype=torch.float32, device=dev)
            mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-6
            xtr = (xtr - mu) / sd
            xte = (torch.tensor(Xte, dtype=torch.float32, device=dev) - mu) / sd
            ytt = torch.tensor(ytr, dtype=torch.long, device=dev)
            net = nn.Sequential(nn.Linear(xtr.shape[1], 2048), nn.ReLU(), nn.Dropout(0.2),
                                nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.2),
                                nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.2),
                                nn.Linear(128, 2)).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-4); lossf = nn.CrossEntropyLoss()
            n, bs = len(ytt), 256
            for ep in range(args.epochs):
                net.train(); perm = torch.randperm(n, device=dev)
                for i in range(0, n, bs):
                    idx = perm[i:i + bs]; opt.zero_grad()
                    lossf(net(xtr[idx]), ytt[idx]).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                return torch.softmax(net(xte), 1).cpu().numpy()[:, 1]

        pl = []
        for L in range(n_layers):
            auc = roc_auc_score(yte, train_mlp(Xtr_L[L], Xte_L[L]))
            pl.append({"layer": L, "auc": round(float(auc), 4)}); res["mlp_per_layer"] = pl; save()
            print(f"  mlp L{L}: {pl[-1]['auc']}", flush=True)
        b = max(pl, key=lambda d: d["auc"]); res["mlp_best"] = b
        res["best_auc"] = b["auc"]; res["best_layer"] = b["layer"]
        res["delta_vs_target"] = round(b["auc"] - TARGET, 4)
        print(f"MLP best L{b['layer']} AUC={b['auc']} target={TARGET}", flush=True)

    res["finished"] = time.strftime("%F %T"); save()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        json.dump({"fatal": traceback.format_exc()}, open("/workspace/bench_fatal.json", "w"))
        print(traceback.format_exc(), flush=True)
