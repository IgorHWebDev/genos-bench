"""Nucleotide-Transformer downstream benchmark on GENOS (frozen encoder + probe).

Faithful protocol: extract ALL 13 hidden-layer mean-pooled embeddings, sweep a probe
per layer, report the BEST layer's test AUC (GENOS's published numbers are max-over-layers).
Two heads: logreg (fast linear baseline) and mlp (the paper's 4096->2048->1024->128->C head).
Embeddings are cached to the network volume so head iterations skip re-extraction.

    python nt_bench.py --task H3 --head mlp     # target ROC-AUC 0.940
"""
import argparse, json, os, time, traceback

import numpy as np

TARGETS = {"H3": 0.940, "H3K36me3": 0.766, "splice_sites_all": 0.799,
           "enhancers": None, "promoter_all": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="H3")
    ap.add_argument("--model", default=os.environ.get("GENOS_PATH", "/workspace/genos-10b-v2"))
    ap.add_argument("--maxlen", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--classes", type=int, default=2)
    ap.add_argument("--head", default="both", choices=["logreg", "mlp", "both"])
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    out = f"/workspace/bench_{args.task}_result.json"
    res = {"task": args.task, "target_auc": TARGETS.get(args.task), "head": args.head,
           "started": time.strftime("%F %T")}

    def save():
        json.dump(res, open(out, "w"), indent=2)

    import torch
    from sklearn.metrics import roc_auc_score
    multiclass = args.classes > 2
    cache = f"/workspace/emb_cache/{args.task}.npz"

    # ---- embeddings (cached on the volume) ----
    if os.path.exists(cache):
        z = np.load(cache)
        n_layers = int(z["n_layers"])
        Xtr_L = [z[f"tr{L}"] for L in range(n_layers)]
        Xte_L = [z[f"te{L}"] for L in range(n_layers)]
        ytr, yte = z["ytr"], z["yte"]
        res["emb_cache"] = "hit"; res["n_hidden_layers"] = n_layers
        print(f"emb cache HIT {cache}", flush=True)
    else:
        from datasets import load_dataset
        from transformers import AutoModel, AutoTokenizer
        tr = te = None
        for repo in ("InstaDeepAI/nucleotide_transformer_downstream_tasks",
                     "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"):
            try:
                d = load_dataset(repo)
                avail = sorted(set(d["train"]["task"]))
                if args.task not in avail:
                    res.setdefault("available_tasks", {})[repo.split("/")[-1]] = avail
                    continue
                tr = d["train"].filter(lambda r: r["task"] == args.task)
                te = d["test"].filter(lambda r: r["task"] == args.task)
                res["dataset_repo"] = repo
                break
            except Exception as e:
                res["load_error_" + repo.split("/")[-1]] = str(e)[:200]
        if tr is None:
            save(); print("task not found; see available_tasks", flush=True); return
        res["n_train"], res["n_test"] = len(tr), len(te)
        save()
        sk = "sequence" if "sequence" in tr.column_names else tr.column_names[0]
        lk = "label" if "label" in tr.column_names else tr.column_names[-1]
        ytr = np.array([int(x) for x in tr[lk]]); yte = np.array([int(x) for x in te[lk]])

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
                          max_length=args.maxlen, add_special_tokens=False).to("cuda")
                m = enc["attention_mask"].unsqueeze(-1)
                with torch.no_grad():
                    hs = model(**enc).hidden_states
                for L in range(n_layers):
                    mm = m.to(hs[L].dtype)
                    acc[L].append(((hs[L] * mm).sum(1) / mm.sum(1).clamp(min=1)).float().cpu().numpy())
                if (i // args.batch) % 40 == 0:
                    print(f"  embed {i+len(chunk)}/{len(seqs)} {time.time()-t0:.0f}s", flush=True)
            return [np.concatenate(a, 0) for a in acc]

        Xtr_L = embed_all(tr[sk]); res["embed_train_done"] = time.strftime("%T"); save()
        Xte_L = embed_all(te[sk]); res["embed_test_done"] = time.strftime("%T"); save()
        os.makedirs("/workspace/emb_cache", exist_ok=True)
        np.savez(cache, n_layers=n_layers, ytr=ytr, yte=yte,
                 **{f"tr{L}": Xtr_L[L] for L in range(n_layers)},
                 **{f"te{L}": Xte_L[L] for L in range(n_layers)})
        res["emb_cache"] = "saved"; save()

    def auc_of(proba):
        if multiclass:
            return roc_auc_score(yte, proba, multi_class="ovr", average="macro")
        return roc_auc_score(yte, proba[:, 1])

    # ---- logreg sweep ----
    if args.head in ("logreg", "both"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        pl = []
        for L in range(n_layers):
            sc = StandardScaler().fit(Xtr_L[L])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr_L[L]), ytr)
            pl.append({"layer": L, "auc": round(float(auc_of(clf.predict_proba(sc.transform(Xte_L[L])))), 4)})
            res["logreg_per_layer"] = pl; save()
        b = max(pl, key=lambda d: d["auc"])
        res["logreg_best"] = {"layer": b["layer"], "auc": b["auc"]}
        print(f"logreg best L{b['layer']} AUC={b['auc']}", flush=True)

    # ---- MLP head sweep (the paper's head) ----
    if args.head in ("mlp", "both"):
        import torch.nn as nn
        torch.manual_seed(42)
        nc = int(max(ytr.max(), yte.max())) + 1

        def train_mlp(Xtr, ytr_, Xte):
            dev = "cuda"
            xtr = torch.tensor(Xtr, dtype=torch.float32, device=dev)
            mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-6
            xtr = (xtr - mu) / sd
            xte = (torch.tensor(Xte, dtype=torch.float32, device=dev) - mu) / sd
            ytt = torch.tensor(ytr_, dtype=torch.long, device=dev)
            net = nn.Sequential(
                nn.Linear(xtr.shape[1], 2048), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, nc)).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-4)
            lossf = nn.CrossEntropyLoss()
            n, bs = len(ytt), 256
            for ep in range(args.epochs):
                net.train()
                perm = torch.randperm(n, device=dev)
                for i in range(0, n, bs):
                    idx = perm[i:i + bs]
                    opt.zero_grad(); loss = lossf(net(xtr[idx]), ytt[idx]); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                return torch.softmax(net(xte), 1).cpu().numpy()

        pl = []
        t0 = time.time()
        for L in range(n_layers):
            pl.append({"layer": L, "auc": round(float(auc_of(train_mlp(Xtr_L[L], ytr, Xte_L[L]))), 4)})
            res["mlp_per_layer"] = pl; save()
            print(f"  mlp L{L}: AUC={pl[-1]['auc']}  ({time.time()-t0:.0f}s)", flush=True)
        b = max(pl, key=lambda d: d["auc"])
        res["mlp_best"] = {"layer": b["layer"], "auc": b["auc"]}
        res["best_auc"] = b["auc"]; res["best_layer"] = b["layer"]
        res["delta_vs_target"] = round(b["auc"] - (TARGETS.get(args.task) or 0), 4)
        print(f"MLP best L{b['layer']} AUC={b['auc']} target={TARGETS.get(args.task)}", flush=True)

    res["finished"] = time.strftime("%F %T"); save()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        json.dump({"fatal": traceback.format_exc()}, open("/workspace/bench_fatal.json", "w"))
        print(traceback.format_exc(), flush=True)
