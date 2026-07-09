"""Nucleotide-Transformer downstream benchmark on GENOS (frozen encoder + linear/MLP probe).

Faithful protocol: extract ALL 13 hidden-layer mean-pooled embeddings, sweep a probe per
layer, report the BEST layer's test AUC (the GENOS paper's numbers are max-over-layers).

    python nt_bench.py --task H3          # target ROC-AUC 0.940
    python nt_bench.py --task splice_sites_all --classes 3

Writes /workspace/bench_<task>_result.json (served on :8000).
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
    args = ap.parse_args()
    out = f"/workspace/bench_{args.task}_result.json"
    res = {"task": args.task, "target_auc": TARGETS.get(args.task), "started": time.strftime("%F %T")}

    def save():
        json.dump(res, open(out, "w"), indent=2)

    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    # dataset is a single 'default' config with a 'task' column -> load + filter by task.
    # original repo has plain histone names (H3, H4...); revised uses ENCODE names.
    tr = te = None
    for repo in ("InstaDeepAI/nucleotide_transformer_downstream_tasks",
                 "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"):
        try:
            d = load_dataset(repo)  # columns: sequence, name, label, task
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
    print(f"{args.task}: train={len(tr)} test={len(te)} repo={res['dataset_repo']}", flush=True)
    save()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", output_hidden_states=True,
    ).cuda().eval()
    n_layers = model.config.num_hidden_layers + 1  # +embedding layer
    res["n_hidden_layers"] = n_layers

    def embed_all_layers(seqs):
        """returns list[L] of (N x 4096) mean-pooled per hidden layer."""
        acc = [[] for _ in range(n_layers)]
        t0 = time.time()
        for i in range(0, len(seqs), args.batch):
            chunk = [s.upper() for s in seqs[i:i + args.batch]]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.maxlen, add_special_tokens=False).to("cuda")
            mask = enc["attention_mask"].unsqueeze(-1)
            with torch.no_grad():
                hs = model(**enc).hidden_states       # tuple len n_layers
            for L in range(n_layers):
                m = mask.to(hs[L].dtype)
                pooled = (hs[L] * m).sum(1) / m.sum(1).clamp(min=1)
                acc[L].append(pooled.float().cpu().numpy())
            if (i // args.batch) % 25 == 0:
                print(f"  embed {i+len(chunk)}/{len(seqs)} {time.time()-t0:.0f}s", flush=True)
        return [np.concatenate(a, 0) for a in acc]

    seq_key = "sequence" if "sequence" in tr.column_names else tr.column_names[0]
    lab_key = "label" if "label" in tr.column_names else tr.column_names[-1]
    ytr = np.array([int(x) for x in tr[lab_key]])   # labels are strings '0'/'1'
    yte = np.array([int(x) for x in te[lab_key]])
    print(f"columns={tr.column_names} seq_key={seq_key} lab_key={lab_key}", flush=True)

    Xtr_L = embed_all_layers(tr[seq_key]); res["embed_train_done"] = time.strftime("%T"); save()
    Xte_L = embed_all_layers(te[seq_key]); res["embed_test_done"] = time.strftime("%T"); save()

    # per-layer logistic-regression probe -> AUC (sweep, report best)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    multiclass = args.classes > 2
    per_layer = []
    for L in range(n_layers):
        sc = StandardScaler().fit(Xtr_L[L])
        clf = LogisticRegression(max_iter=2000, C=1.0)  # sklearn>=1.7 removed multi_class (auto-multinomial)
        clf.fit(sc.transform(Xtr_L[L]), ytr)
        if multiclass:
            proba = clf.predict_proba(sc.transform(Xte_L[L]))
            auc = roc_auc_score(yte, proba, multi_class="ovr", average="macro")
        else:
            proba = clf.predict_proba(sc.transform(Xte_L[L]))[:, 1]
            auc = roc_auc_score(yte, proba)
        per_layer.append({"layer": L, "auc": round(float(auc), 4)})
        print(f"  layer {L}: AUC={auc:.4f}", flush=True)
        res["per_layer"] = per_layer; save()

    best = max(per_layer, key=lambda d: d["auc"])
    res["best_layer"] = best["layer"]
    res["best_auc"] = best["auc"]
    res["delta_vs_target"] = round(best["auc"] - (TARGETS.get(args.task) or 0), 4)
    res["finished"] = time.strftime("%F %T")
    save()
    print(f"BEST layer {best['layer']} AUC={best['auc']} target={TARGETS.get(args.task)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        try:
            r = json.load(open(f"/workspace/bench_result_err.json", "w"))
        except Exception:
            pass
        json.dump({"fatal": tb}, open("/workspace/bench_fatal.json", "w"))
        print(tb, flush=True)
