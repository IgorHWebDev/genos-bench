"""Shared downstream-probe + metric — trains a lightweight head on GENOS embeddings.

Matches the GENOS "linear probe / lightweight head" protocol. Classification -> ROC-AUC;
regression -> Pearson. Supports a held-out split by group (e.g. chr22) or a random split.

    python probe.py --emb emb/clinvar --task clf --head logreg --split random --test_frac 0.2
    python probe.py --emb emb/lrb_enhancer --task clf --head mlp --groups data/lrb.groups.json --holdout chr22
"""
import argparse, json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="prefix: loads <emb>.npy + <emb>.labels.npy")
    ap.add_argument("--task", default="clf", choices=["clf", "reg"])
    ap.add_argument("--head", default="logreg", choices=["logreg", "mlp", "xgb", "ridge"])
    ap.add_argument("--split", default="random", choices=["random", "group"])
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--groups", help="json list parallel to rows (e.g. chromosome per sample)")
    ap.add_argument("--holdout", help="group value reserved for test (e.g. chr22)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X = np.load(args.emb + ".npy")
    y = np.load(args.emb + ".labels.npy", allow_pickle=True)
    print(f"X={X.shape} y={y.shape}")

    if args.split == "group":
        groups = np.array(json.load(open(args.groups)))
        te = groups == args.holdout
        tr = ~te
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(y))
        ncut = int(len(y) * args.test_frac)
        te_idx = perm[:ncut]
        tr = np.ones(len(y), bool); tr[te_idx] = False; te = ~tr
    print(f"train={tr.sum()} test={te.sum()}")

    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    if args.task == "clf":
        ytr = ytr.astype(int); yte = yte.astype(int)
        if args.head == "logreg":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xtr, ytr); score = clf.predict_proba(Xte)[:, 1]
        elif args.head == "mlp":
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=args.seed)
            clf.fit(Xtr, ytr); score = clf.predict_proba(Xte)[:, 1]
        else:  # xgb
            from xgboost import XGBClassifier
            clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=args.seed)
            clf.fit(Xtr, ytr); score = clf.predict_proba(Xte)[:, 1]
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = roc_auc_score(yte, score); ap_ = average_precision_score(yte, score)
        print(json.dumps({"metric": "ROC-AUC", "auc": round(float(auc), 4),
                          "auprc": round(float(ap_), 4), "n_test": int(te.sum()),
                          "head": args.head, "split": args.split}, indent=2))
    else:
        from sklearn.linear_model import Ridge
        from scipy.stats import pearsonr
        reg = Ridge(alpha=1.0); reg.fit(Xtr, ytr.astype(float))
        pred = reg.predict(Xte)
        r, _ = pearsonr(pred, yte.astype(float))
        print(json.dumps({"metric": "Pearson", "pearson": round(float(r), 4),
                          "n_test": int(te.sum()), "head": args.head}, indent=2))


if __name__ == "__main__":
    main()
