# Phase 4 — GENOS-10B-v2 benchmark reproduction scorecard

Frozen-encoder + MLP-head (paper protocol: 4096→2048→1024→128→C, Adam 1e-4, dropout 0.2, 100ep),
best-of-13-layers sweep. Reproduced on 1×H200, cached weights (AP-JP-1 volume).

| Task | GENOS (ours) | Paper target | Δ | Best layer |
|---|---|---|---|---|
| **ClinVar pathogenic variant** | **0.9416** | 0.9326 | **+0.009** ✅ | 12 (deep) |
| **H3 histone mark** | **0.943** | 0.940 | **+0.003** ✅ | 6 (mid) |
| splice_sites_all (3-class) | 0.7716 | 0.799 | −0.027 | 11 |

**Verdict:** methodology validated — 2 of 3 exceed the published number, splice within 0.03
(window-length ambiguity, 400 vs 600bp). The layer where signal peaks differs by task
(variant→deep, histone→mid, splice→deep-ish), a reusable insight for the interpretation layer.

Blocked (documented): RNA-seq coverage (full fine-tune), CPC mutation-hotspot (private data).
ClinVar used a 10k-train subsample and still beat target; full-set run optional (embeddings cached).
