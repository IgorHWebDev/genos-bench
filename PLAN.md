# Phase 4 — Benchmark reproduction plan (verified recipes)

_From the genos-bench-plan workflow (10/11 agents; synthesis agent died on an API hiccup, recipes intact)._

**Verdict:** nothing is runnable-now out-of-the-box; the reproducible tasks are **needs-data-assembly** (public HF datasets, re-cut to windows). RNA-seq full-fine-tune and CPC mutation-hotspot are **blocked** (heavy / private data).

## Execution order (cheapest first)

1. **NT H3 histone** (target AUC 0.940) — 15K short seqs, public HF, <10 GPU-min → harness smoke test.
2. NT H3K36me3 (0.766), splice_sites_all (0.799), open-chromatin (0.762) — same pattern.
3. **ClinVar variant** (AUC 0.9326) — headline; 62K×8192bp forwards, LRB dataset + GRCh38, layer sweep.
4. LRB enhancer/promoter/eQTL (0.75/0.92/0.68) — 8K-bp windows.

**Blocked:** RNA-seq coverage (full fine-tune), CPC mutation-hotspot (private Chinese Pangenome data).

## Recipes (from map agents)

### Long-Range Benchmark suite (enhancer/promoter/causal-eQTL)

- **Enhancer classification (regulatory_element_enhancer_8K) — binary enhancer present/absent at an 8kb ** — target ROC-AUC 0.7532
  - dataset: InstaDeepAI/genomics-long-range-benchmark, task_name='regulatory_element_enhancer' (a.k.a. enhancer). Obtain: `load_dataset("InstaDeepAI/genomics-long-range-benchmark", task_name="regulatory_element_enhancer", sequence_length=8192)`. Re-cut to 8192 bp centered
  - head: benchmarks/evaluation.py MLP head: 3 hidden layers (input_dim/2, /4, /8, capped ~128), ReLU + Dropout(0.2), Adam, lr 1e-4, ~100 epochs, batc · runnable: partial
  - blockers: LRB tasks are commented out in the shipped config.yaml — must be re-enabled; No dataset files bundled; must run HF load_dataset and re-cut to 8192bp then export JSONL yourself; Exact winning layer for the 0.7532 number is UNVERIFIED (shipped default layer_to_eval=[12]); reproducing may require a ful
- **Promoter classification (regulatory_element_promoter_8K) — binary promoter present/absent at an 8kb ** — target ROC-AUC 0.9249
  - dataset: InstaDeepAI/genomics-long-range-benchmark, task_name='regulatory_element_promoter'. `load_dataset(..., task_name="regulatory_element_promoter", sequence_length=8192)`. HF split: train chr1-7,10-22; test chr8-9; GENOS reserves chr22 as eval. Export JSONL {"seq"
  - head: evaluation.py MLP (3 hidden, input/2,/4,/8, ReLU, dropout 0.2, Adam lr 1e-4, ~100 epochs, batch 64) or XGBoost on frozen 4096-d embeddings;  · runnable: partial
  - blockers: Task commented out in shipped config.yaml; Dataset must be assembled from HF and re-cut to 8192bp -> JSONL; Winning layer for 0.9249 UNVERIFIED (default [12]); MLP vs XGBoost for reported number UNVERIFIED
- **Causal-eQTL variant effect (variant_effect_causal_eqtl_8K) — binary: does the SNP causally affect ge** — target ROC-AUC 0.6773
  - dataset: InstaDeepAI/genomics-long-range-benchmark, task_name='variant_effect_causal_eqtl'. `load_dataset(..., task_name="variant_effect_causal_eqtl", sequence_length=8192)`. Each sample = SNP-centered 8192bp reference sequence + alternate-allele sequence. HF split: tr
  - head: evaluation.py MLP (input_dim now ~8192 after concat; 3 hidden input/2,/4,/8, ReLU, dropout 0.2, Adam lr 1e-4, ~100 epochs, batch 64) or XGBo · runnable: partial
  - blockers: Task commented out in shipped config.yaml; Dataset must be assembled from HF (paired ref/alt) and re-cut to 8192bp -> paired JSONL; Exact ref/alt embedding combination for the 0.6773 number is UNVERIFIED (concat is config default; could be difference); Winning layer UNVERIFIED (default [12]); MLP vs

### RNA-seq coverage track (headline log1p-Pearson ~0.93)

- **RNA-seq single-base coverage track prediction (regression), full fine-tune of Genos-1.2B with a 3-la** — target log1p-Pearson 0.9335 (GM12878, whole genome, + strand); 0.9334 gene region; 0.8641 gene-expression matrix. NK cells: 0.9084–0.9267. (Source: PMC12755919 Results)
  - dataset: Single-base RNA-seq coverage tracks derived from ENCODE + GTEx; paper reports 667 metadata groups of single-base transcriptome samples covering both + and - strands. NOT shipped in repo — you must assemble it yourself. For GM12878 (=Human B Lymphocyte, EFO/CL-
  - head: Three 1D conv layers with (kernel, padding, dilation) = (3,1,1), (3,2,2), (1,0,1); channel progression 1024->256, 256->64, 64->1. Each layer · runnable: no
  - blockers: Training code NOT open-sourced: notebook 04 is inference-only (client.rna_coverage_track_pred hosted API); SDK docs state RNA-seq model 'not open-source and cannot be self-hosted'. Entire training/data-prep harness must be written from the paper prose.; Trained conv-head weights are not released — c

### Genomic-element classification suite (coding/splice/histone/chromatin/hotspot)

- **Coding-vs-noncoding (binary coding potential)** — target ROC-AUC 0.9914
  - dataset: Genomic Benchmarks task 'demo_coding_vs_intergenomic_seqs'. Obtain via `pip install genomic-benchmarks` then `from genomic_benchmarks.loc2seq import download_dataset; download_dataset('demo_coding_vs_intergenomic_seqs')` (also mirrored at HF katielink/genomic-
  - head: Default MLP (benchmarks/evaluation.py train_mlp_classifier / MLPClassifier): Linear(4096->2048)->ReLU->Dropout(0.2)->Linear(2048->1024)->ReL · runnable: partial
  - blockers: Dataset not shipped — must pip-install genomic-benchmarks and download, then convert its class-folder/fasta layout into the repo's expected item dict with seq_key+label_key.; config.yaml ships listing ONLY Human_classify_* tasks; must add demo_coding_vs_intergenomic_seqs (it IS defined in datasets_i
- **Histone mark H3 occupancy (binary)** — target ROC-AUC 0.940
  - dataset: Nucleotide Transformer downstream task 'H3'. HF: InstaDeepAI/nucleotide_transformer_downstream_tasks (or _revised). datasets_info.yaml: 1 seq/item, ~290-500bp, 14,965 samples, 2 classes, 90/10 train/test. NOT repo-shipped.
  - head: Default MLP (4096->2048->1024->128->2), CrossEntropyLoss, Adam lr=1e-4, 100 epochs. · runnable: partial
  - blockers: Load 'H3' split from HF InstaDeepAI/nucleotide_transformer_downstream_tasks(_revised) and convert to repo item format (seq_key/label_key).; Add 'H3' to active dataset list in config.yaml (defined in datasets_info.yaml).; Layer index / model dtype caveats as above.
- **Histone mark H3K36me3 occupancy (binary)** — target ROC-AUC 0.766 (paper table 0.7658)
  - dataset: Nucleotide Transformer downstream task 'H3K36me3'. HF InstaDeepAI/nucleotide_transformer_downstream_tasks(_revised). datasets_info.yaml: 1 seq/item, ~310-500bp (NTB nominal 1000bp), 34,880 samples, 2 classes, 90/10 split. NOT repo-shipped.
  - head: Default MLP (4096->2048->1024->128->2), CrossEntropy, Adam lr=1e-4, 100 epochs. · runnable: partial
  - blockers: Load 'H3K36me3' from HF NTB dataset and convert to repo item format.; Add task to config.yaml active list.; Layer/dtype caveats as above. Lower absolute AUC (~0.77) means metric is more sensitive to layer choice — worth sweeping layers.
- **Splice site prediction (3-class: donor/acceptor/negative)** — target ROC-AUC 0.799 (macro OvR)
  - dataset: Nucleotide Transformer downstream task 'splice_sites_all'. HF InstaDeepAI/nucleotide_transformer_downstream_tasks(_revised): 30,000 train / 3,000 test, 3 labels, 400-600bp. datasets_info.yaml mirrors: 1 seq/item, 400bp fixed, 30,000 samples, 3 classes, 90/10 s
  - head: Default MLP with num_classes=3 output (4096->2048->1024->128->3), CrossEntropyLoss, Adam lr=1e-4, 100 epochs. · runnable: partial
  - blockers: Load 'splice_sites_all' from HF NTB and convert to repo item format; ensure 3-class labels map correctly and num_classes=3 flows to the MLP output layer and macro-OvR AUC.; Add task to config.yaml active list.; Confirm repo 400bp window vs HF 600bp — window mismatch can shift AUC (UNVERIFIED which t
- **Open chromatin region detection (binary OCR)** — target ROC-AUC 0.762 (paper 0.7623)
  - dataset: Genomic Benchmarks task 'human_ocr_ensembl'. `pip install genomic-benchmarks`; download_dataset('human_ocr_ensembl') (also HF katielink/genomic-benchmarks). datasets_info.yaml: 1 seq/item, variable 71-593bp, 174,756 samples, 2 classes, 80/20 split (~69.9K per 
  - head: Default MLP (4096->2048->1024->128->2), CrossEntropy, Adam lr=1e-4, 100 epochs. · runnable: partial
  - blockers: Dataset not shipped — download via genomic-benchmarks package, convert class-folder layout to repo item format.; Add 'human_ocr_ensembl' to config.yaml active list.; Variable-length (71-593bp) padding: masked mean-pool handles it but very short 71bp seqs may need min-length handling.; Layer/dtype ca
- **Mutation-hotspot classification (context-length scaling, up to 128kb)** — target ROC-AUC up to 0.9911 at 131072bp (CPC_8192 0.9522, CPC_32768 0.9625, CPC_131072 0.9911; paper also frames as 3-class Human_classify_* at 8K/32K/128K)
  - dataset: Chinese Pangenome Consortium (CPC) / Chinese Pangenome data — NOT a public HF/NCBI/ENCODE dataset and NOT shipped in the repo. datasets_info.yaml defines two families: (a) CPC_8192/32768/131072 = binary hotspot, 62,729 / 15,391 / 3,825 samples, 94/5/1 split; (
  - head: For Human_classify_*: XGBoost (config override) — n_estimators=100, lr=0.1, max_depth=6, predict_proba (evaluation.py train_xgboost_classifi · runnable: no
  - blockers: CPC / Chinese Pangenome variation data is NOT publicly downloadable and NOT shipped in the repo — hard data blocker; cannot reproduce without access request to the Chinese Pangenome Consortium.; Hotspot label generation (Poisson right-tail test vs chromosomal background) is described in the paper bu

### ClinVar pathogenic variant effect (headline AUC 0.9326)

- **ClinVar pathogenic-vs-benign variant effect classification (repo task id: variant_effect_pathogenic_** — target ROC-AUC 0.9326 (test split, best layer; binary, positive class = Pathogenic/label 1)
  - dataset: Source = InstaDeepAI/genomics-long-range-benchmark (HF) task 'variant_effect_pathogenic_clinvar' (processed from GPN-MSA paper files; positives = ClinVar pathogenic SNVs, negatives = gnomAD common variants MAF>5%; reference GRCh38/hg38). Native LRB split is ch
  - head: MLP (repo default classifer_type: 'MLP'; ClinVar has no per-dataset override, unlike XGB/RF tasks). benchmarks/evaluation.py: input_dim = 2× · runnable: partial
  - blockers: Genos-10B weights are NOT in the repo (gated / cached on volume genos-clu-weights per dossier); model_path in config.yaml is a placeholder ('model_path'). Cannot run extraction without obtaining weights from BGI-HangzhouAI.; The preprocessed 8192bp ref_seq/var_seq dataset is NOT shipped; dataset_pat

### test

- **ClinVar** — target AUC 0.9326
  - dataset: LRB clinvar 8192
  - head: MLP · runnable: partial
  - blockers: no data shipped
- **H3** — target AUC 0.94
  - dataset: NT H3
  - head: MLP · runnable: partial
  - blockers: no data shipped

