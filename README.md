# GENOS-Bench: Adversarially Verified Benchmark Reproduction

`genos-bench` (Phase 4) is the authoritative evaluation framework for the GENOS model family. Its primary mission is to **rigorously reproduce—not merely quote—leaderboard tasks**, providing an adversarially verified foundation for understanding model performance.

## Mission: Reproduce, Don't Quote

In an era of inflated benchmarks, `genos-bench` implements a "zero-trust" evaluation policy. Every benchmark is engineered to independently verify performance, systematically filtering out:
*   **Data Leakage**: Rigorous checks against train/test contamination.
*   **Metric Mismatch**: Standardizing definitions to avoid inflated metrics based on varying implementation details.
*   **Incorrect Splits**: Ensuring all evaluations respect established, robust data-splitting methodologies.

---

## Benchmark Matrix

| Benchmark | Task Description | Target Metric | Status |
| :--- | :--- | :--- | :--- |
| **ClinVar** | Variant Effect Prediction | > 0.92 AUC | Verified |
| **PopClass** | Population/Ancestry Classification | TBD | In-Progress |
| **RNA-seq** | Predictive Gene Expression | 0.9 Correlation | In-Progress |
| **DNALongBench**| Enhancer-Promoter/eQTL | TBD | In-Progress |
| **Cross-Species**| Zero-Shot/Fine-tuned Transfer | TBD | Planned |

---

## Architecture Components

The benchmark pipeline is decoupled into a feature extraction phase (`embed.py`) and an evaluation phase (`probe.py`).

### 1. Harness (`/harness`)
*   `embed.py`: Handles high-performance inference, mapping input sequences to latent representations.
*   `probe.py`: Trains and executes probing heads on the latent space to evaluate representation quality across model layers.

### 2. Task Definitions (`/tasks`)
Each task is modularized with specific loading logic:
*   Data preprocessing pipelines.
*   Adversarial split enforcement.
*   Task-specific metric calculators.

---

## Usage Guide

### Prerequisite Environment
This harness requires pre-computed model features or access to a running inference engine (see `genos-serve`).

### Running a Benchmark
The workflow consists of extracting features followed by training a probing head.

```bash
# 1. Extract features
python3 harness/embed.py --task clinvar --data ./data/clinvar --output ./features/clinvar

# 2. Run adversarial probe (with layer-sweep)
python3 harness/probe.py --task clinvar --feature_dir ./features/clinvar --layer_sweep 1-24
```

### Interpretation of Results
Results are generated in the `/results` directory. The harness produces:
1.  **Metric Report (`metrics.json`)**: Raw scores for each probe head.
2.  **Verification Log (`verification.log`)**: Output from the adversarial check pipeline (leakage alerts, distribution checks).

---

## Extending `genos-bench`

To add a new benchmark, adhere to the Phase 4 standards:

1.  **Define Task in `/tasks`**: Implement the data loader ensuring strict split adherence.
2.  **Adversarial Check**: Write a validation script that confirms the absence of train/test overlap before pipeline execution.
3.  **Harness Integration**: Ensure the task defines an compatible output mapping for the `probe.py` harness.
4.  **Verification**: Submit proof-of-reproducibility with your pull request, demonstrating alignment with established leaderboard figures within standard deviation tolerances.
