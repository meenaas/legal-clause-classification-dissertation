# LLM-Based Summarisation and Clause Classification for Legal Contracts

MSc Data Science dissertation project (University of Surrey, 2024–2026, Distinction) combining hybrid summarisation and multi-label clause classification on the [CUAD](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus Dataset) legal contract corpus.

**Author:** Raja Meenakshi Shanmuga Sundaram
**Supervisor:** Dr Alaa Marshan, University of Surrey

## ⚠️ Status note

This repo contains the code and results as reported in the original MSc dissertation (Aug/Sep 2025). During follow-on research for a peer-reviewed submission, a **data-leakage issue in the evaluation pipeline** was independently identified and corrected — the correction reversed a previously favourable result. That corrected work is written up separately and is currently under double-blind review, so it is **not included in this repo** while review is ongoing.

This repo is kept as-is (dissertation-stage) for transparency and reproducibility of what was originally reported. If you're looking at this as part of a PhD application review: the leakage discovery and correction is the more interesting research contribution, and I can point you to it once the review period allows public disclosure.

## Repository structure

```
.
├── notebooks/
│   ├── Clause_Classification_Legal_Pro_Bert_Final.ipynb        # Stage 2: clause classification
│   ├── Hybrid_Summarisation_Pipeline_1908.ipynb                # Stage 1: hybrid summarisation
│   └── Hybrid_Summarisation_Clause_Classification.ipynb        # Integration: joins Stage 1 + Stage 2, full evaluation
├── alternatives_explored/
│   ├── LegalLongformer2208.ipynb                                # Legal-Longformer classifier (considered, not adopted)
│   └── extractive_summarisation_pipeline.ipynb                  # Supervised Longformer extractive ranker (considered, not adopted)
├── requirements.txt
└── README.md
```

### Why three notebooks, not one

The dissertation pipeline (Figure 3.1-1) has two stages that were developed and evaluated independently, then joined:

1. **`Hybrid_Summarisation_Pipeline_1908.ipynb`** — Stage 1. Combines extractive (TextRank) and abstractive (Legal-Pegasus) summarisation into several hybrid variants (`hybrid_blend`, `hybrid_mmr_dyn`, `hybrid_aug`, `hybrid_strong`), evaluated with SBERT-based precision/recall/F1 (Table 5.1-1).
2. **`Clause_Classification_Legal_Pro_Bert_Final.ipynb`** — Stage 2. Trains and compares Legal-BERT and LegalPro-BERT under different loss functions and optimisers (Table 5.2-1), then tunes per-class thresholds (Table 5.2-2). Saves the best checkpoint (`legalpro_bert_clause_cls_best.pt`) and label map (`clause_to_idx.json`) used by the integration stage.
3. **`Hybrid_Summarisation_Clause_Classification.ipynb`** — Integration. Loads the Stage 2 checkpoint, classifies the Stage 1 hybrid summaries (with overlapping-chunk handling for long contracts), and computes the full downstream evaluation: Clause Coverage Score (CCS), Recall of Critical Clauses (RCR_critical), ROUGE, and BERTScore (Tables 5.3-2 through 5.3-4).

### Alternatives explored, not adopted

Two model variants were trained and evaluated but explicitly not adopted in the final pipeline, as discussed in dissertation §3.2:

- **Legal-Longformer** (`alternatives_explored/LegalLongformer2208.ipynb`) was considered as a clause classifier for its longer context window (4,096 tokens) but not adopted, since CUAD clause spans typically fit within 512 tokens and the computational cost of long-sequence training was not justified.
- A **supervised Longformer-based extractive ranker** (`alternatives_explored/extractive_summarisation_pipeline.ipynb`) was trialled as an alternative to TextRank for the extractive stage, but TextRank was retained in the final pipeline as it required no training data and performed comparably for this dataset size.

These are included for transparency, not as part of the reported results.

## Key results

**Clause classification (validation set, Table 5.2-1):**

| Model | Loss | Optimiser | Micro-F1 |
|---|---|---|---|
| LegalPro-BERT | BCEWithLogitsLoss | AdamW | **0.7201** |
| LegalBERT | Focal Loss | AdamW | 0.6251 |
| LegalBERT | Focal Loss | Adagrad | 0.6133 |
| LegalPro-BERT | Focal Loss | Adagrad | 0.5908 |
| LegalPro-BERT | Focal Loss | AdamW | 0.5907 |
| LegalBERT | BCEWithLogitsLoss | AdamW | 0.5474 |
| LegalBERT | BCEWithLogitsLoss | Adagrad | 0.5075 |

LegalPro-BERT + BCEWithLogitsLoss + AdamW was adopted as the default model for the integration pipeline.

**Integration pipeline (Table 5.3-2):** all summarisation variants preserved 100% of critical clauses (CSS = 1.0, RCR_critical = 1.0). Hybrid Blend achieved the best practical trade-off, compressing contracts to ~12% of their original length while retaining full clause coverage.

*Note: the 0.72 micro-F1 above is a validation-set figure from this dissertation-stage pipeline. It should not be confused with the leakage-corrected full-text micro-F1 reported separately — see the status note above.*

## Data

This repo does **not** include `CUAD_v1.json`. Download it from the official source:
[https://www.atticusprojectai.org/cuad](https://www.atticusprojectai.org/cuad)

Place it in the working directory (or update the path in each notebook) before running. Each notebook expects `CUAD_v1.json` in its own runtime working directory — see the load cell near the top of each notebook.

## Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
```

Notebooks were developed and run on Google Colab Pro (single NVIDIA T4 GPU, High-RAM mode). GPU is recommended for both training and inference cells.

## Reproducibility

- Random seed fixed to 42 for data splits and sampling throughout.
- Trained artefacts (`legalpro_bert_clause_cls_best.pt`, `clause_to_idx.json`) are produced by `Clause_Classification_Legal_Pro_Bert_Final.ipynb` and consumed by `Hybrid_Summarisation_Clause_Classification.ipynb` — run notebooks in the order listed above.
- Full methodology, hyperparameters, and evaluation protocol are documented in the accompanying dissertation report (not included in this repo; available on request).

## Citation

If referencing this work, please cite the dissertation:

> Shanmuga Sundaram, R.M. (2025). *LLM-based Summarisation and Clause Classification*. MSc Dissertation, University of Surrey. Supervised by Dr Alaa Marshan.
