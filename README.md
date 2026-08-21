# LLM-Based Summarisation and Clause Classification for Legal Contracts

MSc Data Science dissertation project (University of Surrey, 2025–2026, Distinction). Combines hybrid summarisation and multi-label clause classification on CUAD, the Contract Understanding Atticus Dataset.

**Author:** Raja Meenakshi Shanmuga Sundaram
**Supervisor:** Dr Alaa Marshan, University of Surrey

## Status note

This is the code and results as they stood in the original MSc dissertation (Aug/Sep 2025). While working on a follow-up peer-reviewed submission, I found a data-leakage issue in the evaluation pipeline and corrected it. The fix actually reversed a result that had originally looked favourable. That corrected work is written up separately and is currently under double-blind review, so it isn't in this repo yet.

I'm keeping this repo as-is, leakage and all, so it stays an honest record of what was originally reported. If you're reviewing this as part of a PhD application: the leakage discovery is genuinely the more interesting part of the story, and I'm happy to share more on that once the review period is over.

## Repository structure

notebooks/
alternatives_explored/
pipeline_out/
requirements.txt
README.md

### Why three notebooks

The pipeline (Figure 3.1-1 in the dissertation) has two stages that were built and evaluated separately, then joined together:

1. **`Hybrid_Summarisation_Pipeline_1908.ipynb`** – Stage 1. Combines extractive (TextRank) and abstractive (Legal-Pegasus) summarisation into a few hybrid variants (`hybrid_blend`, `hybrid_mmr_dyn`, `hybrid_aug`, `hybrid_strong`), evaluated with SBERT-based precision/recall/F1 (Table 5.1-1).
2. **`Clause_Classification_Legal_Pro_Bert_Final.ipynb`** – Stage 2. Trains and compares Legal-BERT and LegalPro-BERT under different loss functions and optimisers (Table 5.2-1), then tunes per-class thresholds (Table 5.2-2). Saves the best checkpoint (`legalpro_bert_clause_cls_best.pt`) and label map (`clause_to_idx.json`) that the integration stage needs.
3. **`Hybrid_Summarisation_Clause_Classification.ipynb`** – Integration. Loads the Stage 2 checkpoint, classifies the Stage 1 summaries (with chunk handling for long contracts), and runs the full downstream evaluation: Clause Coverage Score (CCS), Recall of Critical Clauses (RCR_critical), ROUGE, and BERTScore (Tables 5.3-2 through 5.3-4).

### Alternatives explored, not adopted

Two things I tried and didn't end up using, as discussed in 3.2 of the dissertation:

- **Legal-Longformer** (`alternatives_explored/LegalLongformer2208.ipynb`) as a clause classifier, for its longer context window (4,096 tokens). I didn't adopt it because CUAD clause spans mostly fit under 512 tokens anyway, and the extra training cost wasn't worth it for this dataset.
- A **supervised Longformer extractive ranker** (`alternatives_explored/extractive_summarisation_pipeline.ipynb`), tried as an alternative to TextRank. TextRank stayed in the final pipeline since it needs no training data and did just as well at this dataset size.

Keeping these in for transparency, not because they're part of the reported results.

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

LegalPro-BERT + BCEWithLogitsLoss + AdamW ended up as the default model for the integration pipeline.

**Integration pipeline (Table 5.3-2):** every summarisation variant kept 100% of critical clauses (CSS = 1.0, RCR_critical = 1.0). Hybrid Blend gave the best practical trade-off, compressing contracts to around 12% of their original length while still holding onto full clause coverage.

Raw summary CSVs behind these tables are in [`pipeline_out/`](./pipeline_out/).

*Note: the 0.72 micro-F1 above is a validation-set figure from this dissertation-stage pipeline. Don't confuse it with the leakage-corrected full-text micro-F1 reported separately, see the status note above.*

## Data

`CUAD_v1.json` isn't included here. Grab it from the official source:
https://www.atticusprojectai.org/cuad/

Drop it in the working directory (or update the path in each notebook) before running. Every notebook expects `CUAD_v1.json` in its own runtime working directory, check the load cell near the top of each one.

## Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
```


I built and ran these on Google Colab Pro with a single NVIDIA T4 GPU in High-RAM mode. A GPU is recommended for both training and inference.

## Reproducibility

- Random seed fixed to 42 for data splits and sampling throughout.
- The trained artefacts (`legalpro_bert_clause_cls_best.pt`, `clause_to_idx.json`) come out of `Clause_Classification_Legal_Pro_Bert_Final.ipynb` and feed into `Hybrid_Summarisation_Clause_Classification.ipynb`, so run the notebooks in the order listed above.
- Full methodology, hyperparameters, and evaluation protocol live in the dissertation report itself (not included here, available on request).

## Citation

If you're referencing this work, please cite the dissertation:

> Shanmuga Sundaram, R.M. (2025). *LLM-based Summarisation and Clause Classification*. MSc Dissertation, University of Surrey. Supervised by Dr Alaa Marshan.
