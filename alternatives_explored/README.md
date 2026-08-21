# Alternatives explored, not adopted

These two notebooks were trained and evaluated during the project but were **not** used to produce the results reported in the dissertation. They're kept here for transparency, referenced in the main README and in dissertation §3.2.

- **`LegalLongformer2208.ipynb`** — Legal-Longformer as a clause classifier (up to 4,096-token context). Not adopted: CUAD clause spans typically fit within 512 tokens, so the added computational cost of long-sequence training wasn't justified for this dataset.
- **`extractive_summarisation_pipeline.ipynb`** — a supervised Legal-Longformer extractive ranker, trialled as an alternative to TextRank for the extractive summarisation stage. TextRank was retained in the final pipeline as it required no training data and performed comparably at this dataset size.
