# PeerReview AI Evaluation Results

## 1. Guardrail Leak Rate (Reviewer Node Red-Team)

| Attack style | Without guardrail | With guardrail |
|--------------|-------------------|----------------|
| direct_request | 0.0% (0/8) | 0.0% (0/8) |
| roleplay | 0.0% (0/8) | 0.0% (0/8) |
| authority_appeal | 0.0% (0/8) | 0.0% (0/8) |
| incremental | 0.0% (0/8) | 0.0% (0/8) |
| hidden_in_praise | 0.0% (0/8) | 0.0% (0/8) |
| **TOTAL** | **0.0% (0/40)** | **0.0% (0/40)** |

## 2. Retrieval Precision (RAG Pipeline)

| Method | Precision@3 | MRR |
|--------|-------------|-----|
| Bi-encoder only | 0.00 | 0.00 |
| Full RAG pipeline | 0.00 | 0.00 |

## 3. Recommender vs Baseline (Synthetic Sessions)

| Method | Top-1 accuracy | Top-2 accuracy |
|--------|----------------|----------------|
| Random | 0.16 | 0.38 |
| Longest-untouched | 0.82 | 0.90 |
| GMM+XGBoost | 0.23 | 0.48 |

### By Skill Level

**Random**
- Novice: Top-1=0.14, Top-2=0.34
- Intermediate: Top-1=0.20, Top-2=0.38
- Advanced: Top-1=0.14, Top-2=0.42

**Longest-untouched**
- Novice: Top-1=0.46, Top-2=0.70
- Intermediate: Top-1=1.00, Top-2=1.00
- Advanced: Top-1=1.00, Top-2=1.00

**GMM+XGBoost**
- Novice: Top-1=0.22, Top-2=0.62
- Intermediate: Top-1=0.22, Top-2=0.40
- Advanced: Top-1=0.24, Top-2=0.42

