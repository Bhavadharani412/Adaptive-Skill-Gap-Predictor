# ASGP Scientific Reproducibility Checklist

This checklist verifies compliance with empirical machine learning standards, ensuring every dataset, training pipeline, and policy simulation can be deterministically reproduced from scratch.

---

## 1. Environment & Dependency Specifications

- [x] **Python Runtime:** Python 3.12 (tested on 3.12.10)
- [x] **Node.js Runtime:** Node.js v24 (tested on v24.12.0)
- [x] **Hardware Portability:** CPU-compatible classical ML models (Random Forest, Logistic Regression); GPU not required.
- [x] **Dependency Locking:** `requirements.txt` specifies minimum and verified package versions (`scikit-learn>=1.5`, `pandas>=2.2`, `networkx>=3.3`, `fastapi>=0.115`, `scipy>=1.14`, `SQLAlchemy>=2.0`).

---

## 2. Deterministic Pipeline Execution

All random numbers utilize explicit generators (`numpy.random.default_rng(seed)`) and fixed random seeds.

| Pipeline Component | CLI Script | Default Seed | Expected Deterministic Output |
|---|---|---|---|
| Synthetic Dataset Generation | `python scripts/generate_synthetic_data.py` | `42` | 40 occs, 220 skills (26 generic), 759 edges, 600 emps |
| Knowledge Graph Construction | `python scripts/build_knowledge_graph.py` | N/A | 260 nodes, 1,942 edges, 8 clusters |
| Classifier Training & Evaluation | `python scripts/train_classifier.py` | `42` (Paper), `[42, 101, 202, 303, 404]` (GroupKFold) | Paper RF F1 ~ 0.88, Improved RF F1 ~ 0.894 (95% CI: [0.890, 0.898]) |
| Recommendation Policy Evaluation | `python scripts/evaluate_policies.py` | `42` | Matched seeds: Adaptive & Static ~6.83 steps vs Random ~8.03 steps (p < 0.001) |
| Ablation & Sensitivity Analysis | `python scripts/run_ablations.py` | `42` | 5 policy variants, 3 publication plots |
| Database Seeding | `python -m backend.app.database.init_db` | N/A | SQLite database at `data/asgp.db` |
| Automated Test Suite | `pytest backend/tests/ -v` | N/A | 11 passed |

---

## 3. Methodological Transparency & Limitations

- [x] **Circularity Disclosure:** The baseline paper's target definition (top-1/3 relevance weight quantile) is circular because relevance weight is provided as an input feature. This was experimentally verified: Random Forest assigns 87% feature importance to relevance weight.
- [x] **Data Leakage Prevention:** In Phase 7 improved evaluation, `GroupKFold` grouped by `employee_id` guarantees that skills from the same employee never appear in both training and testing folds simultaneously.
- [x] **Non-circular Target:** The Phase 7 model introduces a non-circular simulated outcome target based on downstream role readiness impact. On this target, Random Forest outperforms the deterministic rule (F1: 0.894 vs 0.612).
- [x] **Statistical Significance:** Paired Wilcoxon signed-rank tests are evaluated across matched starting states, confirming that the adaptive policy yields statistically significant step reductions ($p < 10^{-15}$).
- [x] **Fairness & Governance:** No demographic, protected, or proxy attributes are collected or utilized in feature matrices.
