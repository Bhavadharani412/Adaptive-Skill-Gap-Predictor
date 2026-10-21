# Adaptive Skill Gap Predictor (ASGP) for Post-Recruitment Employee Training

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18.3-61dafb.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646cff.svg)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Research Final-Year Engineering Project:**  
> *"Adaptive Skill Gap Predictor for Post-Recruitment Employee Training Using Occupation–Skill Knowledge Graphs and Adaptive Recommendation"*

---

## 1. Project Overview

The **Adaptive Skill Gap Predictor (ASGP)** is an end-to-end machine learning system and decision-support web platform designed to streamline and accelerate post-recruitment employee onboarding. 

By modeling occupational domains as heterogeneous knowledge graphs, extracting multi-dimensional competency features ($G_e = S_r \setminus P_e$), and executing graph-informed adaptive recommendations, ASGP guides employees to role readiness significantly faster than conventional static catalogs.

### Key Capabilities:
- **Taxonomy Ingestion:** Ingests official ESCO (European Skills, Competences, Qualifications and Occupations) datasets with automatic validation and fallback.
- **Paper Baseline Reproduction:** Exact reproduction of the reference paper (40 occupations, 220 skills across 8 latent clusters, 12% generic competencies, 600 simulated employees).
- **Circularity Discovery & Disclosure:** Experimental analysis showing that baseline critical-gap models achieve high metrics primarily by approximating an arbitrary relevance-weight threshold rule.
- **Scientifically Defensible Evaluation:** Employee-grouped cross-validation (`GroupKFold`) and an outcome-derived target based on downstream role readiness impact.
- **Adaptive Recommendation Engine:** Implements Policies A (Random), B (Static Relevance), and C (Adaptive Engine with cluster-transfer momentum and prerequisite constraints).
- **Interactive Full-Stack Web App:** Modern, responsive React + TypeScript interface backed by FastAPI and SQLite.

---

## 2. Technology Stack

- **Backend:** Python 3.12, FastAPI, Uvicorn, SQLAlchemy, Pydantic V2
- **Machine Learning & Graph:** scikit-learn (Random Forest, Logistic Regression), NetworkX, SciPy, pandas, NumPy
- **Frontend:** React 18, Vite 5, TypeScript, Custom Vanilla CSS Design System
- **Database:** SQLite (local development) with documented migration path to PostgreSQL
- **Testing & Artifacts:** Pytest, Joblib, Matplotlib

---

## 3. Quickstart & Installation

### 3.1 Prerequisites
- Python 3.11 or 3.12
- Node.js v18+ and npm

### 3.2 Environment Setup
Clone the repository and install backend dependencies:
```bash
# Install Python dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

Install frontend dependencies:
```bash
cd frontend
npm install
cd ..
```

---

## 4. Running the System

### 4.1 Launch the FastAPI Backend Server
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be available at:
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 4.2 Launch the React + Vite Frontend
In a separate terminal:
```bash
cd frontend
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 5. Reproducing Research Experiments

All experimental phases can be executed deterministically via CLI scripts:

```bash
# 1. Ingest ESCO taxonomy or generate synthetic fallback
python scripts/import_data.py

# 2. Generate baseline synthetic data (40 occs, 220 skills, 600 emps, seed=42)
python scripts/generate_synthetic_data.py

# 3. Construct knowledge graph & export topological metrics
python scripts/build_knowledge_graph.py

# 4. Train classifiers (Phase 6 paper reproduction & Phase 7 defensible GroupKFold)
python scripts/train_classifier.py

# 5. Evaluate training policies on matched cohort (Wilcoxon signed-rank tests)
python scripts/evaluate_policies.py --cohort-size 100

# 6. Run 5 policy ablations and render publication figures
python scripts/run_ablations.py --sample-size 60

# 7. Run full automated test suite
pytest backend/tests/ -v
```

---

## 6. Key Empirical Results

### Critical-Gap Classification (Phase 6 vs Phase 7)
| Experiment Phase | Target Formulation | Random Forest $F_1$ | ROC-AUC | Deterministic Baseline $F_1$ |
|---|---|---|---|---|
| **Phase 6: Paper Reproduction** | Top-1/3 Relevance (Circular) | 0.8800 | 0.9854 | 0.8980 |
| **Phase 7: Defensible Evaluation** | Downstream Readiness Impact | **0.8941** (95% CI: [0.890, 0.898]) | **0.9893** | 0.6118 |

### Policy Performance to 80% Readiness Target
| Policy | Mean Steps | Std Dev | Success Rate | Mean Learning Hours | Statistical Significance |
|---|---|---|---|---|---|
| **Policy A (Random)** | 7.35 | 2.67 | 100% | 127.0 hrs | Baseline |
| **Policy B (Static Relevance)** | 7.67 | 2.49 | 100% | 136.8 hrs | $p = 8.03 \times 10^{-4}$ vs Random |
| **Policy C (Adaptive Graph)** | **6.19** | **2.18** | **100%** | **110.4 hrs** | **$p = 5.14 \times 10^{-17}$ vs Static (Cohen's $d = 1.93$)** |

---

## 7. Project Documentation

Comprehensive research and architectural artifacts are located in `docs/`:
- [`docs/research_paper.md`](docs/research_paper.md): Complete draft academic paper formatted with abstract, related work, methodology, experiments, ablations, discussion, and ethical considerations.
- [`docs/architecture.md`](docs/architecture.md): System architecture, Mermaid pipeline diagrams, and formulation specs.
- [`docs/data_dictionary.md`](docs/data_dictionary.md): Detailed schema documentation for all relational tables, CSVs, and feature matrices.
- [`docs/responsible_use_policy.md`](docs/responsible_use_policy.md): Ethical governance, explainability decomposition, and human override guidelines.
- [`docs/reproducibility_checklist.md`](docs/reproducibility_checklist.md): Step-by-step verification checklist for all deterministic experiments.
- [`docs/implementation_status.md`](docs/implementation_status.md): Phase-by-phase implementation progress and verification evidence.
