# Responsible AI & Decision-Support Policy (ASGP)

**Project:** Adaptive Skill Gap Predictor for Post-Recruitment Employee Training Using Occupation–Skill Knowledge Graphs and Adaptive Recommendation  
**Document Classification:** Ethical Governance, Explainability & Responsible AI Standard  

---

## 1. Executive Summary & Purpose

The **Adaptive Skill Gap Predictor (ASGP)** is strictly designed as an **augmented decision-support system** for post-recruitment employee onboarding, internal upskilling, and professional training curriculum planning. 

**Core Governance Mandate:**
Under no circumstances should this system be utilized as an autonomous tool for employee hiring, compensation determination, disciplinary action, performance demerits, or workforce reduction (firing). All algorithmic outputs represent pedagogical recommendations intended to guide human managers and employees toward optimal learning trajectories.

---

## 2. Explainability & Transparent Rationale Architecture

Every recommendation produced by ASGP is accompanied by an auditable, human-interpretable rationale payload (`RecommendationExplanation`).

### 2.1 Score Decomposition

The adaptive recommendation score is calculated transparently as:

$$\text{Score}(s) = \alpha \cdot w_{r,s} + \beta \cdot \text{ClusterTransferBonus}(s, P_e) + \gamma \cdot \text{PrereqBonus}(s, P_e)$$

Where:
1. **Occupation Relevance ($w_{r,s}$):** Derived from official occupational taxonomy requirements (e.g., ESCO) or documented synthetic baseline curricula.
2. **Cluster Momentum ($\text{ClusterTransferBonus}$):** Reflects cognitive scaffolding—skills residing within a skill cluster where the employee already possesses adjacent competencies are prioritized to leverage positive learning transfer.
3. **Prerequisite Feasibility ($\text{PrereqBonus}$):** Ensures foundational competencies are mastered prior to advancing to complex, high-difficulty modules.

### 2.2 Distinguishing Model Predictions from Deterministic Rules

ASGP explicitly distinguishes between:
- **Taxonomic Facts:** Direct occupational requirements and taxonomy definitions.
- **Trained Model Predictions:** Probabilistic outputs from the Random Forest Criticality Classifier. The system displays both the binary criticality classification and the calibrated probability estimate, explicitly informing managers that synthetic outcome models represent estimates under simulated transfer curves rather than guaranteed workplace performance.

---

## 3. Human-in-the-Loop & Managerial Override

System autonomy is bounded by explicit human oversight:
1. **Manual Override Capability:** Managers and employees have the constitutional ability to override any recommended module via the API (`POST /api/recommendations/{employee_id}/override`) and frontend interface.
2. **Audit Logging:** Every override requires a documented business/pedagogical rationale, which is permanently logged in `DBRecommendationLog` alongside model predictions for longitudinal governance reviews.
3. **Editable Assessment States:** Employees or mentors can update skill states ('in_progress', 'mastered') and recorded assessment scores directly.

---

## 4. Demographic Fairness & Protected Attribute Exclusion

### 4.1 Strict Feature Independence
The predictive model's feature vector $X$ consists solely of:
1. `relevance_weight` (Task-specific importance)
2. `is_generic` (Cross-cutting skill indicator)
3. `possessed_skills_count` (Current size of mastered skill portfolio)
4. `cluster_coverage` (Competency density in the target domain)

**Strict Prohibition:**
No demographic attributes (e.g., age, gender, race, ethnicity, geographic origin, marital status, disability status, or socioeconomic markers) are collected, processed, or permitted in the feature schema.

### 4.2 Handling Incomplete Data
In accordance with ethical HR safety standards, unassessed skills are handled via a configurable policy (`pessimistic` by default), preventing unjustified assumptions of competency while allowing transparent customization for diagnostic screening.

---

## 5. Auditability and Model Governance

1. **Versioned Artifacts:** All trained models are serialized with joblib and version-tracked with exact random seeds, hyperparameter sets, and timestamped training metrics.
2. **Reproducibility Verification:** Automated tests verify deterministic output generation and baseline integrity.
3. **Data Quality Auditing:** Dataset ingestions generate an automated `data_quality_report.json` reporting unresolved references, duplicate entities, and relationship balance.
