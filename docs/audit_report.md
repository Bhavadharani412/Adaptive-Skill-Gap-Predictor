# Comprehensive Research Audit Report: ASGP Conference Paper Revision

**Paper Title:** Adaptive Skill Gap Predictor for Post-Recruitment Employee Training Using Occupation–Skill Knowledge Graphs and Adaptive Recommendation  
**Audit Date:** 21 October 2026  
**Decision Context:** Revise & Resubmit (R&R) Conference Audit  
**Author:** Senior Machine Learning & Systems Research Auditor  

---

## 1. Executive Summary & Verification Matrix

This audit evaluated the codebase, simulation environment, machine learning pipelines, evaluation protocols, and reported metrics in `docs/research_paper.md` against generated outputs in `experiments/results/`.

### 1.1 Result Verification Matrix

| Claim / Metric | Source in Paper | Source in Repository Artifact | Audit Status | Discrepancy & Methodological Assessment |
|---|---|---|---|---|
| **Original Paper Reference:** RF $F_1 = 0.915$, $\text{ROC-AUC} = 0.995$ | Sec. 5.1 | `docs/prd.md` line 225 | **Verified as Reference** | Reported in PRD as reference from literature. Not an output of this codebase. |
| **Phase 6 RF Accuracy = 0.9235, $F_1 = 0.8800$** | Sec. 5.1 | `classifier_results.json` lines 30-37 | **Verified (Metric)** | Reproducible with seed 42 on 70/30 split. |
| **Phase 6 RF Precision = 0.8521, Recall = 0.9100** | Sec. 5.1 Table | `classifier_results.json` lines 32-33 | **INCONSISTENT** | **Discrepancy:** JSON output records Precision = **0.8246**, Recall = **0.9434**. Paper table deviates from actual output. |
| **Phase 6 Deterministic Baseline: Accuracy = 0.9327, $F_1 = 0.8980$** | Sec. 5.1 Table | `classifier_results.json` line 49 | **Verified** | Reproducible. Outperforms ML models ($0.8980 > 0.8800$), confirming circularity. |
| **Phase 7 RF $F_1 = 0.8941$, $\text{ROC-AUC} = 0.9893$** | Sec. 5.2 Table | `classifier_results.json` lines 103-112 | **Verified (Output)** | GroupKFold (5 folds $\times$ 5 seeds = 25 runs) achieves $0.8941$ (95% CI: $[0.8901, 0.8982]$). |
| **Phase 7 Deterministic Baseline $F_1 = 0.6118$** | Sec. 5.2 Table | `classifier_results.json` line 129 | **Verified (Output)** | Deterministic rule drops to 0.6118 on the composite target. |
| **Phase 7 Target Non-Circularity Claim** | Sec. 5.2 Text | `ml_classifier.py` lines 86-98 | **METHODOLOGICALLY FLAWED** | Target is defined directly from two input features: `relevance_weight` ($\ge 0.055$) AND `cluster_coverage` ($< 0.50$). This is not an independent simulated outcome; it is feature-target leakage. |
| **Policy Steps: Random = 7.35, Static = 7.67, Adaptive = 6.19** | Sec. 6.2 Table | `policy_evaluation_results.json` lines 8, 20, 32 | **NUMERICALLY VERIFIED / EXPERIMENTALLY BIASED** | Reproducible under the script, but the result is an **artifact of seed assignment** (see Section 4). |
| **Policy Hours: Random = 127.0h, Static = 136.8h, Adaptive = 110.4h** | Sec. 6.2 Table | `policy_evaluation_results.json` lines 13, 25, 37 | **NUMERICALLY VERIFIED / EXPERIMENTALLY BIASED** | Directly tied to the step count artifact. |
| **Wilcoxon Test: $W = 142.0$ (Adaptive vs. Static)** | Sec. 6.3 Text | `policy_evaluation_results.json` line 56 | **INCONSISTENT** | **Discrepancy:** JSON output records $W = \mathbf{0.0}$ ($p = 5.14 \times 10^{-17}$), not $W = 142.0$. |
| **Wilcoxon Test: $W = 210.5$ (Adaptive vs. Random)** | Sec. 6.3 Text | `policy_evaluation_results.json` line 50 | **INCONSISTENT** | **Discrepancy:** JSON output records $W = \mathbf{69.0}$ ($p = 2.01 \times 10^{-14}$), not $W = 210.5$. |
| **Ablation Claim: Cluster Transfer Improves Adaptive Sequencing** | Abstract & Sec. 7.1 | `ablation_results.json` lines 4, 11, 25 | **UNSUPPORTED BY DATA** | Relevance-Only (V1) achieved **6.02 steps**, while Full Adaptive (V4) achieved **6.05 steps**. Cluster bonus did not reduce steps. |

---

## 2. Investigation of Model Targets & Feature Leakage (Item 4)

### 2.1 Phase 6 Baseline Target: 100% Deterministic Circularity
In `backend/app/services/ml_classifier.py`:
```python
thresholds[occ_id] = float(np.quantile(weights, 0.667))
df["criticality_threshold"] = df["occupation_id"].map(thresholds).fillna(0.75)
df["is_critical"] = (df["relevance_weight"] >= df["criticality_threshold"]).astype(int)
```
Input feature matrix:
$$\mathbf{x} = [\text{relevance\_weight}, \text{is\_generic}, \text{possessed\_skills\_count}, \text{cluster\_coverage}]$$

**Audit Finding:**  
- The ground truth $y$ is computed solely by thresholding feature $x_1$ (`relevance_weight`).
- Across all 6,137 gap records in the synthetic dataset, the match rate between $y$ and the relevance threshold is **6,137 / 6,137 (100.00%)**.
- Random Forest feature importance confirms this: `relevance_weight` accounts for **87.02%** of split decisions.
- A zero-parameter deterministic rule ($x_1 \ge \theta_{occ}$) achieves an $F_1$ score of **0.8980**, outperforming the Random Forest ($F_1 = 0.8800$). The model merely introduces approximation noise into an exact rule.

### 2.2 Phase 7 "Improved" Target: Direct Feature Leakage
In `backend/app/services/ml_classifier.py` (`assign_simulated_outcome_labels`):
```python
w = row["relevance_weight"]
cov = row["cluster_coverage"]
total_w = sum(req_map.values()) if req_map else 1.0
rel_impact = w / total_w if total_w > 0 else 0

# Critical if relevance impact >= 0.055 and cluster coverage < 0.50
crit = 1 if (rel_impact >= 0.055 and cov < 0.50) else 0
```

**Audit Finding:**  
- The paper claimed Phase 7 evaluated a "non-circular simulated outcome target based on downstream role readiness impact."
- However, the code generates the label as an explicit, deterministic Boolean conjunction of **Feature 1** (`relevance_weight`) and **Feature 4** (`cluster_coverage`).
- The Random Forest model achieved $F_1 = 0.8941$ and $\text{ROC-AUC} = 0.9893$ because a tree-based ensemble trivially isolates axis-aligned rectangular boundaries:
  $$\hat{y} = \mathbb{I}(x_1 \ge \tau_1) \land \mathbb{I}(x_4 < 0.50)$$
- The deterministic baseline rule dropped to $F_1 = 0.6118$ solely because it was only allowed to threshold $x_1$, not because machine learning was discovering complex latent patterns.
- **R&R Requirement:** This target construction must be transparently disclosed as a *rule-based vulnerability rubric*, not an empirical post-training outcome. True outcome-derived evaluation would label a gap as critical based on empirical failure rates or simulated post-training project performance independent of input features.

---

## 3. Investigation of Sample Counts & Confusion Matrix Discrepancies (Items 5 & 6)

### 3.1 Sample Count Accounting
- **Total Employees:** 600
- **Total Occupation-Skill Requirement Instances:** 11,385
- **Mastered Skills at Onboarding ($P_e$):** 5,248 (46.10%)
- **Total Missing Skill Gaps ($G_e = S_r \setminus P_e$):** Exactly **6,137 instances**
- **Train/Test Split (70/30):**
  - Train set ($70\%$): 4,295 instances
  - Test set ($30\%$): 1,842 instances
  - $4,295 + 1,842 = 6,137$ (Mathematically exact).

### 3.2 Confusion Matrix Discrepancy in Paper Table 5.1
In `experiments/results/classifier_results.json`:
- **True Negative (TN):** 1,184
- **False Positive (FP):** 110
- **False Negative (FN):** 31
- **True Positive (TP):** 517
- **Total Test Samples:** $1,184 + 110 + 31 + 517 = 1,842$.

Recalculating exact metrics from this confusion matrix:
$$\text{Accuracy} = \frac{1184 + 517}{1842} = \frac{1701}{1842} = 0.92345 \approx \mathbf{0.9235}$$
$$\text{Precision} = \frac{517}{517 + 110} = \frac{517}{627} = 0.82456 \approx \mathbf{0.8246}$$
$$\text{Recall} = \frac{517}{517 + 31} = \frac{517}{548} = 0.94343 \approx \mathbf{0.9434}$$
$$F_1 = \frac{2 \times 0.82456 \times 0.94343}{0.82456 + 0.94343} = \mathbf{0.8800}$$

**Discrepancy in `research_paper.md`:**  
Section 5.1 of the draft paper reported:
- Precision: **0.8521** (Actual: **0.8246**)
- Recall: **0.9100** (Actual: **0.9434**)
- PR-AUC: **0.9712** (Actual: **0.9678**)

The draft paper inverted the precision/recall trade-off (reporting higher precision and lower recall than the actual classifier output).

---

## 4. Audit of Policy Simulation & Seed Independence (Item 8)

### 4.1 Discovery of the Seed Assignment Flaw in `policy_evaluator.py`
In `backend/app/services/policy_evaluator.py` (`evaluate_cohort`, lines 186-199):
```python
for _, emp in sample_emps.iterrows():
    ...
    # Policy A: Random
    res_a = self.simulate_episode(..., seed=random_seed + 1)  # seed=43
    # Policy B: Static
    res_b = self.simulate_episode(..., seed=random_seed + 2)  # seed=44
    # Policy C: Adaptive
    res_c = self.simulate_episode(..., seed=random_seed + 3)  # seed=45
```

**Critical Audit Findings:**
1. **Lack of Per-Employee Seed Variation:** For all 100 employees in the cohort loop, the simulation seed passed to `simulate_episode` was reset to the exact same constant (`43` for Random, `44` for Static, `45` for Adaptive). Consequently, every employee experienced the identical sequence of pseudorandom numbers for assessment pass/fail checks.
2. **Confounding Seed Bias Across Policies:** Static was evaluated with seed 44, while Adaptive was evaluated with seed 45. In stochastic assessment simulation:
   $$\text{pass\_prob} = \text{clip}(0.78 + 0.15 \cdot \text{cov} + \mathcal{N}(0, 0.05), 0.50, 0.98)$$
   The specific sequence produced by seed 44 generated more early assessment failures than seed 45, artificially inflating Static steps to 7.67 while Adaptive recorded 6.19.

### 4.2 Re-Execution Under Rigorous Matched Per-Employee Seeds
When we audited the exact same 100 employees, but assigned each employee an **independent matched seed** ($seed_i = 42000 + i$, identical across all three policies for that specific employee):

| Policy | Original Flawed Run (Seeds 43 vs 44 vs 45) | **Audited Run (Matched Per-Employee Seeds)** |
|---|---|---|
| **Policy A (Random)** | 7.35 steps (127.0 hrs) | **8.03 $\pm$ 2.84 steps (140.6 hrs)** |
| **Policy B (Static Relevance)** | 7.67 steps (136.8 hrs) | **6.83 $\pm$ 2.50 steps (122.2 hrs)** |
| **Policy C (Adaptive Graph)** | 6.19 steps (110.4 hrs) | **6.83 $\pm$ 2.50 steps (122.2 hrs)** |
| **Wilcoxon Adaptive vs. Static** | $W=0.0, p = 5.14 \times 10^{-17}$ | **$W=5.0, p = 1.000$, Cohen’s $d = 0.0000$** |
| **Wilcoxon Adaptive vs. Random** | $W=69.0, p = 2.01 \times 10^{-14}$ | **$W=237.5, p = 1.55 \times 10^{-10}$** |

**Crucial Scientific Conclusion:**  
Under mathematically matched stochastic conditions on this dataset, **Static Relevance and Adaptive Graph recommendations achieve virtually identical efficiency (6.83 steps)**. The previously claimed advantage of 6.19 vs. 7.67 steps was entirely an artifact of different fixed seeds.

### 4.3 Comparison with Original Literature
The original paper reported:
- Random policy: **42.66 steps**
- Static policy: **21.75 steps**
- Adaptive policy: **24.24 steps**

In the original paper:
- **Static was faster than Adaptive by 2.49 steps.**
- In a readiness formula defined as:
  $$\text{Readiness}(e) = \frac{\sum_{s \in P_e \cap S_r} w_{r,s}}{\sum_{s \in S_r} w_{r,s}}$$
  a greedy static ranking of $w_{r,s}$ descending is mathematically optimal for maximizing the numerator in the fewest steps.
- Adaptive recommendation incorporates cluster transfer ($\beta$) and prerequisites ($\gamma$), which intentionally diversifies selection away from pure top-weighted skills to provide structural domain scaffolding.
- Claiming that Adaptive achieved fewer steps than Static was inconsistent with both the original literature and the underlying mathematics of greedy numerator optimization.

---

## 5. Audit of the Ablation Study (Item 7)

### 5.1 The Ablation Data
In `experiments/results/ablation_results.json`:
- **Variant 1 (Relevance-Only):** Mean Steps = **6.02 $\pm$ 2.53**, Mean Hours = **108.5 hrs**
- **Variant 2 (Relevance + Cluster):** Mean Steps = **6.05 $\pm$ 2.57**, Mean Hours = **109.1 hrs**
- **Variant 3 (Relevance + Prereq):** Mean Steps = **6.02 $\pm$ 2.53**, Mean Hours = **108.5 hrs**
- **Variant 4 (Full Adaptive):** Mean Steps = **6.05 $\pm$ 2.57**, Mean Hours = **109.1 hrs**
- **Variant 5 (Random Baseline):** Mean Steps = **8.62 $\pm$ 2.27**, Mean Hours = **151.3 hrs**

### 5.2 The Claim Discrepancy
In `docs/research_paper.md` (Abstract and Section 7), the paper claimed:
> *"Systematic policy ablations and sensitivity analyses confirm the cognitive value of cluster-transfer momentum and prerequisite-aware sequencing in reducing steps to readiness."*

**Audit Finding:**  
This claim is **directly contradicted by the ablation results**:
1. Variant 1 (Relevance-Only) achieved **6.02 steps**.
2. Adding cluster transfer (Variant 2) or full adaptive scoring (Variant 4) resulted in **6.05 steps** (+0.03 steps / +0.6 hours).
3. Cluster-transfer sensitivity analysis showed that increasing $\beta$ from 0.0 to 0.85 systematically increases steps from 6.02 to 6.15.
4. **Why this occurs:** In the simulation, base pass rates are relatively high ($\sim 85\%$). Diverting recommendations to lower-relevance skills within the same cluster slows readiness progression without providing enough failure-prevention benefit to offset the smaller numerator gain.

---

## 6. List of Claims in the Paper That Must Be Rewritten (Item 10)

The following claims in `docs/research_paper.md` are methodologically indefensible or factually inconsistent with generated outputs and must be rewritten for the R&R submission:

1. **Abstract & Conclusion (Adaptive Step Superiority):**
   - *Current claim:* "Policy C achieves target readiness in significantly fewer steps (mean 6.19 vs. Static 7.67, $p = 5.14 \times 10^{-17}$)."
   - *Correction:* Under matched stochastic seeds, Adaptive and Static achieve equivalent efficiency (**6.83 steps** vs. Random **8.03 steps**, $p = 1.55 \times 10^{-10}$). Adaptive's value is domain coherence and prerequisite compliance, not a reduction in greedy step count.
2. **Abstract & Section 7 (Ablation Confirmation of Cluster Bonus):**
   - *Current claim:* "Ablations confirm the cognitive value of cluster-transfer momentum in accelerating readiness."
   - *Correction:* Disclose that in the ablation study, Relevance-Only achieved 6.02 steps while Full Adaptive achieved 6.05 steps. Acknowledge that cluster transfer introduces a trade-off between cognitive scaffolding and greedy readiness optimization.
3. **Section 5.1 Table (Precision and Recall Mismatches):**
   - *Current claim:* RF Precision = 0.8521, Recall = 0.9100.
   - *Correction:* Update Table 5.1 to reflect the exact outputs in `classifier_results.json`: RF Precision = **0.8246**, Recall = **0.9434**, PR-AUC = **0.9678**.
4. **Section 5.2 (Simulated Outcome Non-Circularity):**
   - *Current claim:* Phase 7 is an "independent, non-circular simulated outcome target."
   - *Correction:* Disclose that Phase 7 targets role vulnerability through a multi-attribute rule combining `relevance_weight` and `cluster_coverage`. Clarify that the Random Forest outperforms the 1D threshold rule because it models interaction between these two features.
5. **Section 6.3 ($W$-Statistics):**
   - *Current claim:* $W = 142.0$ (Adaptive vs. Static) and $W = 210.5$ (Adaptive vs. Random).
   - *Correction:* Update to the actual values recorded in `policy_evaluation_results.json` ($W = 0.0$ and $W = 69.0$) or the matched-seed re-evaluation ($W = 5.0, p = 1.000$ vs. Static; $W = 237.5, p = 1.55 \times 10^{-10}$ vs. Random).

---

## 7. Change Log & Reproduction Instructions

### 7.1 Actions Taken
1. **Protected Research Papers from Git Tracking:** Updated `.gitignore` to ignore `docs/*paper*.md` and `docs/prd.md`.
2. **Created Verification Script:** Created `scripts/audit_analysis.py` to audit feature correlations, seed independence, and ablation dynamics.
3. **Verified Existing Test Suite:** All 11 unit and integration tests in `backend/tests/` passed cleanly in 10.39s.

### 7.2 Exact Reproduction Commands
```powershell
# 1. Verify test suite (11 passed)
pytest backend/tests/ -v

# 2. Ingest / Validate Dataset
python scripts/import_data.py

# 3. Generate Baseline Synthetic Cohort (seed 42)
python scripts/generate_synthetic_data.py --seed 42

# 4. Construct Knowledge Graph
python scripts/build_knowledge_graph.py

# 5. Train Classifiers (produces experiments/results/classifier_results.json)
python scripts/train_classifier.py

# 6. Evaluate Matched Policy Cohort (produces experiments/results/policy_evaluation_results.json)
python scripts/evaluate_policies.py --cohort-size 100

# 7. Execute Ablation Studies (produces experiments/results/ablation_results.json & plots)
python scripts/run_ablations.py --sample-size 60

# 8. Run Complete Audit & Verification Script
python scripts/audit_analysis.py
```
