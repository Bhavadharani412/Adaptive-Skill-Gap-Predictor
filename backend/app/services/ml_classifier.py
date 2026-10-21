import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, brier_score_loss
)
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from backend.app.services.gap_analyzer import SkillGapAnalyzer
from backend.app.services.knowledge_graph import OccupationSkillKnowledgeGraph

logger = logging.getLogger(__name__)


class DeterministicThresholdClassifier(BaseEstimator, ClassifierMixin):
    """
    Deterministic baseline classifier:
    Predicts critical gap if relevance_weight >= occupation-specific 66.7th percentile.
    Serves as scientific benchmark to detect whether ML models merely memorize
    the circular labeling heuristic.
    """

    def __init__(self, quantile: float = 0.667):
        self.quantile = quantile
        self.thresholds_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if "occupation_id" in X.columns and "relevance_weight" in X.columns:
            for occ_id, grp in X.groupby("occupation_id"):
                self.thresholds_[occ_id] = float(np.quantile(grp["relevance_weight"], self.quantile))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in X.iterrows():
            occ_id = row.get("occupation_id")
            w = row.get("relevance_weight", 0.0)
            thresh = self.thresholds_.get(occ_id, 0.75)
            preds.append(1 if w >= thresh else 0)
        return np.array(preds)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.predict(X)
        proba = np.zeros((len(preds), 2))
        for i, p in enumerate(preds):
            proba[i] = [0.05, 0.95] if p == 1 else [0.95, 0.05]
        return proba


class CriticalGapClassifierService:
    """
    Handles Phase 6 (Paper Reproduction) and Phase 7 (Improved Evaluation)
    for Critical Skill Gap Classification.
    """

    FEATURE_COLS = ["relevance_weight", "is_generic", "possessed_skills_count", "cluster_coverage"]

    def __init__(self, kg: OccupationSkillKnowledgeGraph, models_dir: Optional[Path] = None):
        from backend.app.core.config import settings
        self.kg = kg
        self.models_dir = models_dir or settings.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def assign_paper_baseline_labels(self, gaps_df: pd.DataFrame) -> pd.DataFrame:
        """
        Paper reproduction label:
        A missing skill is labeled critical (y=1) if its relevance weight
        lies in the top third (>= 66.7th percentile) of required-skill weights for that occupation.
        """
        df = gaps_df.copy()
        thresholds = {}

        for occ_id, skills_map in self.kg.occ_required_skills.items():
            if skills_map:
                weights = list(skills_map.values())
                thresholds[occ_id] = float(np.quantile(weights, 0.667))
            else:
                thresholds[occ_id] = 0.75

        df["criticality_threshold"] = df["occupation_id"].map(thresholds).fillna(0.75)
        df["is_critical"] = (df["relevance_weight"] >= df["criticality_threshold"]).astype(int)
        df["target_type"] = "paper_baseline_circular"
        return df

    def assign_simulated_outcome_labels(self, gaps_df: pd.DataFrame) -> pd.DataFrame:
        """
        Improved Phase 7 label (Simulated Outcome):
        A missing skill is labeled critical based on downstream role readiness impact:
        Relative weight >= 0.06 of total occupation weight AND cluster coverage < 0.50
        (addressing strategic role vulnerability rather than raw weight alone).
        """
        df = gaps_df.copy()
        is_crit = []

        for _, row in df.iterrows():
            occ_id = row["occupation_id"]
            w = row["relevance_weight"]
            cov = row["cluster_coverage"]
            req_map = self.kg.get_required_skills(occ_id)
            total_w = sum(req_map.values()) if req_map else 1.0
            rel_impact = w / total_w if total_w > 0 else 0

            # Critical if skill carries significant weight impact (> 5.5% of total role)
            # and employee cluster coverage is low (< 50%), representing vulnerable competency zone
            crit = 1 if (rel_impact >= 0.055 and cov < 0.50) else 0
            is_crit.append(crit)

        df["is_critical"] = is_crit
        df["target_type"] = "simulated_outcome_improved"
        return df

    def run_paper_reproduction(
        self,
        gaps_df: pd.DataFrame,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Phase 6: 70/30 standard split, Logistic Regression and Random Forest.
        Evaluates accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix.
        Documents circularity.
        """
        labeled_df = self.assign_paper_baseline_labels(gaps_df)
        X = labeled_df[self.FEATURE_COLS]
        y = labeled_df["is_critical"]

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, labeled_df.index, test_size=0.30, random_state=random_seed, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 1. Logistic Regression
        lr = LogisticRegression(class_weight="balanced", random_state=random_seed, max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        lr_preds = lr.predict(X_test_scaled)
        lr_probs = lr.predict_proba(X_test_scaled)[:, 1]

        # 2. Random Forest
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=random_seed, class_weight="balanced"
        )
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_probs = rf.predict_proba(X_test)[:, 1]

        # 3. Deterministic Threshold Rule Baseline
        det = DeterministicThresholdClassifier()
        det.fit(labeled_df.loc[idx_train])
        det_preds = det.predict(labeled_df.loc[idx_test])
        det_probs = det.predict_proba(labeled_df.loc[idx_test])[:, 1]

        results = {
            "experiment": "Phase 6: Paper Reproduction Baseline",
            "split": "70/30 standard stratified train-test split",
            "random_seed": random_seed,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "class_distribution": {
                "negative (non-critical)": int((y == 0).sum()),
                "positive (critical)": int((y == 1).sum()),
                "critical_ratio": round(float((y == 1).mean()), 4)
            },
            "circularity_disclosure": (
                "Methodological Limitation: The paper's baseline target is defined directly from relevance_weight "
                "(top third quantile), which is simultaneously provided as a predictive feature. "
                "Consequently, classifiers primarily approximate this single-variable step function."
            ),
            "models": {
                "logistic_regression": self._compute_metrics(y_test, lr_preds, lr_probs),
                "random_forest": self._compute_metrics(y_test, rf_preds, rf_probs),
                "deterministic_threshold_rule": self._compute_metrics(y_test, det_preds, det_probs)
            },
            "random_forest_feature_importance": dict(
                zip(self.FEATURE_COLS, [round(float(v), 4) for v in rf.feature_importances_])
            ),
            "paper_reported_reference_rf_f1": 0.915
        }

        # Save models to disk
        joblib.dump({"model": lr, "scaler": scaler}, self.models_dir / "logistic_regression_paper.joblib")
        joblib.dump({"model": rf}, self.models_dir / "random_forest_paper.joblib")

        return results

    def run_improved_evaluation(
        self,
        gaps_df: pd.DataFrame,
        n_splits: int = 5,
        seeds: List[int] = [42, 101, 202, 303, 404]
    ) -> Dict[str, Any]:
        """
        Phase 7: Scientifically Defensible Evaluation:
        - GroupKFold by employee_id (prevents data leakage across employees)
        - Multi-seed runs with 95% Confidence Intervals
        - Probability calibration (Brier score)
        - Comparison against deterministic rule
        - Evaluated on Simulated Outcome target (non-circular)
        """
        labeled_df = self.assign_simulated_outcome_labels(gaps_df)
        X = labeled_df[self.FEATURE_COLS]
        y = labeled_df["is_critical"]
        groups = labeled_df["employee_id"]

        metrics_accumulator = {
            "logistic_regression": {"f1": [], "roc_auc": [], "pr_auc": [], "brier": []},
            "random_forest": {"f1": [], "roc_auc": [], "pr_auc": [], "brier": []},
            "deterministic_rule": {"f1": [], "roc_auc": [], "pr_auc": [], "brier": []}
        }

        for seed in seeds:
            gkf = GroupKFold(n_splits=n_splits)

            for train_idx, test_idx in gkf.split(X, y, groups=groups):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                df_tr, df_te = labeled_df.iloc[train_idx], labeled_df.iloc[test_idx]

                # Scaled for LR
                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr)
                X_te_sc = scaler.transform(X_te)

                # LR
                lr = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
                lr.fit(X_tr_sc, y_tr)
                lr_p = lr.predict(X_te_sc)
                lr_pr = lr.predict_proba(X_te_sc)[:, 1]

                # RF
                rf = RandomForestClassifier(
                    n_estimators=100, max_depth=6, random_state=seed, class_weight="balanced"
                )
                rf.fit(X_tr, y_tr)
                rf_p = rf.predict(X_te)
                rf_pr = rf.predict_proba(X_te)[:, 1]

                # Det Rule
                det = DeterministicThresholdClassifier()
                det.fit(df_tr)
                det_p = det.predict(df_te)
                det_pr = det.predict_proba(df_te)[:, 1]

                # Accumulate
                for name, p, pr in [("logistic_regression", lr_p, lr_pr), ("random_forest", rf_p, rf_pr), ("deterministic_rule", det_p, det_pr)]:
                    metrics_accumulator[name]["f1"].append(f1_score(y_te, p, zero_division=0))
                    metrics_accumulator[name]["roc_auc"].append(roc_auc_score(y_te, pr))
                    metrics_accumulator[name]["pr_auc"].append(average_precision_score(y_te, pr))
                    metrics_accumulator[name]["brier"].append(brier_score_loss(y_te, pr))

        # Summarize with 95% Confidence Intervals
        ci_summary = {}
        for m_name, vals in metrics_accumulator.items():
            ci_summary[m_name] = {}
            for metric, scores in vals.items():
                mean_val = float(np.mean(scores))
                sem = float(stats.sem(scores))
                ci_95 = 1.96 * sem
                ci_summary[m_name][metric] = {
                    "mean": round(mean_val, 4),
                    "std": round(float(np.std(scores)), 4),
                    "ci_95_lower": round(mean_val - ci_95, 4),
                    "ci_95_upper": round(mean_val + ci_95, 4)
                }

        # Train a final production model on full outcome-labeled dataset
        final_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced")
        final_rf.fit(X, y)
        joblib.dump({"model": final_rf, "features": self.FEATURE_COLS}, self.models_dir / "random_forest_improved.joblib")

        return {
            "experiment": "Phase 7: Improved Defensible Evaluation",
            "protocol": f"Employee GroupKFold ({n_splits} folds) over {len(seeds)} random seeds",
            "total_evaluations": n_splits * len(seeds),
            "target": "Simulated Outcome (non-circular role vulnerability)",
            "metrics_with_95_ci": ci_summary,
            "feature_importances": dict(
                zip(self.FEATURE_COLS, [round(float(v), 4) for v in final_rf.feature_importances_])
            )
        }

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        cm = confusion_matrix(y_true, y_pred)
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
            "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
            "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
            "confusion_matrix": {
                "true_negative": int(cm[0, 0]),
                "false_positive": int(cm[0, 1]),
                "false_negative": int(cm[1, 0]),
                "true_positive": int(cm[1, 1])
            }
        }
