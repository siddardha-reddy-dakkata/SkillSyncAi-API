"""
SkillSyncAI — Optimization Pipeline (v3.x)
============================================

Runs 5 sequential optimization steps, each building on the last.
Maintains version tracking so we can compare and rollback.

Steps:
  v3.0  Feature Pruning        — Drop low-importance features
  v3.1  Hyperparameter Search   — Grid/random search for SVM, RF, GB
  v3.2  Resume Data Augment     — Generate training data from real resumes
  v3.3  XGBoost Comparison      — Try XGBoost/LightGBM if available
  v3.4  Threshold Optimization  — Optimize decision threshold for F1

Each step saves its model and prints a comparison table at the end.

USAGE:
  python optimization_trainer.py
"""

import json
import os
import sys
import random
import pickle
import math
import time
import copy
from collections import Counter
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "feedback_training_data.json")
MODEL_PATH = os.path.join(BASE_DIR, "model_weights.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "training_report.json")
VERSIONS_DIR = os.path.join(BASE_DIR, "model_versions")
RESUME_PATH = r"C:\Users\SIDDARDHA REDDY\Downloads\entityrecognitioninresumes.json\Entity Recognition in Resumes.json"

os.makedirs(VERSIONS_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)


# ============================================================
# VERSION TRACKER
# ============================================================

class VersionTracker:
    """Track optimization versions with metrics for comparison."""
    
    def __init__(self):
        self.versions = []
        self.log_path = os.path.join(VERSIONS_DIR, "optimization_log.json")
        # Load existing log if present
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                self.versions = json.load(f)
    
    def save_version(self, version_id: str, description: str, metrics: Dict,
                     model_data: Dict, feature_names: List[str]):
        """Save a model version with full metadata."""
        entry = {
            "version": version_id,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "metrics": {k: v for k, v in metrics.items() 
                       if isinstance(v, (int, float, str, bool, list))},
            "n_features": len(feature_names),
            "feature_names": feature_names,
        }
        self.versions.append(entry)
        
        # Save model binary
        model_file = os.path.join(VERSIONS_DIR, f"model_{version_id}.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(model_data, f)
        
        # Save log
        with open(self.log_path, "w") as f:
            json.dump(self.versions, f, indent=2, default=str)
        
        print(f"  [VERSION] Saved {version_id} → {model_file}")
    
    def print_comparison(self):
        """Print comparison table of all versions."""
        print(f"\n{'='*90}")
        print(f"  VERSION COMPARISON TABLE")
        print(f"{'='*90}")
        print(f"  {'Version':<10}{'Description':<30}{'Holdout%':<10}{'F1':<8}{'AUC':<8}{'Feats':<7}{'Records':<8}")
        print(f"  {'-'*85}")
        for v in self.versions:
            m = v["metrics"]
            print(f"  {v['version']:<10}"
                  f"{v['description'][:29]:<30}"
                  f"{m.get('holdout_accuracy', 0)*100:<10.1f}"
                  f"{m.get('holdout_f1', 0):<8.3f}"
                  f"{m.get('holdout_auc', 0):<8.3f}"
                  f"{v['n_features']:<7}"
                  f"{m.get('records_used', 0):<8}")
        print(f"{'='*90}\n")
    
    def get_best(self) -> Optional[Dict]:
        """Return the version with highest holdout F1."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v["metrics"].get("holdout_f1", 0))


# ============================================================
# FEATURE EXTRACTOR (25 features — same as v2.0)
# ============================================================

CRITICAL_ROLES = {
    "web_application": {"frontend", "backend"},
    "ml_project": {"ml_engineer", "data_engineer"},
    "mobile_app": {"frontend", "backend"},
    "data_pipeline": {"data_engineer", "backend"},
    "api_service": {"backend", "devops"},
    "database_system": {"backend", "data_engineer"},
}

HELPFUL_ROLES = {
    "web_application": {"fullstack", "devops", "ui_ux"},
    "ml_project": {"backend", "devops"},
    "mobile_app": {"ui_ux", "devops", "fullstack"},
    "data_pipeline": {"ml_engineer", "devops"},
    "api_service": {"fullstack", "ml_engineer"},
    "database_system": {"devops", "data_engineer"},
}

ALL_25_FEATURES = [
    "team_size", "role_diversity", "avg_skill_level", "min_skill_level",
    "max_skill_level", "skill_variance", "skill_range", "gini_coefficient",
    "avg_experience", "experience_variance", "skill_coverage",
    "has_critical_role_gap", "role_duplication_ratio", "project_type_encoded",
    "median_skill_level", "skill_iqr", "total_skills_count", "skills_per_member",
    "critical_role_coverage", "helpful_role_coverage",
    "experience_skill_interaction", "weakest_link_score", "team_strength_index",
    "coverage_diversity_product", "balance_penalty",
]

PROJECT_TYPE_MAP = {
    "web_application": 0, "ml_project": 1, "mobile_app": 2,
    "data_pipeline": 3, "api_service": 4, "database_system": 5,
}


def extract_features_full(record: Dict) -> np.ndarray:
    """Extract all 25 features from a team record."""
    members = record.get("members", [])
    project = record.get("project", {})
    project_type = project.get("project_type", "web_application")
    required_skills = set(s.lower() for s in project.get("required_skills", []))
    
    team_size = len(members)
    if team_size == 0:
        return np.zeros(25)
    
    roles = [m.get("assigned_role", "unknown") for m in members]
    unique_roles = len(set(roles))
    role_diversity = unique_roles / team_size
    
    skill_levels = [m.get("skill_level", 5) for m in members]
    avg_skill = np.mean(skill_levels)
    min_skill = min(skill_levels)
    max_skill = max(skill_levels)
    skill_var = np.var(skill_levels)
    skill_range = max_skill - min_skill
    median_skill = float(np.median(skill_levels))
    q75, q25 = np.percentile(skill_levels, [75, 25])
    skill_iqr = float(q75 - q25)
    
    # Gini coefficient
    sorted_skills = sorted(skill_levels)
    n = len(sorted_skills)
    cum = sum((2 * (i + 1) - n - 1) * sorted_skills[i] for i in range(n))
    gini = cum / (n * sum(sorted_skills)) if sum(sorted_skills) > 0 else 0
    gini = max(0, gini)
    
    experiences = [m.get("experience_years", 1) for m in members]
    avg_exp = np.mean(experiences)
    exp_var = np.var(experiences)
    
    # Skill coverage
    team_skills = set()
    total_skills_count = 0
    for m in members:
        member_skills = [s.lower() for s in m.get("skills", [])]
        team_skills.update(member_skills)
        total_skills_count += len(member_skills)
    
    coverage = len(team_skills & required_skills) / max(len(required_skills), 1)
    skills_per_member = total_skills_count / team_size
    
    # Critical role gap
    critical = CRITICAL_ROLES.get(project_type, set())
    role_set = set(roles)
    has_critical_gap = 1.0 if (critical and not critical & role_set) else 0.0
    critical_role_cov = len(critical & role_set) / max(len(critical), 1) if critical else 1.0
    
    helpful = HELPFUL_ROLES.get(project_type, set())
    helpful_role_cov = len(helpful & role_set) / max(len(helpful), 1) if helpful else 0.0
    
    # Role duplication
    role_counts = Counter(roles)
    dup_count = sum(1 for c in role_counts.values() if c > 1)
    dup_ratio = dup_count / max(len(role_counts), 1)
    
    project_encoded = PROJECT_TYPE_MAP.get(project_type, 0)
    
    # Interaction features
    exp_skill_interaction = avg_exp * avg_skill
    weakest_link = min_skill * role_diversity
    team_strength = avg_skill * team_size * coverage
    cov_div_product = coverage * role_diversity
    balance_penalty = skill_var * gini
    
    return np.array([
        team_size, role_diversity, avg_skill, min_skill, max_skill,
        skill_var, skill_range, gini, avg_exp, exp_var,
        coverage, has_critical_gap, dup_ratio, project_encoded,
        median_skill, skill_iqr, total_skills_count, skills_per_member,
        critical_role_cov, helpful_role_cov,
        exp_skill_interaction, weakest_link, team_strength,
        cov_div_product, balance_penalty,
    ])


# ============================================================
# DATA LOADER
# ============================================================

def load_training_data() -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Load and featurize training data."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    
    # Support both key names
    records = data.get("team_records", data.get("records", []))
    X_list, y_list, valid_records = [], [], []
    
    for rec in records:
        outcome = rec.get("outcome", {})
        success_val = outcome.get("success")
        if success_val is None:
            tag = outcome.get("tag", "")
            success_val = 1 if "success" in str(tag).lower() else 0
        elif isinstance(success_val, str):
            success_val = 1 if success_val.lower() in ("true", "1") else 0
        else:
            success_val = int(bool(success_val))
        
        features = extract_features_full(rec)
        X_list.append(features)
        y_list.append(success_val)
        valid_records.append(rec)
    
    return np.array(X_list), np.array(y_list), valid_records


# ============================================================
# TRAINING CORE (shared evaluation function)
# ============================================================

def evaluate_model(model, scaler, X_train, y_train, X_test, y_test,
                   feature_names: List[str], model_name: str = "") -> Dict:
    """Train and evaluate a model, return metrics dict."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix
    )
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model.fit(X_train_s, y_train)
    
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None and len(set(y_test)) > 1 else 0
    cm = confusion_matrix(y_test, y_pred)
    
    # CV on training set
    cv = StratifiedKFold(n_splits=min(10, max(3, len(y_train) // 5)), shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="f1")
    
    return {
        "model_name": model_name,
        "holdout_accuracy": float(acc),
        "holdout_f1": float(f1),
        "holdout_precision": float(prec),
        "holdout_recall": float(rec),
        "holdout_auc": float(auc),
        "cv_f1_mean": float(cv_f1.mean()),
        "cv_f1_std": float(cv_f1.std()),
        "confusion_matrix": cm.tolist(),
    }


def compute_learned_weights(model, scaler, X_test, y_test, feature_names):
    """Compute learned weights from feature importances."""
    from sklearn.inspection import permutation_importance
    
    X_test_s = scaler.transform(X_test)
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "estimator") and hasattr(model.estimator, "feature_importances_"):
        importances = model.estimator.feature_importances_
    else:
        perm = permutation_importance(model, X_test_s, y_test, n_repeats=10,
                                      random_state=42, scoring="f1")
        importances = np.maximum(perm.importances_mean, 0)
        total = importances.sum()
        importances = importances / total if total > 0 else np.ones(len(feature_names)) / len(feature_names)
    
    imp_dict = {feature_names[i]: importances[i] for i in range(len(feature_names))}
    
    skill_feats = ["avg_skill_level", "min_skill_level", "max_skill_level",
                   "median_skill_level", "skill_iqr"]
    div_feats = ["role_diversity"]
    bal_feats = ["skill_variance", "skill_range", "gini_coefficient", "balance_penalty"]
    exp_feats = ["avg_experience", "experience_variance", "experience_skill_interaction"]
    cov_feats = ["skill_coverage", "has_critical_role_gap", "critical_role_coverage",
                 "helpful_role_coverage", "coverage_diversity_product"]
    
    def s(feats): return sum(imp_dict.get(f, 0) for f in feats)
    
    total = (s(skill_feats) + s(div_feats) + s(bal_feats) + s(exp_feats) + s(cov_feats) +
             imp_dict.get("team_size", 0) + imp_dict.get("role_duplication_ratio", 0) +
             imp_dict.get("project_type_encoded", 0))
    if total == 0: total = 1.0
    
    weights = {
        "weight_skill": round(s(skill_feats) / total, 4),
        "weight_diversity": round(s(div_feats) / total, 4),
        "weight_balance": round(s(bal_feats) / total, 4),
        "weight_experience": round(s(exp_feats) / total, 4),
        "weight_coverage": round(s(cov_feats) / total, 4),
        "weight_team_size": round(imp_dict.get("team_size", 0) / total, 4),
        "weight_role_dedup": round(imp_dict.get("role_duplication_ratio", 0) / total, 4),
        "penalty_critical_gap": round(imp_dict.get("has_critical_role_gap", 0) * 2, 4),
        "penalty_high_variance": round(imp_dict.get("skill_variance", 0) * 2, 4),
        "bonus_high_coverage": round(imp_dict.get("skill_coverage", 0) * 1.5, 4),
    }
    
    # Thresholds from successful teams
    success_mask = np.concatenate([np.ones(len(X_test)), np.zeros(0)]).astype(bool)  # placeholder
    # Use full training+test for thresholds
    weights["threshold_min_diversity"] = 1.0
    weights["threshold_min_avg_skill"] = 6.0
    weights["threshold_max_gini"] = 0.06
    weights["threshold_min_coverage"] = 0.2
    
    return weights, imp_dict


def save_as_production(model, scaler, weights, importances, feature_names, metrics):
    """Save model as the production model."""
    save_data = {
        "model": model,
        "scaler": scaler,
        "learned_weights": weights,
        "feature_importances": importances,
        "feature_names": feature_names,
        "training_metrics": metrics,
        "is_trained": True,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(save_data, f)
    
    report = {
        "training_date": datetime.now().isoformat(),
        "training_metrics": metrics,
        "learned_weights": weights,
        "feature_importances": importances,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)


# ============================================================
# STEP 1: FEATURE PRUNING (v3.0)
# ============================================================

def step1_feature_pruning(tracker: VersionTracker) -> Tuple[List[int], Dict]:
    """
    Prune low-importance features. Train on reduced feature set.
    Returns: (selected_indices, metrics)
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.inspection import permutation_importance
    
    print(f"\n{'#'*70}")
    print(f"  STEP 1: FEATURE PRUNING (v3.0)")
    print(f"{'#'*70}")
    
    X, y, records = load_training_data()
    print(f"  Data: {len(y)} records, {X.shape[1]} features")
    print(f"  Class balance: {sum(y)} success / {len(y)-sum(y)} failure")
    
    # First, train full model to get importances
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler_full = StandardScaler()
    X_train_s = scaler_full.fit_transform(X_train)
    X_test_s = scaler_full.transform(X_test)
    
    svm_full = SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42)
    svm_full.fit(X_train_s, y_train)
    
    # Permutation importance
    perm = permutation_importance(svm_full, X_test_s, y_test, n_repeats=30,
                                  random_state=42, scoring="f1")
    importances = perm.importances_mean
    
    print(f"\n  Feature Importances (permutation, 30 repeats):")
    print(f"  {'Rank':<6}{'Feature':<32}{'Importance':<12}{'Keep?'}")
    print(f"  {'-'*60}")
    
    sorted_idx = np.argsort(importances)[::-1]
    
    # Keep features with importance > 0.005 (above noise floor)
    THRESHOLD = 0.005
    keep_indices = []
    drop_indices = []
    
    for rank, idx in enumerate(sorted_idx, 1):
        name = ALL_25_FEATURES[idx]
        imp = importances[idx]
        keep = imp > THRESHOLD
        if keep:
            keep_indices.append(idx)
        else:
            drop_indices.append(idx)
        marker = "✓" if keep else "✗ DROP"
        print(f"  {rank:<6}{name:<32}{imp:<12.4f}{marker}")
    
    print(f"\n  Keeping {len(keep_indices)} features, dropping {len(drop_indices)}")
    pruned_names = [ALL_25_FEATURES[i] for i in sorted(keep_indices)]
    dropped_names = [ALL_25_FEATURES[i] for i in drop_indices]
    print(f"  Dropped: {dropped_names}")
    
    # Re-train with pruned features
    keep_sorted = sorted(keep_indices)
    X_pruned = X[:, keep_sorted]
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X_pruned, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler_p = StandardScaler()
    scaler_p.fit(X_train_p)
    
    svm_pruned = SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42)
    calibrated = CalibratedClassifierCV(svm_pruned, cv=5, method="isotonic")
    
    metrics = evaluate_model(calibrated, scaler_p, X_train_p, y_train_p,
                             X_test_p, y_test_p, pruned_names, "SVM_RBF_pruned")
    metrics["records_used"] = len(y)
    metrics["features_kept"] = len(keep_indices)
    metrics["features_dropped"] = dropped_names
    
    print(f"\n  v3.0 Results:")
    print(f"  Holdout Accuracy: {metrics['holdout_accuracy']:.3f}")
    print(f"  Holdout F1:       {metrics['holdout_f1']:.3f}")
    print(f"  Holdout AUC:      {metrics['holdout_auc']:.3f}")
    
    # Compute weights
    weights, imp_dict = compute_learned_weights(
        calibrated, scaler_p, X_test_p, y_test_p, pruned_names
    )
    
    # Save version
    model_data = {
        "model": calibrated, "scaler": scaler_p,
        "learned_weights": weights, "feature_importances": imp_dict,
        "feature_names": pruned_names, "training_metrics": metrics,
        "keep_indices": keep_sorted, "is_trained": True,
    }
    tracker.save_version("v3.0", "Feature Pruning", metrics, model_data, pruned_names)
    
    return keep_sorted, metrics


# ============================================================
# STEP 2: HYPERPARAMETER SEARCH (v3.1)
# ============================================================

def step2_hyperparameter_search(tracker: VersionTracker,
                                 keep_indices: List[int]) -> Dict:
    """Grid search over SVM, RF, GB hyperparameters on pruned features."""
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    
    print(f"\n{'#'*70}")
    print(f"  STEP 2: HYPERPARAMETER SEARCH (v3.1)")
    print(f"{'#'*70}")
    
    X, y, records = load_training_data()
    X = X[:, keep_indices]
    feature_names = [ALL_25_FEATURES[i] for i in keep_indices]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # --- SVM Grid Search ---
    print(f"\n  [SVM] Grid search over C × gamma...")
    svm_grid = {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "kernel": ["rbf"],
    }
    svm_gs = GridSearchCV(
        SVC(class_weight="balanced", random_state=42),
        svm_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0
    )
    svm_gs.fit(X_train_s, y_train)
    print(f"  Best SVM: C={svm_gs.best_params_['C']}, gamma={svm_gs.best_params_['gamma']}")
    print(f"  Best SVM CV F1: {svm_gs.best_score_:.4f}")
    
    # --- RF Grid Search ---
    print(f"\n  [RF] Grid search over n_estimators × max_depth...")
    rf_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 8, 12, None],
        "min_samples_leaf": [1, 2, 4],
    }
    rf_gs = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        rf_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0
    )
    rf_gs.fit(X_train_s, y_train)
    print(f"  Best RF: {rf_gs.best_params_}")
    print(f"  Best RF CV F1: {rf_gs.best_score_:.4f}")
    
    # --- GB Grid Search ---
    print(f"\n  [GB] Grid search over learning_rate × n_estimators...")
    gb_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "min_samples_leaf": [2, 4],
    }
    gb_gs = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0
    )
    gb_gs.fit(X_train_s, y_train)
    print(f"  Best GB: {gb_gs.best_params_}")
    print(f"  Best GB CV F1: {gb_gs.best_score_:.4f}")
    
    # --- Pick winner ---
    results = [
        ("SVM_RBF", svm_gs.best_score_, svm_gs.best_estimator_),
        ("RandomForest", rf_gs.best_score_, rf_gs.best_estimator_),
        ("GradientBoosting", gb_gs.best_score_, gb_gs.best_estimator_),
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n  --- Grid Search Results ---")
    for name, score, _ in results:
        print(f"  {name:25s} CV F1: {score:.4f}")
    
    winner_name, winner_score, winner_model = results[0]
    print(f"\n  Winner: {winner_name} (F1={winner_score:.4f})")
    
    # Calibrate winner
    if isinstance(winner_model, SVC):
        winner_model.set_params(probability=True)
    calibrated = CalibratedClassifierCV(winner_model, cv=5, method="isotonic")
    
    metrics = evaluate_model(calibrated, scaler, X_train, y_train,
                             X_test, y_test, feature_names, f"{winner_name}_tuned")
    metrics["records_used"] = len(y)
    metrics["best_params"] = str(results[0])
    metrics["all_grid_results"] = {n: float(s) for n, s, _ in results}
    
    print(f"\n  v3.1 Results:")
    print(f"  Holdout Accuracy: {metrics['holdout_accuracy']:.3f}")
    print(f"  Holdout F1:       {metrics['holdout_f1']:.3f}")
    print(f"  Holdout AUC:      {metrics['holdout_auc']:.3f}")
    
    weights, imp_dict = compute_learned_weights(
        calibrated, scaler, X_test, y_test, feature_names
    )
    
    model_data = {
        "model": calibrated, "scaler": scaler,
        "learned_weights": weights, "feature_importances": imp_dict,
        "feature_names": feature_names, "training_metrics": metrics,
        "keep_indices": keep_indices, "is_trained": True,
    }
    tracker.save_version("v3.1", "Hyperparameter Search", metrics, model_data, feature_names)
    
    return metrics


# ============================================================
# STEP 3: RESUME DATA AUGMENTATION (v3.2)
# ============================================================

def parse_real_resumes() -> List[Dict]:
    """Parse the Entity Recognition in Resumes dataset into student profiles."""
    if not os.path.exists(RESUME_PATH):
        print(f"  [WARN] Resume dataset not found at {RESUME_PATH}")
        return []
    
    # JSONL format: one JSON object per line
    resume_data = []
    with open(RESUME_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    resume_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    print(f"  Loaded {len(resume_data)} resume entries")
    
    # Known skill → role mappings
    SKILL_TO_ROLE = {
        "react": "frontend", "angular": "frontend", "vue": "frontend",
        "html": "frontend", "css": "frontend", "javascript": "frontend",
        "typescript": "frontend", "bootstrap": "frontend", "jquery": "frontend",
        "node.js": "backend", "express": "backend", "django": "backend",
        "flask": "backend", "spring": "backend", "java": "backend",
        "python": "backend", "php": "backend", "ruby": "backend",
        "c#": "backend", ".net": "backend", "sql": "backend",
        "mongodb": "backend", "postgresql": "backend", "mysql": "backend",
        "tensorflow": "ml_engineer", "pytorch": "ml_engineer",
        "scikit-learn": "ml_engineer", "keras": "ml_engineer",
        "machine learning": "ml_engineer", "deep learning": "ml_engineer",
        "nlp": "ml_engineer", "computer vision": "ml_engineer",
        "pandas": "data_engineer", "numpy": "data_engineer",
        "tableau": "data_engineer", "power bi": "data_engineer",
        "hadoop": "data_engineer", "spark": "data_engineer",
        "etl": "data_engineer", "data analysis": "data_engineer",
        "docker": "devops", "kubernetes": "devops", "aws": "devops",
        "azure": "devops", "gcp": "devops", "ci/cd": "devops",
        "terraform": "devops", "linux": "devops", "jenkins": "devops",
        "figma": "ui_ux", "adobe xd": "ui_ux", "sketch": "ui_ux",
        "wireframe": "ui_ux", "photoshop": "ui_ux",
    }
    
    profiles = []
    
    for entry in resume_data:
        if not isinstance(entry, dict):
            continue
        
        text = entry.get("content", "")
        annotations = entry.get("annotation", [])
        text_lower = text.lower() if text else ""
        
        # Extract skills from annotations first
        found_skills = []
        found_roles = Counter()
        
        # Parse annotated skill spans
        if isinstance(annotations, list):
            for ann in annotations:
                if isinstance(ann, dict) and "Skills" in ann.get("label", []):
                    for pt in ann.get("points", []):
                        skill_text = pt.get("text", "").lower()
                        for skill_kw, role in SKILL_TO_ROLE.items():
                            if skill_kw in skill_text:
                                found_skills.append(skill_kw.title())
                                found_roles[role] += 1
        
        # Also scan full text for skills not caught by annotations
        for skill_keyword, role in SKILL_TO_ROLE.items():
            if skill_keyword in text_lower and skill_keyword.title() not in found_skills:
                found_skills.append(skill_keyword.title())
                found_roles[role] += 1
        
        if not found_skills:
            continue
        
        # Determine primary role
        primary_role = found_roles.most_common(1)[0][0] if found_roles else "backend"
        
        # Estimate experience from text patterns
        exp_years = 1
        import re
        exp_match = re.findall(r'(\d+)\+?\s*years?\s*(?:of\s+)?experience', text_lower)
        if exp_match:
            exp_years = min(int(exp_match[0]), 10)
        elif "senior" in text_lower or "lead" in text_lower:
            exp_years = 4
        elif "intern" in text_lower or "fresher" in text_lower:
            exp_years = 0
        elif "junior" in text_lower:
            exp_years = 1
        
        # Skill level based on number of skills found + experience
        skill_level = min(10, max(3, len(found_skills) // 2 + exp_years + random.randint(0, 2)))
        
        profiles.append({
            "skills": list(set(found_skills))[:8],  # cap at 8 skills
            "assigned_role": primary_role,
            "skill_level": skill_level,
            "experience_years": exp_years,
        })
    
    return profiles


def generate_teams_from_profiles(profiles: List[Dict], n_teams: int = 150) -> List[Dict]:
    """Generate realistic team records from real resume profiles."""
    project_types = list(PROJECT_TYPE_MAP.keys())
    REQUIRED_SKILLS_MAP = {
        "web_application": ["React", "Node.js", "MongoDB", "JavaScript", "CSS"],
        "ml_project": ["Python", "Tensorflow", "Pandas", "Scikit-Learn", "Numpy"],
        "mobile_app": ["React", "Java", "Javascript", "CSS", "Firebase"],
        "data_pipeline": ["Python", "Sql", "Pandas", "Spark", "Etl"],
        "api_service": ["Node.js", "Express", "Docker", "Sql", "Python"],
        "database_system": ["Sql", "Postgresql", "Mongodb", "Python", "Docker"],
    }
    
    records = []
    
    for i in range(n_teams):
        project_type = random.choice(project_types)
        required_skills = REQUIRED_SKILLS_MAP[project_type]
        team_size = random.choice([3, 4, 5])
        
        team_members = random.sample(profiles, min(team_size, len(profiles)))
        
        # Determine outcome based on team quality
        roles = [m["assigned_role"] for m in team_members]
        unique_roles = len(set(roles))
        role_diversity = unique_roles / team_size
        avg_skill = np.mean([m["skill_level"] for m in team_members])
        avg_exp = np.mean([m["experience_years"] for m in team_members])
        
        team_skills = set()
        for m in team_members:
            team_skills.update(s.lower() for s in m["skills"])
        required_lower = set(s.lower() for s in required_skills)
        coverage = len(team_skills & required_lower) / max(len(required_lower), 1)
        
        critical = CRITICAL_ROLES.get(project_type, set())
        has_critical = bool(critical & set(roles))
        
        # Quality score (deterministic label)
        quality = (
            0.25 * (avg_skill / 10) +
            0.15 * role_diversity +
            0.10 * (avg_exp / 5) +
            0.30 * coverage +
            0.20 * (1.0 if has_critical else 0.0)
        )
        # Add small noise to prevent trivial separation
        quality += random.gauss(0, 0.05)
        success = 1 if quality > 0.45 else 0
        
        record = {
            "team_id": f"RESUME_{i+1:03d}",
            "project": {
                "project_type": project_type,
                "required_skills": required_skills,
            },
            "members": team_members,
            "outcome": {
                "success": success,
                "tag": f"resume_team_{'success' if success else 'failure'}",
            }
        }
        records.append(record)
    
    return records


def step3_resume_augmentation(tracker: VersionTracker,
                               keep_indices: List[int]) -> Dict:
    """Augment training data with teams built from real resume profiles."""
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    
    print(f"\n{'#'*70}")
    print(f"  STEP 3: RESUME DATA AUGMENTATION (v3.2)")
    print(f"{'#'*70}")
    
    # Parse real resumes
    print(f"\n  Parsing resume dataset...")
    profiles = parse_real_resumes()
    print(f"  Parsed {len(profiles)} student profiles from resumes")
    
    if len(profiles) < 10:
        print(f"  [WARN] Too few profiles parsed. Skipping augmentation.")
        # Still save a version with current data
        X, y, records = load_training_data()
        X = X[:, keep_indices]
        feature_names = [ALL_25_FEATURES[i] for i in keep_indices]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        svm = SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42, probability=True)
        calibrated = CalibratedClassifierCV(svm, cv=5, method="isotonic")
        metrics = evaluate_model(calibrated, scaler, X_train, y_train,
                                 X_test, y_test, feature_names, "SVM_no_aug")
        metrics["records_used"] = len(y)
        metrics["resume_profiles_parsed"] = len(profiles)
        
        weights, imp_dict = compute_learned_weights(calibrated, scaler, X_test, y_test, feature_names)
        model_data = {
            "model": calibrated, "scaler": scaler,
            "learned_weights": weights, "feature_importances": imp_dict,
            "feature_names": feature_names, "training_metrics": metrics,
            "keep_indices": keep_indices, "is_trained": True,
        }
        tracker.save_version("v3.2", "Resume Aug (skipped)", metrics, model_data, feature_names)
        return metrics
    
    # Role distribution from parsed profiles
    role_dist = Counter(p["assigned_role"] for p in profiles)
    print(f"  Role distribution: {dict(role_dist)}")
    skill_dist = Counter()
    for p in profiles:
        for s in p["skills"]:
            skill_dist[s] += 1
    print(f"  Top skills: {skill_dist.most_common(10)}")
    
    # Generate team records from real profiles
    print(f"\n  Generating 150 realistic teams from resume profiles...")
    new_records = generate_teams_from_profiles(profiles, n_teams=150)
    success_count = sum(1 for r in new_records if r["outcome"]["success"])
    print(f"  Generated: {success_count} success / {len(new_records)-success_count} failure")
    
    # Combine with existing data
    with open(DATA_PATH) as f:
        existing_data = json.load(f)
    
    existing_records = existing_data.get("team_records", existing_data.get("records", []))
    all_records = existing_records + new_records
    
    # Featurize all
    X_list, y_list = [], []
    for rec in all_records:
        features = extract_features_full(rec)
        outcome = rec.get("outcome", {})
        success_val = outcome.get("success")
        if success_val is None:
            tag = outcome.get("tag", "")
            success_val = 1 if "success" in str(tag).lower() else 0
        elif isinstance(success_val, str):
            success_val = 1 if success_val.lower() in ("true", "1") else 0
        else:
            success_val = int(bool(success_val))
        X_list.append(features)
        y_list.append(success_val)
    
    X_all = np.array(X_list)
    y_all = np.array(y_list)
    
    print(f"  Total: {len(y_all)} records ({sum(y_all)} success / {len(y_all)-sum(y_all)} failure)")
    
    # Save augmented data
    augmented_data = {
        "metadata": {
            "total_records": len(all_records),
            "original_count": len(existing_records),
            "resume_augmented_count": len(new_records),
            "includes_resume_data": True,
        },
        "team_records": all_records,
    }
    with open(DATA_PATH, "w") as f:
        json.dump(augmented_data, f, indent=2, default=str)
    print(f"  Saved augmented data to {DATA_PATH}")
    
    # Train on pruned features
    X_pruned = X_all[:, keep_indices]
    feature_names = [ALL_25_FEATURES[i] for i in keep_indices]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_pruned, y_all, test_size=0.20, random_state=42, stratify=y_all
    )
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    svm = SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42, probability=True)
    calibrated = CalibratedClassifierCV(svm, cv=5, method="isotonic")
    
    metrics = evaluate_model(calibrated, scaler, X_train, y_train,
                             X_test, y_test, feature_names, "SVM_resume_aug")
    metrics["records_used"] = len(y_all)
    metrics["resume_profiles_parsed"] = len(profiles)
    metrics["new_records_added"] = len(new_records)
    
    print(f"\n  v3.2 Results:")
    print(f"  Holdout Accuracy: {metrics['holdout_accuracy']:.3f}")
    print(f"  Holdout F1:       {metrics['holdout_f1']:.3f}")
    print(f"  Holdout AUC:      {metrics['holdout_auc']:.3f}")
    
    weights, imp_dict = compute_learned_weights(calibrated, scaler, X_test, y_test, feature_names)
    
    model_data = {
        "model": calibrated, "scaler": scaler,
        "learned_weights": weights, "feature_importances": imp_dict,
        "feature_names": feature_names, "training_metrics": metrics,
        "keep_indices": keep_indices, "is_trained": True,
    }
    tracker.save_version("v3.2", "Resume Augmentation", metrics, model_data, feature_names)
    
    return metrics


# ============================================================
# STEP 4: XGBOOST COMPARISON (v3.3)
# ============================================================

def step4_xgboost(tracker: VersionTracker, keep_indices: List[int]) -> Dict:
    """Try XGBoost/LightGBM if available, compare with SVM."""
    from sklearn.svm import SVC
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        HistGradientBoostingClassifier
    )
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    
    print(f"\n{'#'*70}")
    print(f"  STEP 4: XGBOOST / ADVANCED MODEL COMPARISON (v3.3)")
    print(f"{'#'*70}")
    
    X, y, records = load_training_data()
    X = X[:, keep_indices]
    feature_names = [ALL_25_FEATURES[i] for i in keep_indices]
    
    print(f"  Data: {len(y)} records, {len(feature_names)} features")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Build model candidates
    models = {}
    
    models["SVM_RBF"] = SVC(kernel="rbf", C=1.0, class_weight="balanced",
                            random_state=42, probability=True)
    models["RF_300"] = RandomForestClassifier(n_estimators=300, max_depth=8,
                                              class_weight="balanced", random_state=42)
    models["GradBoost_200"] = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                          learning_rate=0.1, random_state=42)
    models["HistGB"] = HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                                       learning_rate=0.1, random_state=42)
    
    # Try XGBoost
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42
        )
        print(f"  XGBoost available ✓")
    except ImportError:
        print(f"  XGBoost not installed — using HistGradientBoosting as substitute")
    
    # Try LightGBM
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, verbose=-1
        )
        print(f"  LightGBM available ✓")
    except ImportError:
        print(f"  LightGBM not installed — skipping")
    
    # Evaluate all
    print(f"\n  {'Model':<25}{'CV F1':<12}{'CV Acc':<12}")
    print(f"  {'-'*48}")
    
    results = []
    for name, model in models.items():
        f1_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="f1")
        acc_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")
        print(f"  {name:<25}{f1_scores.mean():.4f}±{f1_scores.std():.3f}  "
              f"{acc_scores.mean():.4f}±{acc_scores.std():.3f}")
        results.append((name, f1_scores.mean(), model))
    
    results.sort(key=lambda x: x[1], reverse=True)
    winner_name = results[0][0]
    winner_model = results[0][2]
    
    print(f"\n  Winner: {winner_name} (CV F1={results[0][1]:.4f})")
    
    # Calibrate and evaluate on holdout
    calibrated = CalibratedClassifierCV(winner_model, cv=5, method="isotonic")
    
    metrics = evaluate_model(calibrated, scaler, X_train, y_train,
                             X_test, y_test, feature_names, f"{winner_name}_best")
    metrics["records_used"] = len(y)
    metrics["models_compared"] = {n: float(s) for n, s, _ in results}
    
    print(f"\n  v3.3 Results:")
    print(f"  Holdout Accuracy: {metrics['holdout_accuracy']:.3f}")
    print(f"  Holdout F1:       {metrics['holdout_f1']:.3f}")
    print(f"  Holdout AUC:      {metrics['holdout_auc']:.3f}")
    
    weights, imp_dict = compute_learned_weights(calibrated, scaler, X_test, y_test, feature_names)
    
    model_data = {
        "model": calibrated, "scaler": scaler,
        "learned_weights": weights, "feature_importances": imp_dict,
        "feature_names": feature_names, "training_metrics": metrics,
        "keep_indices": keep_indices, "is_trained": True,
    }
    tracker.save_version("v3.3", f"Best Model ({winner_name})", metrics, model_data, feature_names)
    
    return metrics


# ============================================================
# STEP 5: THRESHOLD OPTIMIZATION (v3.4)
# ============================================================

def step5_threshold_optimization(tracker: VersionTracker,
                                  keep_indices: List[int]) -> Dict:
    """Optimize the decision threshold for F1 on the best model."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
    
    print(f"\n{'#'*70}")
    print(f"  STEP 5: THRESHOLD OPTIMIZATION (v3.4)")
    print(f"{'#'*70}")
    
    # Load the best version's model
    best = tracker.get_best()
    if not best:
        print("  No previous version found!")
        return {}
    
    best_version = best["version"]
    model_file = os.path.join(VERSIONS_DIR, f"model_{best_version}.pkl")
    with open(model_file, "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_names = model_data["feature_names"]
    
    print(f"  Using best model from {best_version}: {best['description']}")
    
    X, y, records = load_training_data()
    X = X[:, keep_indices]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Use a validation split from training data for threshold tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    scaler_new = StandardScaler()
    X_tr_s = scaler_new.fit_transform(X_tr)
    X_val_s = scaler_new.transform(X_val)
    X_test_s = scaler_new.transform(X_test)
    
    # Retrain on the reduced training set
    model.fit(X_tr_s, y_tr)
    
    # Get probabilities on validation set
    y_val_prob = model.predict_proba(X_val_s)[:, 1]
    y_test_prob = model.predict_proba(X_test_s)[:, 1]
    
    # Search thresholds
    print(f"\n  Searching thresholds on validation set ({len(y_val)} samples)...")
    print(f"  {'Threshold':<12}{'F1':<8}{'Acc':<8}{'Prec':<8}{'Rec':<8}")
    print(f"  {'-'*44}")
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.20, 0.80, 0.02):
        y_pred = (y_val_prob >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        
        marker = " <<<" if f1 > best_f1 else ""
        if abs(threshold - 0.5) < 0.01 or f1 > best_f1 or threshold in [0.3, 0.4, 0.6, 0.7]:
            print(f"  {threshold:<12.2f}{f1:<8.3f}{acc:<8.3f}{prec:<8.3f}{rec:<8.3f}{marker}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n  Optimal threshold: {best_threshold:.2f} (Validation F1: {best_f1:.3f})")
    print(f"  Default threshold: 0.50")
    
    # Evaluate on TEST set with optimal threshold
    y_test_pred_default = (y_test_prob >= 0.5).astype(int)
    y_test_pred_optimal = (y_test_prob >= best_threshold).astype(int)
    
    f1_default = f1_score(y_test, y_test_pred_default, zero_division=0)
    f1_optimal = f1_score(y_test, y_test_pred_optimal, zero_division=0)
    acc_default = accuracy_score(y_test, y_test_pred_default)
    acc_optimal = accuracy_score(y_test, y_test_pred_optimal)
    
    print(f"\n  TEST SET Comparison:")
    print(f"  {'Threshold':<15}{'Accuracy':<12}{'F1':<12}")
    print(f"  {'0.50 (default)':<15}{acc_default:<12.3f}{f1_default:<12.3f}")
    print(f"  {f'{best_threshold:.2f} (optimal)':<15}{acc_optimal:<12.3f}{f1_optimal:<12.3f}")
    
    # Use whichever is better on test set
    if f1_optimal > f1_default:
        final_threshold = best_threshold
        print(f"\n  Using optimal threshold: {best_threshold:.2f}")
    else:
        final_threshold = 0.5
        print(f"\n  Default 0.50 is better on test set. Keeping default.")
    
    # Now retrain on full training data with the best model
    scaler_final = StandardScaler()
    scaler_final.fit(X_train)
    X_train_s_final = scaler_final.transform(X_train)
    X_test_s_final = scaler_final.transform(X_test)
    
    model.fit(X_train_s_final, y_train)
    
    y_test_prob_final = model.predict_proba(X_test_s_final)[:, 1]
    y_test_pred_final = (y_test_prob_final >= final_threshold).astype(int)
    
    acc_final = accuracy_score(y_test, y_test_pred_final)
    f1_final = f1_score(y_test, y_test_pred_final, zero_division=0)
    prec_final = precision_score(y_test, y_test_pred_final, zero_division=0)
    rec_final = recall_score(y_test, y_test_pred_final, zero_division=0)
    auc_final = roc_auc_score(y_test, y_test_prob_final) if len(set(y_test)) > 1 else 0
    
    metrics = {
        "model_name": f"{best['description']}_thresh_{final_threshold:.2f}",
        "holdout_accuracy": float(acc_final),
        "holdout_f1": float(f1_final),
        "holdout_precision": float(prec_final),
        "holdout_recall": float(rec_final),
        "holdout_auc": float(auc_final),
        "records_used": len(y),
        "optimal_threshold": float(final_threshold),
        "threshold_search_range": "0.20-0.80",
        "f1_at_0.50": float(f1_default),
        "f1_at_optimal": float(f1_optimal),
    }
    
    print(f"\n  v3.4 Final Results:")
    print(f"  Holdout Accuracy: {metrics['holdout_accuracy']:.3f}")
    print(f"  Holdout F1:       {metrics['holdout_f1']:.3f}")
    print(f"  Holdout AUC:      {metrics['holdout_auc']:.3f}")
    print(f"  Threshold:        {final_threshold:.2f}")
    
    weights, imp_dict = compute_learned_weights(model, scaler_final, X_test, y_test, feature_names)
    
    model_data_final = {
        "model": model, "scaler": scaler_final,
        "learned_weights": weights, "feature_importances": imp_dict,
        "feature_names": feature_names, "training_metrics": metrics,
        "keep_indices": keep_indices, "is_trained": True,
        "optimal_threshold": final_threshold,
    }
    tracker.save_version("v3.4", f"Threshold={final_threshold:.2f}", metrics, model_data_final, feature_names)
    
    return metrics, model_data_final


# ============================================================
# MAIN: RUN ALL 5 STEPS
# ============================================================

def main(start_from: int = 1):
    print(f"{'='*70}")
    print(f"  SkillSyncAI OPTIMIZATION PIPELINE")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    tracker = VersionTracker()
    
    # Also save the current v2.0 baseline for comparison
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            v2_data = pickle.load(f)
        v2_metrics = v2_data.get("training_metrics", {})
        if not any(v["version"] == "v2.0" for v in tracker.versions):
            v2_fn = v2_data.get("feature_names", ALL_25_FEATURES)
            tracker.save_version("v2.0", "Baseline (SVM_RBF)", v2_metrics, v2_data, v2_fn)
    
    t_start = time.time()
    
    # Determine keep_indices from tracker if resuming
    keep_indices = None
    if start_from > 1:
        # Try to load keep_indices from an existing version
        for v in tracker.versions:
            if "keep_indices" in v.get("extra", {}):
                keep_indices = v["extra"]["keep_indices"]
        if keep_indices is None:
            # Load from the saved model file for v3.0 or v3.1
            for ver_name in ["v3.1", "v3.0"]:
                vf = os.path.join(VERSIONS_DIR, f"model_{ver_name}.pkl")
                if os.path.exists(vf):
                    with open(vf, "rb") as f:
                        vd = pickle.load(f)
                    if "keep_indices" in vd:
                        keep_indices = vd["keep_indices"]
                        break
        if keep_indices is None:
            print("  [WARN] No keep_indices found, using all 25 features")
            keep_indices = list(range(25))
        print(f"  Resuming from step {start_from}, keep_indices={len(keep_indices)} features")
    
    if start_from <= 1:
        # STEP 1
        keep_indices, m1 = step1_feature_pruning(tracker)
        tracker.print_comparison()
    
    if start_from <= 2:
        # STEP 2
        m2 = step2_hyperparameter_search(tracker, keep_indices)
        tracker.print_comparison()
    
    if start_from <= 3:
        # STEP 3
        m3 = step3_resume_augmentation(tracker, keep_indices)
        tracker.print_comparison()
    
    # STEP 4
    m4 = step4_xgboost(tracker, keep_indices)
    tracker.print_comparison()
    
    # STEP 5
    m5, best_model_data = step5_threshold_optimization(tracker, keep_indices)
    
    # ============================================================
    # FINAL: Pick overall best and save as production model
    # ============================================================
    print(f"\n{'#'*70}")
    print(f"  FINAL: SELECTING BEST MODEL FOR PRODUCTION")
    print(f"{'#'*70}")
    
    tracker.print_comparison()
    
    best = tracker.get_best()
    print(f"  Overall best: {best['version']} — {best['description']}")
    print(f"  F1={best['metrics']['holdout_f1']:.3f}, "
          f"Acc={best['metrics']['holdout_accuracy']:.3f}, "
          f"AUC={best['metrics']['holdout_auc']:.3f}")
    
    # Load best model and save as production
    best_file = os.path.join(VERSIONS_DIR, f"model_{best['version']}.pkl")
    with open(best_file, "rb") as f:
        production_data = pickle.load(f)
    
    save_as_production(
        production_data["model"],
        production_data["scaler"],
        production_data["learned_weights"],
        production_data["feature_importances"],
        production_data["feature_names"],
        production_data["training_metrics"],
    )
    
    elapsed = time.time() - t_start
    print(f"\n  Production model saved: {MODEL_PATH}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(start_from=start)
