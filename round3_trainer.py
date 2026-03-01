"""
SkillSyncAI — Training Round 3: Deterministic Edge Cases
=========================================================

Round 1 (v2.0): 94.2% holdout, AUC 0.996 — excellent
Round 2 (v2.1): 78.0% — degraded due to label noise from random outcomes

Round 3 strategy: Keep Round 1 base (256 records). Add 100 more records
where labels are DETERMINISTIC (computed from a quality formula, no randomness).
This ensures no label noise while adding genuine difficulty.
"""

import json
import os
import random
import pickle
import math
import time
import numpy as np

from rigorous_trainer import (
    extract_features_v2, extract_batch_v2, FEATURE_NAMES,
    SKILL_POOLS, PROJECT_CONFIGS, NAMES,
    _make_member, _pick_required, DATA_PATH, MODEL_PATH, REPORT_PATH,
    CRITICAL_ROLES as CRIT_ROLES_MAP,
)

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score, log_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from collections import Counter

random.seed(777)
np.random.seed(777)

ALL_ROLES = list(SKILL_POOLS.keys())
PROJECT_TYPES = list(PROJECT_CONFIGS.keys())


def compute_team_quality(members, project_type):
    """
    Deterministic quality score for a team.
    Returns a float 0-100. Teams scoring >= 55 are labeled success.
    """
    if not members:
        return 0

    skill_levels = [m["skill_level"] for m in members]
    experiences = [m.get("experience_years", 0) for m in members]
    roles = [m["assigned_role"] for m in members]

    avg_skill = sum(skill_levels) / len(skill_levels)
    min_skill = min(skill_levels)

    # Role coverage
    critical = CRIT_ROLES_MAP.get(project_type, set())
    roles_set = set(roles)
    if critical:
        critical_coverage = len(critical & roles_set) / len(critical)
    else:
        critical_coverage = 1.0

    # Unique role ratio
    unique_roles = len(set(roles))
    role_diversity = unique_roles / len(members)

    # Skill spread penalty
    skill_range = max(skill_levels) - min(skill_levels)
    spread_penalty = min(skill_range * 2, 15)  # max -15 for huge spread

    # Role duplication penalty
    role_counts = Counter(roles)
    dups = sum(c - 1 for c in role_counts.values() if c > 1)
    dup_penalty = dups * 5  # -5 per duplicate

    # Experience bonus (diminishing returns)
    avg_exp = sum(experiences) / len(experiences)
    exp_bonus = min(avg_exp * 2, 10)  # max +10

    # Team size factor (penalty for too large)
    size = len(members)
    size_factor = 0
    if size > 5:
        size_factor = -(size - 5) * 3
    elif size < 3:
        size_factor = -(3 - size) * 2

    # Skill coverage (how many project skills the team covers)
    helpful = set(PROJECT_CONFIGS.get(project_type, {}).get("helpful_roles", []))
    helpful_coverage = len(helpful & roles_set) / max(len(helpful), 1) if helpful else 0.5

    # Composite score
    score = (
        avg_skill * 7                  # 0-70 (skill is primary)
        + critical_coverage * 20       # 0-20 (covering critical roles is vital)
        + helpful_coverage * 5         # 0-5 (helpful roles add minor value)
        + exp_bonus                    # 0-10
        + role_diversity * 5           # 0-5
        - spread_penalty               # 0 to -15
        - dup_penalty                  # 0 to -20+
        + size_factor                  # -6 to 0
        + min_skill * 1.5             # 0-15 (weakest link matters)
    )
    return max(0, min(100, score))


def generate_deterministic_edge_cases(n=100):
    """
    Generate team records where labels are deterministic: computed from a
    quality formula. No randomness in label assignment.
    """
    records = []
    counter = 800

    def make_rec(members, pt, tag):
        nonlocal counter
        counter += 1
        cfg = PROJECT_CONFIGS[pt]
        req = random.sample(cfg["required_skills_pool"],
                            min(random.randint(3, 5), len(cfg["required_skills_pool"])))

        quality = compute_team_quality(members, pt)
        success = quality >= 55  # deterministic threshold

        if quality >= 80:
            grade = "A"
        elif quality >= 70:
            grade = "B+"
        elif quality >= 60:
            grade = "B"
        elif quality >= 55:
            grade = "B-"
        elif quality >= 45:
            grade = "C+"
        elif quality >= 35:
            grade = "C"
        else:
            grade = "D"

        return {
            "team_id": f"DET_{counter:03d}",
            "project": {
                "project_type": pt,
                "project_name": random.choice(cfg["names"]),
                "required_skills": req,
            },
            "members": members,
            "outcome": {
                "success": success,
                "grade": grade,
                "score": round(quality, 1),
                "completion_status": "completed_on_time" if quality > 70 else
                                     "completed_late" if quality > 50 else "incomplete",
                "tag": tag,
                "quality_score": round(quality, 1),
            }
        }

    # ── Sweep skill levels with good fit ──────────────────
    for skill in range(3, 10):
        for pt in PROJECT_TYPES:
            critical = list(PROJECT_CONFIGS[pt]["critical_roles"])
            helpful = list(PROJECT_CONFIGS[pt]["helpful_roles"])
            roles = critical + helpful[:1] if helpful else critical
            members = [_make_member(r, skill, random.randint(1, 2)) for r in roles]
            records.append(make_rec(members, pt, f"skill_sweep_{skill}"))

    # ── Sweep skill levels with bad fit ───────────────────
    for skill in range(4, 9):
        for pt in PROJECT_TYPES:
            wrong_roles = [r for r in ALL_ROLES if r not in PROJECT_CONFIGS[pt]["critical_roles"]]
            roles = random.sample(wrong_roles, min(3, len(wrong_roles)))
            members = [_make_member(r, skill, random.randint(1, 2)) for r in roles]
            records.append(make_rec(members, pt, f"bad_fit_skill_{skill}"))

    # ── Mixed skill levels in same team ───────────────────
    for spread in [(3, 8), (4, 7), (5, 6), (2, 9), (4, 8), (5, 7)]:
        for pt in random.sample(PROJECT_TYPES, 3):
            critical = list(PROJECT_CONFIGS[pt]["critical_roles"])
            helpful = list(PROJECT_CONFIGS[pt]["helpful_roles"])
            roles = critical + helpful[:1] if helpful else critical
            members = []
            for i, r in enumerate(roles):
                skill = spread[i % 2]
                members.append(_make_member(r, skill, random.randint(1, 3)))
            records.append(make_rec(members, pt, f"mixed_{spread[0]}_{spread[1]}"))

    # ── Team size variations ──────────────────────────────
    for size in [2, 3, 4, 5, 6]:
        for pt in random.sample(PROJECT_TYPES, 3):
            critical = list(PROJECT_CONFIGS[pt]["critical_roles"])
            helpful = list(PROJECT_CONFIGS[pt]["helpful_roles"])
            all_roles_for_pt = critical + helpful
            roles = (all_roles_for_pt * 3)[:size]  # fill to required size
            members = [_make_member(r, random.randint(6, 7), random.randint(1, 3)) for r in roles]
            records.append(make_rec(members, pt, f"size_{size}"))

    # ── Role duplication sweep ────────────────────────────
    for dup_level in [1, 2, 3]:
        for pt in random.sample(PROJECT_TYPES, 3):
            critical = list(PROJECT_CONFIGS[pt]["critical_roles"])
            base_role = critical[0] if critical else "backend"
            roles = [base_role] * (1 + dup_level) + [random.choice(ALL_ROLES)]
            members = [_make_member(r, 7, 2) for r in roles]
            records.append(make_rec(members, pt, f"dup_{dup_level}"))

    # ── Experience sweep ──────────────────────────────────
    for exp in [0, 1, 2, 3, 4, 5]:
        for pt in random.sample(PROJECT_TYPES, 2):
            critical = list(PROJECT_CONFIGS[pt]["critical_roles"])
            helpful = list(PROJECT_CONFIGS[pt]["helpful_roles"])
            roles = critical + helpful[:1] if helpful else critical
            members = [_make_member(r, 6, exp) for r in roles]
            records.append(make_rec(members, pt, f"exp_{exp}"))

    # ── Critical role partial coverage ────────────────────
    for pt in PROJECT_TYPES:
        critical = list(PROJECT_CONFIGS[pt]["critical_roles"])
        if len(critical) >= 2:
            # Only cover first critical role
            roles = [critical[0], random.choice(PROJECT_CONFIGS[pt]["helpful_roles"]),
                     random.choice(ALL_ROLES)]
            members = [_make_member(r, random.randint(6, 8), 2) for r in roles]
            records.append(make_rec(members, pt, "partial_critical"))

    random.shuffle(records)
    return records[:n]


def run_round3():
    start = time.time()

    print("\n" + "=" * 70)
    print("  TRAINING ROUND 3: DETERMINISTIC EDGE CASES")
    print("=" * 70)

    # Load and filter to only original + round 1 EDGE_ records
    with open(DATA_PATH, "r") as f:
        current = json.load(f)
    existing = [r for r in current.get("team_records", [])
                if not r.get("team_id", "").startswith("EDGE2_")
                and not r.get("team_id", "").startswith("DET_")]
    print(f"  Base records (original + round 1): {len(existing)}")

    # Generate deterministic edge cases
    det_cases = generate_deterministic_edge_cases(100)
    tags = Counter(r["outcome"].get("tag", "").split("_")[0] for r in det_cases)
    det_success = sum(1 for r in det_cases if r["outcome"]["success"])
    print(f"  Deterministic edge cases: {len(det_cases)} ({det_success} success, {len(det_cases)-det_success} fail)")

    all_records = existing + det_cases
    success_count = sum(1 for r in all_records if r["outcome"]["success"])
    fail_count = len(all_records) - success_count
    print(f"  Total dataset:  {len(all_records)}")
    print(f"  Success / Fail: {success_count} / {fail_count}")

    # Save
    merged = {
        "metadata": {
            "total_records": len(all_records),
            "generated_date": "2026-03-02",
            "purpose": "feedback_loop_training_v2.2_round3",
            "success_count": success_count,
            "failure_count": fail_count,
            "includes_edge_cases": True,
            "deterministic_added": len(det_cases),
        },
        "team_records": all_records,
    }
    with open(DATA_PATH, "w") as f:
        json.dump(merged, f, indent=2)

    # Extract
    X, y = extract_batch_v2(all_records)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"  Train / Test: {len(y_train)} / {len(y_test)}")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # ── Model Comparison ─────────────────────────────────
    print(f"\n  ─── Model Comparison ───")
    models = {
        "RF(300,d10)": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=2,
            class_weight="balanced", random_state=42),
        "RF(500,d12)": RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=42),
        "GB(200,d5,lr0.1)": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        "GB(300,d4,lr0.05)": GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42),
        "SVM_RBF(C1)": SVC(
            kernel="rbf", C=1.0, probability=True,
            class_weight="balanced", random_state=42),
        "SVM_RBF(C10)": SVC(
            kernel="rbf", C=10.0, probability=True,
            class_weight="balanced", random_state=42),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=2,
            class_weight="balanced", random_state=42),
    }

    best_f1 = 0
    best_name = None
    results = {}
    for name, model in models.items():
        f1_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="f1")
        acc_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")
        auc_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="roc_auc")
        results[name] = {
            "cv_f1": float(f1_scores.mean()),
            "cv_acc": float(acc_scores.mean()),
            "cv_auc": float(auc_scores.mean()),
        }
        print(f"  {name:25s}  Acc={acc_scores.mean():.3f}  F1={f1_scores.mean():.3f}  AUC={auc_scores.mean():.3f}")
        if f1_scores.mean() > best_f1:
            best_f1 = f1_scores.mean()
            best_name = name

    print(f"\n  Best single: {best_name} (F1={best_f1:.3f})")

    # ── Stacking ─────────────────────────────────────────
    print(f"\n  ─── Stacking Ensemble ───")
    stacker = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=10,
                                          class_weight="balanced", random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                              learning_rate=0.1, random_state=42)),
            ("svm", SVC(kernel="rbf", C=1.0, probability=True,
                        class_weight="balanced", random_state=42)),
            ("et", ExtraTreesClassifier(n_estimators=200, max_depth=10,
                                         class_weight="balanced", random_state=42)),
        ],
        final_estimator=LogisticRegression(C=10.0, max_iter=1000),
        cv=5,
        stack_method="predict_proba",
        passthrough=True,
    )
    stack_f1 = cross_val_score(stacker, X_train_s, y_train, cv=cv, scoring="f1")
    print(f"  Stacking CV F1: {stack_f1.mean():.3f} ± {stack_f1.std():.3f}")

    # ── Voting ───────────────────────────────────────────
    print(f"\n  ─── Voting Ensemble ───")
    voter = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=10,
                                          class_weight="balanced", random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                              learning_rate=0.1, random_state=42)),
            ("svm", SVC(kernel="rbf", C=1.0, probability=True,
                        class_weight="balanced", random_state=42)),
        ],
        voting="soft",
    )
    vote_f1 = cross_val_score(voter, X_train_s, y_train, cv=cv, scoring="f1")
    print(f"  Voting CV F1: {vote_f1.mean():.3f} ± {vote_f1.std():.3f}")

    # Pick winner
    candidates = {
        best_name: (models[best_name], best_f1),
        "Stacking": (stacker, stack_f1.mean()),
        "Voting": (voter, vote_f1.mean()),
    }
    winner_name = max(candidates, key=lambda k: candidates[k][1])
    winner_model, winner_cv_f1 = candidates[winner_name]
    print(f"\n  >>> WINNER: {winner_name} (CV F1={winner_cv_f1:.3f})")

    # ── Holdout ──────────────────────────────────────────
    print(f"\n  ─── Holdout Evaluation ({winner_name}) ───")
    winner_model.fit(X_train_s, y_train)
    y_pred = winner_model.predict(X_test_s)
    y_prob = winner_model.predict_proba(X_test_s)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred)
    test_rec = recall_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)
    test_ll = log_loss(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy:   {test_acc:.3f}")
    print(f"  F1:         {test_f1:.3f}")
    print(f"  Precision:  {test_prec:.3f}")
    print(f"  Recall:     {test_rec:.3f}")
    print(f"  ROC-AUC:    {test_auc:.3f}")
    print(f"  Log Loss:   {test_ll:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"    Predicted:  Fail  Success")
    print(f"    Actual Fail:  {cm[0][0]:3d}    {cm[0][1]:3d}")
    print(f"    Actual Pass:  {cm[1][0]:3d}    {cm[1][1]:3d}")
    print(f"\n" + classification_report(y_test, y_pred, target_names=["Failure", "Success"]))

    # ── Multi-seed stability ─────────────────────────────
    print(f"  ─── Multi-Seed Stability ───")
    seed_scores = []
    for seed in [0, 7, 13, 21, 42, 99, 137, 256, 512, 1024]:
        cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        s = cross_val_score(winner_model, X_train_s, y_train, cv=cv_s, scoring="f1")
        seed_scores.append(s.mean())
    seed_scores = np.array(seed_scores)
    stability = "STABLE" if seed_scores.std() < 0.05 else "UNSTABLE"
    print(f"  F1 across 10 seeds: {seed_scores.mean():.3f} ± {seed_scores.std():.3f} ({stability})")

    # ── Calibrate & save ─────────────────────────────────
    print(f"\n  ─── Calibrate & Save ───")
    calibrated = CalibratedClassifierCV(winner_model, cv=5, method="isotonic")
    calibrated.fit(X_train_s, y_train)

    # Feature importances
    if hasattr(winner_model, "feature_importances_"):
        imp = winner_model.feature_importances_
    elif hasattr(winner_model, "named_estimators_"):
        imp_list = [est.feature_importances_ for nm, est in winner_model.named_estimators_.items()
                    if hasattr(est, "feature_importances_")]
        imp = np.mean(imp_list, axis=0) if imp_list else np.ones(X.shape[1]) / X.shape[1]
    else:
        perm = permutation_importance(winner_model, X_test_s, y_test, n_repeats=10,
                                       random_state=42, scoring="f1")
        imp = np.maximum(perm.importances_mean, 0)
        total_p = imp.sum()
        imp = imp / total_p if total_p > 0 else np.ones(X.shape[1]) / X.shape[1]

    feature_importances = {FEATURE_NAMES[i]: float(imp[i]) for i in range(len(FEATURE_NAMES))}

    sorted_idx = np.argsort(imp)[::-1]
    print(f"  {'Rank':<6}{'Feature':<32}{'Importance':<12}")
    print(f"  {'-'*50}")
    for rank, idx in enumerate(sorted_idx[:15], 1):
        print(f"  {rank:<6}{FEATURE_NAMES[idx]:<32}{imp[idx]:.4f}")

    # Learned weights
    sk = feature_importances
    skill_imp = sk.get("avg_skill_level", 0) + sk.get("min_skill_level", 0) + sk.get("max_skill_level", 0) + sk.get("median_skill_level", 0)
    div_imp = sk.get("role_diversity", 0) + sk.get("coverage_diversity_product", 0) * 0.5
    bal_imp = sk.get("skill_variance", 0) + sk.get("gini_coefficient", 0) + sk.get("skill_iqr", 0) + sk.get("balance_penalty", 0)
    exp_imp = sk.get("avg_experience", 0) + sk.get("experience_variance", 0) + sk.get("experience_skill_interaction", 0) * 0.5
    cov_imp = sk.get("skill_coverage", 0) + sk.get("critical_role_coverage", 0) + sk.get("helpful_role_coverage", 0)
    ts_imp = sk.get("team_size", 0) + sk.get("team_strength_index", 0)
    dup_imp = sk.get("role_duplication_ratio", 0)

    total_imp = skill_imp + div_imp + bal_imp + exp_imp + cov_imp + ts_imp + dup_imp
    if total_imp == 0:
        total_imp = 1.0

    learned_weights = {
        "weight_skill": round(skill_imp / total_imp, 4),
        "weight_diversity": round(div_imp / total_imp, 4),
        "weight_balance": round(bal_imp / total_imp, 4),
        "weight_experience": round(exp_imp / total_imp, 4),
        "weight_coverage": round(cov_imp / total_imp, 4),
        "weight_team_size": round(ts_imp / total_imp, 4),
        "weight_role_dedup": round(dup_imp / total_imp, 4),
        "penalty_critical_gap": round(sk.get("has_critical_role_gap", 0) * 2, 4),
        "penalty_high_variance": round(sk.get("skill_variance", 0) * 2, 4),
        "bonus_high_coverage": round(sk.get("skill_coverage", 0) * 1.5, 4),
    }

    success_mask = y_train == 1
    X_success = X_train[success_mask]
    fi = {name: i for i, name in enumerate(FEATURE_NAMES)}
    learned_weights["threshold_min_diversity"] = round(float(np.percentile(X_success[:, fi["role_diversity"]], 25)), 2)
    learned_weights["threshold_min_avg_skill"] = round(float(np.percentile(X_success[:, fi["avg_skill_level"]], 25)), 2)
    learned_weights["threshold_max_gini"] = round(float(np.percentile(X_success[:, fi["gini_coefficient"]], 75)), 2)
    learned_weights["threshold_min_coverage"] = round(float(np.percentile(X_success[:, fi["skill_coverage"]], 25)), 2)

    print(f"\n  Learned Weights:")
    for k, v in learned_weights.items():
        print(f"    {k}: {v}")

    save_data = {
        "model": calibrated,
        "scaler": scaler,
        "learned_weights": learned_weights,
        "feature_importances": feature_importances,
        "training_metrics": {
            "version": "v2.2_round3_deterministic",
            "total_records": len(all_records),
            "train_records": len(y_train),
            "test_records": len(y_test),
            "success_count": success_count,
            "failure_count": fail_count,
            "final_model": winner_name,
            "cv_f1_mean": float(winner_cv_f1),
            "holdout_accuracy": float(test_acc),
            "holdout_f1": float(test_f1),
            "holdout_precision": float(test_prec),
            "holdout_recall": float(test_rec),
            "holdout_auc": float(test_auc),
            "holdout_logloss": float(test_ll),
            "confusion_matrix": cm.tolist(),
            "multi_seed_f1_mean": float(seed_scores.mean()),
            "multi_seed_f1_std": float(seed_scores.std()),
            "stability": stability,
            "model_comparison": results,
        },
        "is_trained": True,
        "feature_names": FEATURE_NAMES,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(save_data, f)

    report = {
        "training_date": "2026-03-02",
        "version": "v2.2_round3_deterministic",
        "training_metrics": save_data["training_metrics"],
        "learned_weights": learned_weights,
        "feature_importances": feature_importances,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"  ROUND 3 COMPLETE — v2.2")
    print(f"  Final Model:     {winner_name}")
    print(f"  Records:         {len(all_records)}")
    print(f"  Holdout Acc:     {test_acc:.3f}")
    print(f"  Holdout F1:      {test_f1:.3f}")
    print(f"  Holdout AUC:     {test_auc:.3f}")
    print(f"  Seed Stability:  {seed_scores.mean():.3f} ± {seed_scores.std():.3f}")
    print(f"  Time:            {elapsed:.1f}s")
    print(f"{'='*70}\n")

    return save_data["training_metrics"]


if __name__ == "__main__":
    run_round3()
