"""
SkillSyncAI — Training Iteration Round 2
=========================================

Builds on v2.0 results:
- Adds 150 MORE targeted edge cases (especially near decision boundary)
- Tries Stacking meta-learner 
- Adds noise injection for robustness
- Analyses misclassifications to generate targeted counter-examples
"""

import json
import os
import random
import pickle
import math
import time
import numpy as np

# Reuse everything from rigorous_trainer
from rigorous_trainer import (
    extract_features_v2, extract_batch_v2, FEATURE_NAMES,
    SKILL_POOLS, PROJECT_CONFIGS, NAMES, gini_coefficient,
    _make_member, _pick_required, DATA_PATH, MODEL_PATH, REPORT_PATH
)

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, cross_val_predict,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score, log_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

random.seed(123)
np.random.seed(123)

ALL_ROLES = list(SKILL_POOLS.keys())
PROJECT_TYPES = list(PROJECT_CONFIGS.keys())


def generate_targeted_edge_cases(n=150):
    """
    Round 2 edge cases: focus on the HARDEST patterns.
    
    The v2.0 model misclassifies cases where:
    1. High skills but wrong domain (predicts success, actually fails)
    2. Low skills but perfect fit (predicts fail, actually succeeds)
    
    We generate MORE of these patterns with controlled noise.
    """
    records = []
    counter = 500

    def make_rec(members, pt, success, grade, score, tag):
        nonlocal counter
        counter += 1
        cfg = PROJECT_CONFIGS[pt]
        req = random.sample(cfg["required_skills_pool"],
                            min(random.randint(3, 5), len(cfg["required_skills_pool"])))
        return {
            "team_id": f"EDGE2_{counter:03d}",
            "project": {
                "project_type": pt,
                "project_name": random.choice(cfg["names"]),
                "required_skills": req,
            },
            "members": members,
            "outcome": {
                "success": success,
                "grade": grade,
                "score": score,
                "completion_status": random.choice([
                    "completed_on_time", "completed_late",
                    "completed_with_issues", "incomplete"
                ]),
                "tag": tag,
            }
        }

    # ── Type 1: Skill level exactly 6 (boundary) with varying other factors ──
    for _ in range(20):
        pt = random.choice(PROJECT_TYPES)
        critical = PROJECT_CONFIGS[pt]["critical_roles"]
        helpful = PROJECT_CONFIGS[pt]["helpful_roles"]
        # All members skill=6 exactly, but good role fit → borderline success
        roles = list(critical) + random.sample(helpful, min(1, len(helpful)))
        members = [_make_member(r, 6, random.randint(1, 3)) for r in roles]
        success = random.random() < 0.6  # 60% success rate at boundary
        grade = random.choice(["B-", "C+"]) if success else "C"
        score = random.uniform(58, 68) if success else random.uniform(48, 60)
        records.append(make_rec(members, pt, success, grade, score, "boundary_skill6"))

    for _ in range(20):
        pt = random.choice(PROJECT_TYPES)
        # Skill=6, BAD role fit → borderline failure
        wrong_roles = [r for r in ALL_ROLES if r not in PROJECT_CONFIGS[pt]["critical_roles"]]
        roles = random.sample(wrong_roles, min(3, len(wrong_roles)))
        members = [_make_member(r, 6, random.randint(1, 2)) for r in roles]
        success = random.random() < 0.25  # 25% success
        grade = "C" if success else random.choice(["C-", "D+"])
        score = random.uniform(52, 60) if success else random.uniform(38, 52)
        records.append(make_rec(members, pt, success, grade, score, "boundary_skill6_bad_fit"))

    # ── Type 2: Experience compensates for lower skill ──────
    for _ in range(15):
        pt = random.choice(PROJECT_TYPES)
        critical = PROJECT_CONFIGS[pt]["critical_roles"]
        roles = list(critical) + [random.choice(PROJECT_CONFIGS[pt]["helpful_roles"])]
        # Low skill (4-5) but HIGH experience (4-5 yrs)
        members = [_make_member(r, random.randint(4, 5), random.randint(4, 5)) for r in roles]
        success = random.random() < 0.45  # somewhat helps
        grade = "C+" if success else "C-"
        score = random.uniform(55, 65) if success else random.uniform(42, 55)
        records.append(make_rec(members, pt, success, grade, score, "experienced_low_skill"))

    # ── Type 3: Mixed-level teams with good chemistry ────────
    for _ in range(15):
        pt = random.choice(PROJECT_TYPES)
        critical = PROJECT_CONFIGS[pt]["critical_roles"]
        helpful = PROJECT_CONFIGS[pt]["helpful_roles"]
        roles = list(critical) + random.sample(helpful, min(1, len(helpful)))
        # One high (8), one medium (5-6), one low (3-4) → mixed outcome
        members = [
            _make_member(roles[0], 8, 3),
            _make_member(roles[min(1, len(roles)-1)], random.randint(5, 6), 2),
        ]
        if len(roles) > 2:
            members.append(_make_member(roles[2], random.randint(3, 4), 1))
        success = random.random() < 0.55
        grade = "B-" if success else "C+"
        score = random.uniform(60, 72) if success else random.uniform(50, 62)
        records.append(make_rec(members, pt, success, grade, score, "mixed_level_chemistry"))

    # ── Type 4: Partial critical role coverage ────────────────
    for _ in range(15):
        pt = random.choice(PROJECT_TYPES)
        critical = list(PROJECT_CONFIGS[pt]["critical_roles"])
        if len(critical) >= 2:
            # Have 1 of 2 critical roles
            roles = [critical[0]] + [random.choice(PROJECT_CONFIGS[pt]["helpful_roles"])]
        else:
            roles = [critical[0], random.choice(ALL_ROLES)]
        members = [_make_member(r, random.randint(6, 8), random.randint(1, 3)) for r in roles]
        success = random.random() < 0.5
        grade = "B-" if success else "C+"
        score = random.uniform(58, 70) if success else random.uniform(48, 60)
        records.append(make_rec(members, pt, success, grade, score, "partial_critical"))

    # ── Type 5: Team size extremes ───────────────────────────
    for _ in range(10):
        # Very small (2 members) — can go either way
        pt = random.choice(PROJECT_TYPES)
        critical = list(PROJECT_CONFIGS[pt]["critical_roles"])
        roles = critical[:min(2, len(critical))]
        while len(roles) < 2:
            roles.append(random.choice(ALL_ROLES))
        members = [_make_member(r, random.randint(6, 8), random.randint(1, 3)) for r in roles]
        success = random.random() < 0.5
        grade = "B" if success else "C"
        score = random.uniform(62, 72) if success else random.uniform(45, 58)
        records.append(make_rec(members, pt, success, grade, score, "tiny_team"))

    for _ in range(10):
        # Large team (5-6) with diminishing returns
        pt = random.choice(PROJECT_TYPES)
        critical = list(PROJECT_CONFIGS[pt]["critical_roles"])
        helpful = list(PROJECT_CONFIGS[pt]["helpful_roles"])
        roles = critical + helpful[:2] + [random.choice(ALL_ROLES)] * 2
        roles = roles[:6]
        members = [_make_member(r, random.randint(5, 7), random.randint(1, 3)) for r in roles]
        success = random.random() < 0.55
        grade = "B-" if success else "C"
        score = random.uniform(60, 70) if success else random.uniform(48, 58)
        records.append(make_rec(members, pt, success, grade, score, "large_team"))

    # ── Type 6: Perfect coverage but low individual skill ────
    for _ in range(15):
        pt = random.choice(PROJECT_TYPES)
        critical = list(PROJECT_CONFIGS[pt]["critical_roles"])
        helpful = list(PROJECT_CONFIGS[pt]["helpful_roles"])
        roles = critical + helpful[:min(2, len(helpful))]
        members = [_make_member(r, random.randint(4, 5), random.randint(0, 2)) for r in roles]
        success = random.random() < 0.35  # hard to succeed with all-low skill even with coverage
        grade = "C+" if success else random.choice(["C-", "D+"])
        score = random.uniform(52, 62) if success else random.uniform(35, 52)
        records.append(make_rec(members, pt, success, grade, score, "covered_but_weak"))

    # ── Type 7: Noisy versions of clear success/failure ──────
    for _ in range(15):
        # Clear success pattern but with 1 problem member
        pt = random.choice(PROJECT_TYPES)
        critical = list(PROJECT_CONFIGS[pt]["critical_roles"])
        roles = critical + [random.choice(PROJECT_CONFIGS[pt]["helpful_roles"])]
        members = [_make_member(r, random.randint(7, 9), random.randint(2, 4)) for r in roles]
        # Replace one member with a weak one
        weak_idx = random.randint(0, len(members) - 1)
        members[weak_idx]["skill_level"] = random.randint(2, 3)
        members[weak_idx]["experience_years"] = 0
        success = random.random() < 0.6  # usually still succeeds
        grade = "B-" if success else "C+"
        score = random.uniform(60, 72) if success else random.uniform(50, 62)
        records.append(make_rec(members, pt, success, grade, score, "mostly_good_one_weak"))

    for _ in range(15):
        # Clear failure pattern but 1 strong member
        pt = random.choice(PROJECT_TYPES)
        roles = [random.choice(ALL_ROLES)] * 3
        members = [_make_member(r, random.randint(3, 4), random.randint(0, 1)) for r in roles]
        star_idx = random.randint(0, len(members) - 1)
        members[star_idx]["skill_level"] = random.randint(8, 9)
        members[star_idx]["experience_years"] = 4
        success = random.random() < 0.3  # usually still fails
        grade = "C" if success else random.choice(["D+", "D"])
        score = random.uniform(50, 58) if success else random.uniform(32, 48)
        records.append(make_rec(members, pt, success, grade, score, "mostly_bad_one_star"))

    random.shuffle(records)
    return records[:n]


def train_stacking(X_train, y_train, X_test, y_test):
    """
    Level 2: Stacking meta-learner using base model predictions as features.
    """
    print(f"\n  ─── Stacking Meta-Learner ───")

    base_estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=2,
            class_weight="balanced", random_state=42)),
        ("gb", GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)),
        ("svm", SVC(
            kernel="rbf", C=1.0, probability=True,
            class_weight="balanced", random_state=42)),
        ("et", ExtraTreesClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=2,
            class_weight="balanced", random_state=42)),
    ]

    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=10.0, max_iter=1000),
        cv=5,
        stack_method="predict_proba",
        passthrough=True,  # also pass original features to meta-learner
    )

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(stacker, X_train, y_train, cv=cv, scoring="f1")
    print(f"  Stacking CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    stacker.fit(X_train, y_train)
    y_pred = stacker.predict(X_test)
    y_prob = stacker.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)
    test_ll = log_loss(y_test, y_prob)

    print(f"  Stacking Holdout Acc:  {test_acc:.3f}")
    print(f"  Stacking Holdout F1:   {test_f1:.3f}")
    print(f"  Stacking Holdout AUC:  {test_auc:.3f}")

    return stacker, cv_scores.mean(), test_acc, test_f1, test_auc


def run_round2():
    start = time.time()

    print("\n" + "=" * 70)
    print("  TRAINING ROUND 2: TARGETED IMPROVEMENT")
    print("=" * 70)

    # Load current data
    with open(DATA_PATH, "r") as f:
        current = json.load(f)
    existing = current.get("team_records", [])
    # Keep only non-EDGE2 records (keep original + EDGE_ round 1)
    existing = [r for r in existing if not r.get("team_id", "").startswith("EDGE2_")]
    print(f"  Base records: {len(existing)}")

    # Generate round 2 edge cases
    edge2 = generate_targeted_edge_cases(150)
    from collections import Counter
    tags = Counter(r["outcome"].get("tag", "") for r in edge2)
    for tag, count in sorted(tags.items(), key=lambda x: -x[1]):
        print(f"  {tag:35s} {count:3d}")

    all_records = existing + edge2
    success_count = sum(1 for r in all_records if r["outcome"]["success"])
    fail_count = len(all_records) - success_count
    print(f"\n  Total dataset:  {len(all_records)} records")
    print(f"  Success / Fail: {success_count} / {fail_count}")

    # Save
    merged = {
        "metadata": {
            "total_records": len(all_records),
            "generated_date": "2026-03-02",
            "purpose": "feedback_loop_training_v2.1_round2",
            "success_count": success_count,
            "failure_count": fail_count,
            "includes_edge_cases": True,
            "round2_added": len(edge2),
        },
        "team_records": all_records,
    }
    with open(DATA_PATH, "w") as f:
        json.dump(merged, f, indent=2)

    # Extract features
    X, y = extract_batch_v2(all_records)
    print(f"  Features: {X.shape[1]}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"  Train / Test: {len(y_train)} / {len(y_test)}")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # ── Compare models on expanded data ──────────────────
    print(f"\n  ─── Single Model Comparison ───")
    models = {
        "RF(300,d8)": RandomForestClassifier(n_estimators=300, max_depth=8,
                                              min_samples_leaf=2, class_weight="balanced", random_state=42),
        "GB(200,d4)": GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                  learning_rate=0.1, random_state=42),
        "SVM_RBF(C1)": SVC(kernel="rbf", C=1.0, probability=True,
                           class_weight="balanced", random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, max_depth=8,
                                            min_samples_leaf=2, class_weight="balanced", random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=150, learning_rate=0.1,
                                        random_state=42),
    }

    best_single_f1 = 0
    best_single_name = None
    for name, model in models.items():
        scores_f1 = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="f1")
        scores_acc = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")
        print(f"  {name:20s}  Acc={scores_acc.mean():.3f}  F1={scores_f1.mean():.3f}  (±{scores_f1.std():.3f})")
        if scores_f1.mean() > best_single_f1:
            best_single_f1 = scores_f1.mean()
            best_single_name = name

    print(f"\n  Best single model: {best_single_name} (F1={best_single_f1:.3f})")

    # ── Stacking ─────────────────────────────────────────
    stacker, stack_cv_f1, stack_acc, stack_f1, stack_auc = train_stacking(
        X_train_s, y_train, X_test_s, y_test
    )

    # ── Soft Voting ──────────────────────────────────────
    print(f"\n  ─── Soft Voting Ensemble ───")
    voter = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=8,
                                          min_samples_leaf=2, class_weight="balanced", random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                              learning_rate=0.1, random_state=42)),
            ("svm", SVC(kernel="rbf", C=1.0, probability=True,
                        class_weight="balanced", random_state=42)),
            ("et", ExtraTreesClassifier(n_estimators=200, max_depth=8,
                                         class_weight="balanced", random_state=42)),
        ],
        voting="soft",
    )
    voter_f1 = cross_val_score(voter, X_train_s, y_train, cv=cv, scoring="f1")
    print(f"  Voting CV F1: {voter_f1.mean():.3f} ± {voter_f1.std():.3f}")

    # ── Pick winner ──────────────────────────────────────
    candidates = {
        best_single_name: (models[best_single_name], best_single_f1),
        "Stacking": (stacker, stack_cv_f1),
        "Voting": (voter, voter_f1.mean()),
    }
    winner_name = max(candidates, key=lambda k: candidates[k][1])
    winner_model, winner_f1 = candidates[winner_name]
    print(f"\n  >>> WINNER: {winner_name} (CV F1={winner_f1:.3f})")

    # ── Final evaluation on holdout ──────────────────────
    print(f"\n  ─── Final Holdout Evaluation ({winner_name}) ───")

    if winner_name != "Stacking":
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
    print(f"  F1 across 10 seeds: {seed_scores.mean():.3f} ± {seed_scores.std():.3f}")

    # ── Calibrate and save ───────────────────────────────
    print(f"\n  ─── Calibration & Save ───")
    calibrated = CalibratedClassifierCV(winner_model, cv=5, method="isotonic")
    calibrated.fit(X_train_s, y_train)

    # Feature importances (try multiple approaches)
    feature_importances = {}
    if hasattr(winner_model, "feature_importances_"):
        imp = winner_model.feature_importances_
    elif hasattr(winner_model, "named_estimators_"):
        imp_list = []
        for nm, est in winner_model.named_estimators_.items():
            if hasattr(est, "feature_importances_"):
                imp_list.append(est.feature_importances_)
        imp = np.mean(imp_list, axis=0) if imp_list else np.ones(X.shape[1]) / X.shape[1]
    else:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(winner_model, X_test_s, y_test, n_repeats=10, random_state=42, scoring="f1")
        imp = np.maximum(perm.importances_mean, 0)
        total_p = imp.sum()
        imp = imp / total_p if total_p > 0 else np.ones(X.shape[1]) / X.shape[1]

    for i, name in enumerate(FEATURE_NAMES):
        feature_importances[name] = float(imp[i])

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

    # Thresholds from successful training samples
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
            "version": "v2.1_round2",
            "total_records": len(all_records),
            "train_records": len(y_train),
            "test_records": len(y_test),
            "success_count": success_count,
            "failure_count": fail_count,
            "final_model": winner_name,
            "cv_f1_mean": float(winner_f1),
            "holdout_accuracy": float(test_acc),
            "holdout_f1": float(test_f1),
            "holdout_precision": float(test_prec),
            "holdout_recall": float(test_rec),
            "holdout_auc": float(test_auc),
            "holdout_logloss": float(test_ll),
            "confusion_matrix": cm.tolist(),
            "multi_seed_f1_mean": float(seed_scores.mean()),
            "multi_seed_f1_std": float(seed_scores.std()),
        },
        "is_trained": True,
        "feature_names": FEATURE_NAMES,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(save_data, f)

    report = {
        "training_date": "2026-03-02",
        "version": "v2.1_round2",
        "training_metrics": save_data["training_metrics"],
        "learned_weights": learned_weights,
        "feature_importances": feature_importances,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - start

    print(f"\n{'='*70}")
    print(f"  ROUND 2 TRAINING COMPLETE — v2.1")
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
    run_round2()
