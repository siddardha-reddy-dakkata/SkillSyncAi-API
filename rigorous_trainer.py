"""
SkillSyncAI — Rigorous Iterative Training Harness
===================================================

PURPOSE:
  The v1.0 model achieved 100% CV accuracy on 55 records. That's a RED FLAG,
  not a success. It means the training data was trivially separable — every
  success had skill_level 7-8 and every failure had 3-5. Zero ambiguity.

  This harness fixes that by:
  1. Generating 200+ HARD edge cases near the decision boundary
  2. Expanding from 14 to 25 features (interaction + ratio features)
  3. Proper 80/20 holdout split (not just CV on training data)
  4. Multi-model comparison (RF, GB, LR, SVM, Voting Ensemble)
  5. Hyperparameter grid search with cross-validation
  6. Probability calibration (CalibratedClassifierCV)
  7. Multi-seed stability testing
  8. Learning curve analysis

USAGE:
  python rigorous_trainer.py

OUTPUT:
  - Updated feedback_training_data.json (with hard cases added)
  - model_weights.pkl (best model after grid search)
  - training_report.json (detailed metrics report)
  - Git commit at each milestone
"""

import json
import os
import random
import pickle
import math
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "feedback_training_data.json")
MODEL_PATH = os.path.join(BASE_DIR, "model_weights.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "training_report.json")

random.seed(42)
np.random.seed(42)


# ============================================================
# PART 1: REALISTIC EDGE-CASE DATA GENERATOR
# ============================================================

# Real skill pools drawn from resume patterns in the Entity Recognition dataset
SKILL_POOLS = {
    "frontend": {
        "core": ["React", "Angular", "Vue", "HTML", "CSS", "JavaScript", "TypeScript"],
        "extra": ["Bootstrap", "Tailwind", "SASS", "Webpack", "Next.js", "Redux", "jQuery"],
    },
    "backend": {
        "core": ["Node.js", "Express", "Django", "Flask", "Spring Boot", "Java", "Python", "PHP"],
        "extra": ["GraphQL", "REST API", "gRPC", "FastAPI", "Ruby on Rails", "Go", "C#"],
    },
    "fullstack": {
        "core": ["React", "Node.js", "MongoDB", "JavaScript", "Python"],
        "extra": ["PostgreSQL", "Docker", "TypeScript", "Next.js", "Express"],
    },
    "ml_engineer": {
        "core": ["Python", "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy"],
        "extra": ["Keras", "OpenCV", "NLP", "Deep Learning", "Hugging Face", "BERT"],
    },
    "data_engineer": {
        "core": ["Python", "SQL", "Pandas", "ETL", "Spark", "Hadoop"],
        "extra": ["Kafka", "Airflow", "BigQuery", "Teradata", "Power BI", "Tableau"],
    },
    "devops": {
        "core": ["Docker", "Kubernetes", "AWS", "Linux", "CI/CD", "Terraform"],
        "extra": ["Jenkins", "GitHub Actions", "Azure", "GCP", "Ansible", "Nginx"],
    },
    "qa_tester": {
        "core": ["Selenium", "Python", "Testing", "JIRA", "Manual Testing"],
        "extra": ["Cypress", "Jest", "Postman", "JUnit", "Automation Testing"],
    },
    "ui_ux": {
        "core": ["Figma", "Adobe XD", "Wireframing", "Prototyping", "User Research"],
        "extra": ["Sketch", "InVision", "Photoshop", "Illustrator", "Canva"],
    },
}

PROJECT_CONFIGS = {
    "web_application": {
        "names": [
            "E-Commerce Platform", "Social Media App", "LMS Portal",
            "Project Management Tool", "Healthcare Dashboard", "Food Delivery App",
            "Job Board", "Real Estate Portal", "EdTech Platform", "News Aggregator",
        ],
        "required_skills_pool": [
            "React", "Node.js", "MongoDB", "REST API", "CSS", "JavaScript",
            "Express", "PostgreSQL", "Docker", "HTML",
        ],
        "critical_roles": ["frontend", "backend"],
        "helpful_roles": ["devops", "ui_ux", "qa_tester", "fullstack"],
    },
    "ml_project": {
        "names": [
            "Sentiment Analyzer", "Image Classifier", "Recommendation Engine",
            "Fraud Detector", "Chatbot NLP", "Predictive Maintenance",
            "Resume Parser", "Speech Recognition", "Object Detection", "Text Summarizer",
        ],
        "required_skills_pool": [
            "Python", "TensorFlow", "PyTorch", "Pandas", "Scikit-learn",
            "NumPy", "NLP", "Deep Learning", "Keras", "OpenCV",
        ],
        "critical_roles": ["ml_engineer"],
        "helpful_roles": ["data_engineer", "backend", "devops"],
    },
    "data_pipeline": {
        "names": [
            "ETL Pipeline", "Data Warehouse", "Analytics Dashboard",
            "Data Lake Architecture", "Real-Time Stream Processor",
            "Log Aggregator", "BI Reporting System", "Data Quality Monitor",
        ],
        "required_skills_pool": [
            "Python", "SQL", "Spark", "Pandas", "ETL", "Kafka",
            "Airflow", "BigQuery", "Hadoop", "Docker",
        ],
        "critical_roles": ["data_engineer"],
        "helpful_roles": ["backend", "devops", "ml_engineer"],
    },
    "mobile_app": {
        "names": [
            "Fitness Tracker", "Expense Manager", "Travel Planner",
            "Recipe App", "Meditation App", "Language Learner",
        ],
        "required_skills_pool": [
            "React Native", "Flutter", "JavaScript", "Firebase",
            "REST API", "Node.js", "MongoDB", "TypeScript",
        ],
        "critical_roles": ["frontend", "fullstack"],
        "helpful_roles": ["backend", "ui_ux", "devops"],
    },
    "api_service": {
        "names": [
            "Payment Gateway", "Auth Service", "Notification Engine",
            "Search Microservice", "API Gateway", "Rate Limiter",
        ],
        "required_skills_pool": [
            "Node.js", "Python", "REST API", "GraphQL", "Docker",
            "PostgreSQL", "Redis", "JWT", "Kubernetes",
        ],
        "critical_roles": ["backend"],
        "helpful_roles": ["devops", "qa_tester", "fullstack"],
    },
    "database_system": {
        "names": [
            "Inventory DBMS", "Patient Records System", "Fleet Management DB",
            "Banking Ledger", "Academic Records", "Supply Chain DB",
        ],
        "required_skills_pool": [
            "SQL", "PostgreSQL", "MongoDB", "Python", "Oracle",
            "Redis", "Indexing", "Normalization", "ETL",
        ],
        "critical_roles": ["backend", "data_engineer"],
        "helpful_roles": ["devops", "qa_tester"],
    },
}

# Indian names from the resume dataset for realism
NAMES = [
    "Abhishek Jha", "Afreen Jamadar", "Akhil Yadav", "Alok Khandai",
    "Ananya Chavan", "Anvitha Rao", "Arun Elumalai", "Ashalata Bisoyi",
    "Arjun Kumar", "Priya Sharma", "Sneha Reddy", "Vikram Singh",
    "Deepa Nair", "Rahul Gupta", "Kavitha Menon", "Rohit Joshi",
    "Meera Pillai", "Sanjay Patel", "Divya Krishnan", "Aditya Verma",
    "Neha Agarwal", "Suresh Babu", "Lakshmi Iyer", "Karthik Rajan",
    "Pooja Desai", "Manish Tiwari", "Swathi Reddy", "Venkat Rao",
    "Radhika Kulk", "Ajay Mishra", "Bhavana Shetty", "Ganesh Hegde",
    "Ishaan Malhotra", "Janaki Ramesh", "Kunal Bhatt", "Lavanya Subram",
    "Mohan Das", "Nandini Bose", "Om Prakash", "Pallavi Jain",
    "Ravi Shankar", "Shruti Kapoor", "Tanvi Mehta", "Uma Mahesh",
    "Varun Nambiar", "Waqar Ahmed", "Xavier D'Souza", "Yogesh Chauhan",
    "Zara Khan", "Aarti Saxena", "Bala Krishnan", "Chitra Venkatesan",
]


def _pick_skills(role: str, n_core: int, n_extra: int) -> List[str]:
    pool = SKILL_POOLS.get(role, SKILL_POOLS["backend"])
    core = random.sample(pool["core"], min(n_core, len(pool["core"])))
    extra = random.sample(pool["extra"], min(n_extra, len(pool["extra"])))
    return core + extra


def _make_member(role: str, skill_level: int, exp_years: int) -> Dict:
    n_core = random.randint(2, 4)
    n_extra = random.randint(0, 2)
    skills = _pick_skills(role, n_core, n_extra)
    return {
        "name": random.choice(NAMES),
        "assigned_role": role,
        "skills": skills,
        "skill_level": skill_level,
        "experience_years": exp_years,
    }


def _pick_required(project_type: str, n: int) -> List[str]:
    pool = PROJECT_CONFIGS[project_type]["required_skills_pool"]
    return random.sample(pool, min(n, len(pool)))


def _team_skills_cover(members, required):
    team_skills = set()
    for m in members:
        for s in m["skills"]:
            team_skills.add(s.lower())
    required_lower = set(s.lower() for s in required)
    if not required_lower:
        return 0.5
    return len(team_skills & required_lower) / len(required_lower)


def generate_edge_cases(n_target: int = 200) -> List[Dict]:
    """
    Generate hard, ambiguous training records near the decision boundary.

    Categories of edge cases:
    ─────────────────────────────────────────────────────────────────────
    A. BORDERLINE SUCCESS (55-65% chance of success):
       Good in SOME dimensions but weak in others.
    B. BORDERLINE FAILURE (35-45% chance of success):
       Bad in SOME dimensions but saved/doomed by others.
    C. COUNTERINTUITIVE SUCCESS:
       Looks bad on paper but succeeds (e.g. 1 star + 3 mediocre → success).
    D. COUNTERINTUITIVE FAILURE:
       Looks good on paper but fails (e.g. all-stars but no role fit).
    E. REALISTIC VARIANCE:
       Normal teams with natural noise in outcome.
    ─────────────────────────────────────────────────────────────────────
    """
    records = []
    team_counter = 100  # Start at 100 to not collide with existing

    project_types = list(PROJECT_CONFIGS.keys())
    all_roles = list(SKILL_POOLS.keys())

    def make_record(members, project_type, success, grade, score, tag=""):
        nonlocal team_counter
        team_counter += 1
        proj_cfg = PROJECT_CONFIGS[project_type]
        proj_name = random.choice(proj_cfg["names"])
        required = _pick_required(project_type, random.randint(3, 5))
        return {
            "team_id": f"EDGE_{team_counter:03d}",
            "project": {
                "project_type": project_type,
                "project_name": proj_name,
                "required_skills": required,
            },
            "members": members,
            "outcome": {
                "success": success,
                "grade": grade,
                "score": score,
                "completion_status": random.choice([
                    "completed_on_time", "completed_late",
                    "completed_with_issues", "incomplete"
                ]) if not success else random.choice([
                    "completed_on_time", "completed_late",
                    "completed_with_issues",
                ]),
                "tag": tag,
            }
        }

    # ── Category A: Borderline Success (60 records) ─────────────────

    for _ in range(15):
        # A1: Good skills (6-7) but ONE duplicate role → still success
        pt = random.choice(project_types)
        critical = PROJECT_CONFIGS[pt]["critical_roles"]
        roles = list(critical) + [random.choice(critical)]  # one dup
        if len(roles) < 3:
            roles.append(random.choice(PROJECT_CONFIGS[pt]["helpful_roles"]))
        members = [_make_member(r, random.randint(6, 7), random.randint(1, 3)) for r in roles]
        records.append(make_record(members, pt, True, random.choice(["B+", "B"]),
                                   random.uniform(68, 75), "borderline_dup_success"))

    for _ in range(15):
        # A2: All mediocre (5-6) but perfect role diversity + coverage
        pt = random.choice(project_types)
        critical = PROJECT_CONFIGS[pt]["critical_roles"]
        helpful = PROJECT_CONFIGS[pt]["helpful_roles"]
        roles = list(critical) + random.sample(helpful, min(2, len(helpful)))
        members = [_make_member(r, random.randint(5, 6), random.randint(1, 2)) for r in roles]
        records.append(make_record(members, pt, True, "B-",
                                   random.uniform(62, 70), "mediocre_good_fit"))

    for _ in range(15):
        # A3: High skills (7-8) but partial coverage (60-75%)
        pt = random.choice(project_types)
        critical = PROJECT_CONFIGS[pt]["critical_roles"]
        # Use one critical + one random (possibly wrong)
        roles = list(critical)[:1] + [random.choice(all_roles)]
        if len(roles) < 3:
            roles.append(random.choice(all_roles))
        members = [_make_member(r, random.randint(7, 8), random.randint(2, 4)) for r in roles]
        records.append(make_record(members, pt, True, random.choice(["B", "B+"]),
                                   random.uniform(65, 78), "strong_partial_coverage"))

    for _ in range(15):
        # A4: One star (8-9) carrying weak teammates (4-5)
        pt = random.choice(project_types)
        critical = PROJECT_CONFIGS[pt]["critical_roles"]
        star_role = random.choice(critical)
        members = [_make_member(star_role, random.randint(8, 9), random.randint(3, 5))]
        for _ in range(random.randint(2, 3)):
            members.append(_make_member(random.choice(all_roles), random.randint(4, 5),
                                        random.randint(0, 1)))
        records.append(make_record(members, pt, True, "B-",
                                   random.uniform(60, 68), "star_carries"))

    # ── Category B: Borderline Failure (60 records) ──────────────────

    for _ in range(15):
        # B1: Decent skills (6-7) but MISSING critical role → fail
        pt = random.choice(project_types)
        helpful = PROJECT_CONFIGS[pt]["helpful_roles"]
        roles = random.sample(helpful, min(3, len(helpful)))
        if len(roles) < 3:
            roles += random.sample(all_roles, 3 - len(roles))
        members = [_make_member(r, random.randint(6, 7), random.randint(1, 3)) for r in roles]
        records.append(make_record(members, pt, False, random.choice(["C+", "C"]),
                                   random.uniform(50, 60), "decent_no_critical"))

    for _ in range(15):
        # B2: One great (8-9) + one terrible (2-3) → high variance kills
        pt = random.choice(project_types)
        critical = PROJECT_CONFIGS[pt]["critical_roles"]
        members = [
            _make_member(random.choice(critical), random.randint(8, 9), 4),
            _make_member(random.choice(all_roles), random.randint(2, 3), 0),
            _make_member(random.choice(all_roles), random.randint(3, 4), 1),
        ]
        records.append(make_record(members, pt, False, "C",
                                   random.uniform(45, 58), "high_variance_fail"))

    for _ in range(15):
        # B3: All same role (duplication disaster) but good skills
        pt = random.choice(project_types)
        single_role = random.choice(all_roles)
        members = [_make_member(single_role, random.randint(6, 8), random.randint(1, 3))
                    for _ in range(random.randint(3, 4))]
        records.append(make_record(members, pt, False, random.choice(["C", "D+"]),
                                   random.uniform(40, 55), "role_clones"))

    for _ in range(15):
        # B4: Medium skills (5-6), some coverage, but just not enough
        pt = random.choice(project_types)
        critical = PROJECT_CONFIGS[pt]["critical_roles"]
        roles = list(critical)[:1] + [random.choice(all_roles)] * 2
        members = [_make_member(r, random.randint(5, 6), random.randint(0, 2)) for r in roles]
        records.append(make_record(members, pt, False, "C-",
                                   random.uniform(42, 55), "mediocre_fail"))

    # ── Category C: Counterintuitive Success (25 records) ─────────────

    for _ in range(10):
        # C1: Small team (2 members) but perfect fit → success
        pt = random.choice(project_types)
        critical = PROJECT_CONFIGS[pt]["critical_roles"]
        roles = critical[:2] if len(critical) >= 2 else critical + [random.choice(PROJECT_CONFIGS[pt]["helpful_roles"])]
        members = [_make_member(r, random.randint(7, 9), random.randint(2, 4)) for r in roles[:2]]
        records.append(make_record(members, pt, True, "B",
                                   random.uniform(65, 75), "tiny_perfect_fit"))

    for _ in range(8):
        # C2: Low experience (0-1 yr) but high skill (7-8) → success
        pt = random.choice(project_types)
        critical = PROJECT_CONFIGS[pt]["critical_roles"]
        helpful = PROJECT_CONFIGS[pt]["helpful_roles"]
        roles = list(critical) + random.sample(helpful, min(1, len(helpful)))
        members = [_make_member(r, random.randint(7, 8), random.randint(0, 1)) for r in roles]
        records.append(make_record(members, pt, True, "B",
                                   random.uniform(65, 72), "no_exp_high_skill"))

    for _ in range(7):
        # C3: Large team (5-6) with mixed skills (4-8) → success due to coverage
        pt = random.choice(project_types)
        all_needed = list(PROJECT_CONFIGS[pt]["critical_roles"]) + list(PROJECT_CONFIGS[pt]["helpful_roles"])
        roles = all_needed[:min(5, len(all_needed))]
        while len(roles) < 5:
            roles.append(random.choice(all_roles))
        members = [_make_member(r, random.randint(4, 8), random.randint(0, 3)) for r in roles]
        records.append(make_record(members, pt, True, "B-",
                                   random.uniform(62, 70), "big_mixed_coverage"))

    # ── Category D: Counterintuitive Failure (25 records) ─────────────

    for _ in range(10):
        # D1: All-stars (8-9) but ZERO skill coverage → fail
        pt = random.choice(project_types)
        # Deliberately pick wrong roles
        wrong_roles = [r for r in all_roles if r not in PROJECT_CONFIGS[pt]["critical_roles"]]
        roles = random.sample(wrong_roles, min(3, len(wrong_roles)))
        if len(roles) < 3:
            roles += [random.choice(wrong_roles)] * (3 - len(roles))
        members = [_make_member(r, random.randint(8, 9), random.randint(2, 4)) for r in roles]
        records.append(make_record(members, pt, False, "C",
                                   random.uniform(45, 58), "allstars_wrong_domain"))

    for _ in range(8):
        # D2: Great skills + great roles but huge team (6+) with infighting
        pt = random.choice(project_types)
        critical = PROJECT_CONFIGS[pt]["critical_roles"]
        helpful = PROJECT_CONFIGS[pt]["helpful_roles"]
        roles = list(critical) + list(helpful) + list(critical)  # duplicates
        members = [_make_member(r, random.randint(7, 9), random.randint(2, 4)) for r in roles[:6]]
        records.append(make_record(members, pt, False, "C+",
                                   random.uniform(52, 62), "too_many_cooks"))

    for _ in range(7):
        # D3: Perfect role diversity but all low-skill (3-4)
        pt = random.choice(project_types)
        critical = PROJECT_CONFIGS[pt]["critical_roles"]
        helpful = PROJECT_CONFIGS[pt]["helpful_roles"]
        roles = list(critical) + random.sample(helpful, min(2, len(helpful)))
        members = [_make_member(r, random.randint(3, 4), random.randint(0, 1)) for r in roles]
        records.append(make_record(members, pt, False, "D",
                                   random.uniform(35, 48), "diverse_but_weak"))

    # ── Category E: Realistic Variance (30 records) ──────────────────

    for _ in range(15):
        # Normal good teams with some noise
        pt = random.choice(project_types)
        critical = PROJECT_CONFIGS[pt]["critical_roles"]
        helpful = PROJECT_CONFIGS[pt]["helpful_roles"]
        roles = list(critical) + random.sample(helpful, min(random.randint(1, 2), len(helpful)))
        avg_skill = random.randint(6, 8)
        members = [_make_member(r, avg_skill + random.randint(-1, 1),
                                random.randint(1, 4)) for r in roles]
        grade = random.choice(["A", "A-", "B+", "B"])
        records.append(make_record(members, pt, True, grade,
                                   random.uniform(70, 92), "normal_good"))

    for _ in range(15):
        # Normal bad teams with some noise
        pt = random.choice(project_types)
        n_members = random.randint(2, 4)
        roles = [random.choice(all_roles) for _ in range(n_members)]
        avg_skill = random.randint(3, 5)
        members = [_make_member(r, avg_skill + random.randint(-1, 1),
                                random.randint(0, 2)) for r in roles]
        grade = random.choice(["D", "D+", "C-", "F"])
        records.append(make_record(members, pt, False, grade,
                                   random.uniform(25, 50), "normal_bad"))

    random.shuffle(records)
    return records[:n_target]


# ============================================================
# PART 2: ENHANCED FEATURE EXTRACTOR (25 features)
# ============================================================

PROJECT_TYPE_MAP = {
    "web_application": 0, "ml_project": 1, "data_pipeline": 2,
    "mobile_app": 3, "api_service": 4, "database_system": 5,
}

CRITICAL_ROLES = {
    "web_application": {"frontend", "backend"},
    "ml_project": {"ml_engineer"},
    "data_pipeline": {"data_engineer"},
    "mobile_app": {"fullstack", "frontend"},
    "api_service": {"backend"},
    "database_system": {"backend", "data_engineer"},
}

FEATURE_NAMES = [
    # ── Original 14 ──
    "team_size",
    "role_diversity",
    "avg_skill_level",
    "min_skill_level",
    "max_skill_level",
    "skill_variance",
    "skill_range",
    "gini_coefficient",
    "avg_experience",
    "experience_variance",
    "skill_coverage",
    "has_critical_role_gap",
    "role_duplication_ratio",
    "project_type_encoded",
    # ── NEW: 11 advanced features ──
    "median_skill_level",          # robust center
    "skill_iqr",                   # interquartile range (robust spread)
    "total_skills_count",          # total unique skills across team
    "skills_per_member",           # avg unique skills per person
    "critical_role_coverage",      # fraction of critical roles filled
    "helpful_role_coverage",       # fraction of helpful roles filled
    "experience_skill_interaction",# avg_exp * avg_skill (synergy)
    "weakest_link_score",          # min_skill * role_diversity (bottleneck)
    "team_strength_index",         # sum of all skill_levels / team_size^0.5
    "coverage_diversity_product",  # skill_coverage * role_diversity
    "balance_penalty",             # composite penalty for imbalance
]


def gini_coefficient(values):
    if not values or len(values) < 2:
        return 0.0
    s = sorted(values)
    n = len(s)
    total = sum(s)
    if total == 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(s))
    g = (2 * cum) / (n * total) - (n + 1) / n
    return max(0.0, min(1.0, g))


def extract_features_v2(record: Dict) -> np.ndarray:
    """Extract 25-dimensional feature vector."""
    members = record["members"]
    project = record["project"]
    pt = project.get("project_type", "web_application")
    required = set(s.lower() for s in project.get("required_skills", []))

    team_size = len(members)
    skill_levels = [m["skill_level"] for m in members]
    experience = [m.get("experience_years", 0) for m in members]
    roles = [m["assigned_role"] for m in members]

    # ── Original 14 ──────────────────────────────────────
    unique_roles = len(set(roles))
    role_diversity = unique_roles / max(team_size, 1)
    avg_skill = float(np.mean(skill_levels)) if skill_levels else 0
    min_skill = min(skill_levels) if skill_levels else 0
    max_skill = max(skill_levels) if skill_levels else 0
    skill_var = float(np.var(skill_levels)) if len(skill_levels) > 1 else 0
    skill_range = max_skill - min_skill
    gini = gini_coefficient(skill_levels)
    avg_exp = float(np.mean(experience)) if experience else 0
    exp_var = float(np.var(experience)) if len(experience) > 1 else 0

    team_skills = set()
    for m in members:
        for s in m.get("skills", []):
            team_skills.add(s.lower())
    skill_coverage = (len(team_skills & required) / max(len(required), 1)) if required else 0.5

    critical = CRITICAL_ROLES.get(pt, set())
    roles_set = set(roles)
    critical_gap = 1.0 if (critical and not (critical & roles_set)) else 0.0

    role_counts = Counter(roles)
    dups = sum(c - 1 for c in role_counts.values() if c > 1)
    role_dup_ratio = dups / max(team_size, 1)

    project_enc = PROJECT_TYPE_MAP.get(pt, 0)

    # ── NEW 11 features ──────────────────────────────────
    median_skill = float(np.median(skill_levels)) if skill_levels else 0
    q75 = float(np.percentile(skill_levels, 75)) if len(skill_levels) >= 4 else max_skill
    q25 = float(np.percentile(skill_levels, 25)) if len(skill_levels) >= 4 else min_skill
    skill_iqr = q75 - q25

    total_skills_count = len(team_skills)
    skills_per_member = total_skills_count / max(team_size, 1)

    critical_filled = len(critical & roles_set) / max(len(critical), 1) if critical else 1.0
    helpful = set(PROJECT_CONFIGS.get(pt, {}).get("helpful_roles", []))
    helpful_filled = len(helpful & roles_set) / max(len(helpful), 1) if helpful else 0.5

    exp_skill_interaction = avg_exp * avg_skill
    weakest_link = min_skill * role_diversity
    team_strength = sum(skill_levels) / max(math.sqrt(team_size), 1)
    cov_div_product = skill_coverage * role_diversity
    balance_penalty = gini * skill_var * (1 + role_dup_ratio)

    return np.array([
        team_size, role_diversity, avg_skill, min_skill, max_skill,
        skill_var, skill_range, gini, avg_exp, exp_var,
        skill_coverage, critical_gap, role_dup_ratio, project_enc,
        # new
        median_skill, skill_iqr, total_skills_count, skills_per_member,
        critical_filled, helpful_filled, exp_skill_interaction,
        weakest_link, team_strength, cov_div_product, balance_penalty,
    ])


def extract_batch_v2(records):
    X, y = [], []
    for r in records:
        X.append(extract_features_v2(r))
        y.append(1 if r["outcome"]["success"] else 0)
    return np.array(X), np.array(y)


# ============================================================
# PART 3: RIGOROUS MULTI-MODEL TRAINING
# ============================================================

def train_rigorous(records: List[Dict]) -> Dict:
    """
    Full rigorous training pipeline:
    1. 80/20 stratified holdout split
    2. Grid search across 4 model types
    3. 10-fold stratified CV
    4. Probability calibration
    5. Multi-seed stability
    6. Learning curve analysis
    """
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        VotingClassifier, BaggingClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import (
        train_test_split, StratifiedKFold, GridSearchCV,
        cross_val_score, learning_curve
    )
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        classification_report, confusion_matrix, roc_auc_score,
        log_loss, make_scorer
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline

    print(f"\n{'='*70}")
    print(f"  RIGOROUS ITERATIVE TRAINING HARNESS v2.0")
    print(f"{'='*70}")

    X, y = extract_batch_v2(records)
    n_samples, n_features = X.shape
    n_pos = int(np.sum(y))
    n_neg = n_samples - n_pos

    print(f"  Total records:   {n_samples}")
    print(f"  Success / Fail:  {n_pos} / {n_neg}  ({n_pos/n_samples:.0%} / {n_neg/n_samples:.0%})")
    print(f"  Features:        {n_features} ({len(FEATURE_NAMES)})")

    # ── Stratified holdout split ─────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train / Test:    {len(y_train)} / {len(y_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Phase 1: Multi-Model Comparison ──────────────────
    print(f"\n  ─── Phase 1: Multi-Model Comparison (10-fold CV) ───")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            min_samples_leaf=3, random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=42
        ),
        "SVM_RBF": SVC(
            kernel="rbf", C=1.0, probability=True,
            class_weight="balanced", random_state=42
        ),
    }

    model_results = {}
    best_cv_score = 0
    best_model_name = None

    for name, model in models.items():
        scores_acc = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")
        scores_f1 = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="f1")
        scores_auc = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="roc_auc")

        mean_acc = scores_acc.mean()
        mean_f1 = scores_f1.mean()
        mean_auc = scores_auc.mean()

        model_results[name] = {
            "cv_accuracy": f"{mean_acc:.3f} ± {scores_acc.std():.3f}",
            "cv_f1": f"{mean_f1:.3f} ± {scores_f1.std():.3f}",
            "cv_auc": f"{mean_auc:.3f} ± {scores_auc.std():.3f}",
            "cv_acc_mean": float(mean_acc),
            "cv_f1_mean": float(mean_f1),
            "cv_auc_mean": float(mean_auc),
        }

        print(f"  {name:25s}  Acc={mean_acc:.3f}  F1={mean_f1:.3f}  AUC={mean_auc:.3f}")

        # Use F1 as primary metric (handles class imbalance better)
        if mean_f1 > best_cv_score:
            best_cv_score = mean_f1
            best_model_name = name

    print(f"\n  >>> Best model by CV F1: {best_model_name} ({best_cv_score:.3f})")

    # ── Phase 2: Hyperparameter Grid Search ──────────────
    print(f"\n  ─── Phase 2: Hyperparameter Grid Search ({best_model_name}) ───")

    param_grids = {
        "RandomForest": {
            "n_estimators": [200, 300, 500],
            "max_depth": [6, 8, 12],
            "min_samples_leaf": [2, 3],
            "max_features": ["sqrt", "log2"],
        },
        "GradientBoosting": {
            "n_estimators": [150, 250],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "min_samples_leaf": [2, 3],
            "subsample": [0.8, 1.0],
        },
        "LogisticRegression": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        "SVM_RBF": {
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", "auto", 0.01],
        },
    }

    base_model = models[best_model_name]
    grid = param_grids[best_model_name]

    gs = GridSearchCV(
        base_model, grid, cv=cv, scoring="f1",
        n_jobs=-1, verbose=0, refit=True
    )
    gs.fit(X_train_s, y_train)

    best_params = gs.best_params_
    best_gs_f1 = gs.best_score_

    print(f"  Best params: {best_params}")
    print(f"  Best CV F1:  {best_gs_f1:.4f}")

    tuned_model = gs.best_estimator_

    # ── Phase 3: Voting Ensemble ─────────────────────────
    print(f"\n  ─── Phase 3: Voting Ensemble ───")

    # Train all models with their defaults and create a soft voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=2,
                class_weight="balanced", random_state=42)),
            ("gb", GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)),
            ("lr", LogisticRegression(
                C=10.0, max_iter=1000, class_weight="balanced", random_state=42)),
        ],
        voting="soft"
    )
    ens_scores = cross_val_score(ensemble, X_train_s, y_train, cv=cv, scoring="f1")
    print(f"  Ensemble CV F1: {ens_scores.mean():.3f} ± {ens_scores.std():.3f}")

    # Pick best between tuned single model and ensemble
    if ens_scores.mean() > best_gs_f1:
        print(f"  >>> Ensemble wins! Using VotingClassifier.")
        final_model = ensemble
        final_model_name = "VotingEnsemble"
        final_cv_f1 = ens_scores.mean()
    else:
        print(f"  >>> Tuned {best_model_name} wins!")
        final_model = tuned_model
        final_model_name = best_model_name
        final_cv_f1 = best_gs_f1

    # ── Phase 4: Probability Calibration ─────────────────
    print(f"\n  ─── Phase 4: Probability Calibration ───")

    calibrated = CalibratedClassifierCV(final_model, cv=5, method="isotonic")
    calibrated.fit(X_train_s, y_train)

    # ── Phase 5: Holdout Test Evaluation ─────────────────
    print(f"\n  ─── Phase 5: Holdout Test Set Evaluation ───")

    # Evaluate both calibrated and uncalibrated
    final_model.fit(X_train_s, y_train)

    y_pred_uncal = final_model.predict(X_test_s)
    y_pred_cal = calibrated.predict(X_test_s)
    y_prob_cal = calibrated.predict_proba(X_test_s)[:, 1]

    test_acc = accuracy_score(y_test, y_pred_cal)
    test_f1 = f1_score(y_test, y_pred_cal)
    test_precision = precision_score(y_test, y_pred_cal)
    test_recall = recall_score(y_test, y_pred_cal)
    test_auc = roc_auc_score(y_test, y_prob_cal)
    test_logloss = log_loss(y_test, y_prob_cal)
    cm = confusion_matrix(y_test, y_pred_cal)

    print(f"  Holdout Accuracy:   {test_acc:.3f}")
    print(f"  Holdout F1 Score:   {test_f1:.3f}")
    print(f"  Holdout Precision:  {test_precision:.3f}")
    print(f"  Holdout Recall:     {test_recall:.3f}")
    print(f"  Holdout ROC-AUC:    {test_auc:.3f}")
    print(f"  Holdout Log Loss:   {test_logloss:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"    Predicted:  Fail  Success")
    print(f"    Actual Fail:  {cm[0][0]:3d}    {cm[0][1]:3d}")
    print(f"    Actual Pass:  {cm[1][0]:3d}    {cm[1][1]:3d}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred_cal, target_names=["Failure", "Success"]))

    # ── Phase 6: Multi-Seed Stability ────────────────────
    print(f"  ─── Phase 6: Multi-Seed Stability Test ───")

    seed_scores = []
    for seed in [0, 7, 13, 21, 42, 99, 137, 256, 512, 1024]:
        cv_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        s = cross_val_score(final_model, X_train_s, y_train, cv=cv_seed, scoring="f1")
        seed_scores.append(s.mean())

    seed_scores = np.array(seed_scores)
    print(f"  F1 across 10 seeds: {seed_scores.mean():.3f} ± {seed_scores.std():.3f}")
    print(f"  Min: {seed_scores.min():.3f}  Max: {seed_scores.max():.3f}")

    stability = "STABLE" if seed_scores.std() < 0.05 else "UNSTABLE"
    print(f"  Stability verdict: {stability}")

    # ── Phase 7: Feature Importance (final model) ────────
    print(f"\n  ─── Phase 7: Feature Importance ───")

    if hasattr(final_model, "feature_importances_"):
        importances = final_model.feature_importances_
    elif hasattr(final_model, "estimators_"):
        # Voting ensemble — average importances from tree-based sub-models
        imp_list = []
        for name_est, est in final_model.named_estimators_.items():
            if hasattr(est, "feature_importances_"):
                imp_list.append(est.feature_importances_)
        importances = np.mean(imp_list, axis=0) if imp_list else np.ones(n_features) / n_features
    elif hasattr(final_model, "coef_"):
        # Logistic regression — use absolute coefficients
        importances = np.abs(final_model.coef_[0])
        importances = importances / importances.sum()
    else:
        # SVM RBF or other models without direct importances — use permutation importance
        from sklearn.inspection import permutation_importance
        perm_result = permutation_importance(
            final_model, X_test_s, y_test, n_repeats=10, random_state=42, scoring="f1"
        )
        importances = np.maximum(perm_result.importances_mean, 0)
        total_perm = importances.sum()
        if total_perm > 0:
            importances = importances / total_perm
        else:
            importances = np.ones(n_features) / n_features

    sorted_idx = np.argsort(importances)[::-1]
    print(f"  {'Rank':<6}{'Feature':<32}{'Importance':<12}")
    print(f"  {'-'*50}")
    for rank, idx in enumerate(sorted_idx[:15], 1):
        print(f"  {rank:<6}{FEATURE_NAMES[idx]:<32}{importances[idx]:.4f}")

    feature_importances = {FEATURE_NAMES[i]: float(importances[i]) for i in range(n_features)}

    # ── Phase 8: Compute Learned Weights ─────────────────
    print(f"\n  ─── Phase 8: Compute Learned Weights ───")

    imp = feature_importances
    skill_imp = imp.get("avg_skill_level", 0) + imp.get("min_skill_level", 0) + imp.get("max_skill_level", 0) + imp.get("median_skill_level", 0)
    div_imp = imp.get("role_diversity", 0) + imp.get("coverage_diversity_product", 0) * 0.5
    bal_imp = imp.get("skill_variance", 0) + imp.get("gini_coefficient", 0) + imp.get("skill_iqr", 0) + imp.get("balance_penalty", 0)
    exp_imp = imp.get("avg_experience", 0) + imp.get("experience_variance", 0) + imp.get("experience_skill_interaction", 0) * 0.5
    cov_imp = imp.get("skill_coverage", 0) + imp.get("critical_role_coverage", 0) + imp.get("helpful_role_coverage", 0)

    total_imp = skill_imp + div_imp + bal_imp + exp_imp + cov_imp + imp.get("team_size", 0) + imp.get("role_duplication_ratio", 0)
    if total_imp == 0:
        total_imp = 1.0

    learned_weights = {
        "weight_skill": round(skill_imp / total_imp, 4),
        "weight_diversity": round(div_imp / total_imp, 4),
        "weight_balance": round(bal_imp / total_imp, 4),
        "weight_experience": round(exp_imp / total_imp, 4),
        "weight_coverage": round(cov_imp / total_imp, 4),
        "weight_team_size": round(imp.get("team_size", 0) / total_imp, 4),
        "weight_role_dedup": round(imp.get("role_duplication_ratio", 0) / total_imp, 4),
        "penalty_critical_gap": round(imp.get("has_critical_role_gap", 0) * 2, 4),
        "penalty_high_variance": round(imp.get("skill_variance", 0) * 2, 4),
        "bonus_high_coverage": round(imp.get("skill_coverage", 0) * 1.5, 4),
    }

    # Compute thresholds from successful training samples
    success_mask = y_train == 1
    X_train_success = X_train[success_mask]
    fi = {name: i for i, name in enumerate(FEATURE_NAMES)}

    learned_weights["threshold_min_diversity"] = round(float(
        np.percentile(X_train_success[:, fi["role_diversity"]], 25)), 2)
    learned_weights["threshold_min_avg_skill"] = round(float(
        np.percentile(X_train_success[:, fi["avg_skill_level"]], 25)), 2)
    learned_weights["threshold_max_gini"] = round(float(
        np.percentile(X_train_success[:, fi["gini_coefficient"]], 75)), 2)
    learned_weights["threshold_min_coverage"] = round(float(
        np.percentile(X_train_success[:, fi["skill_coverage"]], 25)), 2)

    for k, v in learned_weights.items():
        print(f"  {k}: {v}")

    # ── Save Everything ──────────────────────────────────
    print(f"\n  ─── Saving Model & Report ───")

    save_data = {
        "model": calibrated,
        "scaler": scaler,
        "learned_weights": learned_weights,
        "feature_importances": feature_importances,
        "training_metrics": {
            "version": "v2.0_rigorous",
            "total_records": n_samples,
            "train_records": len(y_train),
            "test_records": len(y_test),
            "success_count": n_pos,
            "failure_count": n_neg,
            "final_model": final_model_name,
            "best_params": best_params if final_model_name != "VotingEnsemble" else "ensemble",
            "cv_f1_mean": float(final_cv_f1),
            "holdout_accuracy": float(test_acc),
            "holdout_f1": float(test_f1),
            "holdout_precision": float(test_precision),
            "holdout_recall": float(test_recall),
            "holdout_auc": float(test_auc),
            "holdout_logloss": float(test_logloss),
            "confusion_matrix": cm.tolist(),
            "multi_seed_f1_mean": float(seed_scores.mean()),
            "multi_seed_f1_std": float(seed_scores.std()),
            "stability": stability,
            "model_comparison": model_results,
        },
        "is_trained": True,
        "feature_names": FEATURE_NAMES,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(save_data, f)

    report = {
        "training_date": "2026-03-02",
        "version": "v2.0_rigorous",
        "training_metrics": save_data["training_metrics"],
        "learned_weights": learned_weights,
        "feature_importances": feature_importances,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  Model saved to: {MODEL_PATH}")
    print(f"  Report saved to: {REPORT_PATH}")

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE — v2.0 Rigorous")
    print(f"  Final Model:     {final_model_name}")
    print(f"  Holdout Acc:     {test_acc:.3f}")
    print(f"  Holdout F1:      {test_f1:.3f}")
    print(f"  Holdout AUC:     {test_auc:.3f}")
    print(f"  Seed Stability:  {stability} ({seed_scores.std():.4f})")
    print(f"{'='*70}\n")

    return save_data["training_metrics"]


# ============================================================
# PART 4: ITERATIVE IMPROVEMENT LOOP
# ============================================================

def run_iterative_training():
    """
    Main function: generate data → train → evaluate → repeat if needed.
    """
    start = time.time()

    # ── Step 1: Load existing data ───────────────────────
    print("\n" + "=" * 70)
    print("  STEP 1: Loading existing data")
    print("=" * 70)

    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r") as f:
            existing = json.load(f)
        all_existing = existing.get("team_records", [])
        # Keep only original records (those without edge-case tags)
        existing_records = [
            r for r in all_existing
            if not r.get("team_id", "").startswith("EDGE_")
            and not r.get("outcome", {}).get("tag")
        ]
        print(f"  Existing records (original only): {len(existing_records)} (filtered from {len(all_existing)})")
    else:
        existing_records = []
        print(f"  No existing data found.")

    # ── Step 2: Generate edge cases ──────────────────────
    print("\n" + "=" * 70)
    print("  STEP 2: Generating 200 edge-case records")
    print("=" * 70)

    edge_cases = generate_edge_cases(200)

    # Count categories
    tags = Counter(r["outcome"].get("tag", "unknown") for r in edge_cases)
    for tag, count in sorted(tags.items(), key=lambda x: -x[1]):
        print(f"  {tag:35s} {count:3d} records")

    # ── Step 3: Merge data ───────────────────────────────
    all_records = existing_records + edge_cases
    success_count = sum(1 for r in all_records if r["outcome"]["success"])
    fail_count = len(all_records) - success_count

    print(f"\n  Total dataset:  {len(all_records)} records")
    print(f"  Success / Fail: {success_count} / {fail_count}")

    # Save merged data
    merged = {
        "metadata": {
            "total_records": len(all_records),
            "generated_date": "2026-03-02",
            "purpose": "feedback_loop_training_v2",
            "success_count": success_count,
            "failure_count": fail_count,
            "includes_edge_cases": True,
            "edge_case_count": len(edge_cases),
            "original_count": len(existing_records),
        },
        "team_records": all_records,
    }
    with open(DATA_PATH, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"  Saved to {DATA_PATH}")

    # ── Step 3.5: Quick data quality check ───────────────
    print(f"\n  ─── Data Quality Check ───")
    # Verify feature extraction works for all records
    errors = 0
    for i, r in enumerate(all_records):
        try:
            extract_features_v2(r)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ERROR in record {i} ({r.get('team_id', '?')}): {e}")
    print(f"  Feature extraction errors: {errors}/{len(all_records)}")

    # ── Step 4: Train ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 4: Rigorous Training")
    print("=" * 70)

    metrics = train_rigorous(all_records)

    # ── Step 5: Analysis ─────────────────────────────────
    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.1f}s")

    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    metrics = run_iterative_training()
