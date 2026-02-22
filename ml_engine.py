"""
SkillSyncAI - ML Engine
Contains all the ML logic for resume parsing, skill extraction, and team formation.
"""

import os
import re
import math
from collections import defaultdict, Counter
from typing import Dict, List, Optional
import pdfplumber

# ============================================================
# SKILL DICTIONARY — Maps keywords to role categories
# ============================================================

SKILL_DICTIONARY = {
    "frontend": [
        "react", "reactjs", "react.js", "angular", "angularjs", "vue", "vuejs",
        "vue.js", "html", "html5", "css", "css3", "javascript", "typescript",
        "jquery", "bootstrap", "tailwind", "tailwindcss", "sass", "scss",
        "webpack", "vite", "next.js", "nextjs", "nuxt", "svelte",
        "responsive design", "front-end", "front end", "redux", "zustand",
        "material ui", "chakra ui", "ant design", "styled components",
    ],
    "backend": [
        "node", "nodejs", "node.js", "express", "expressjs", "django",
        "flask", "fastapi", "spring", "spring boot", "springboot",
        "java", "python", "ruby", "rails", "ruby on rails", "php", "laravel",
        "golang", "go", "rust", "c#", ".net", "asp.net", "rest api", "restful",
        "graphql", "grpc", "microservices", "back-end", "back end",
        "mongodb", "mysql", "postgresql", "postgres", "sqlite", "redis",
        "firebase", "supabase", "dynamodb", "cassandra", "sql", "nosql",
        "orm", "sequelize", "prisma", "mongoose", "hibernate",
        "socket.io", "websocket", "authentication", "jwt", "oauth",
    ],
    "fullstack": [
        "fullstack", "full stack", "full-stack", "mern", "mean",
        "lamp", "mern stack", "mean stack", "t3 stack",
    ],
    "ml": [
        "machine learning", "deep learning", "tensorflow", "pytorch",
        "keras", "scikit-learn", "sklearn", "neural network", "neural networks",
        "nlp", "natural language processing", "computer vision", "opencv",
        "classification", "regression", "clustering", "random forest",
        "xgboost", "lightgbm", "gradient boosting", "svm",
        "support vector", "decision tree", "logistic regression",
        "recurrent neural", "convolutional neural", "cnn", "rnn", "lstm",
        "transformer", "bert", "gpt", "llm", "large language model",
        "hugging face", "huggingface", "generative ai", "gen ai",
        "reinforcement learning", "model training", "model deployment",
        "feature engineering", "hyperparameter", "cross validation",
        "artificial intelligence", "ai", "ml",
    ],
    "data": [
        "data analysis", "data analytics", "data science", "data engineering",
        "pandas", "numpy", "matplotlib", "seaborn", "plotly",
        "data visualization", "data mining", "etl", "data pipeline",
        "big data", "spark", "hadoop", "apache spark", "kafka",
        "tableau", "power bi", "powerbi", "excel", "statistics",
        "statistical analysis", "hypothesis testing", "a/b testing",
        "data wrangling", "data cleaning", "exploratory data analysis",
        "eda", "jupyter", "notebook", "r programming",
    ],
    "uiux": [
        "figma", "adobe xd", "sketch", "invision", "zeplin",
        "ui", "ux", "ui/ux", "ui ux", "user interface", "user experience",
        "wireframe", "wireframing", "prototype", "prototyping",
        "design thinking", "user research", "usability testing",
        "interaction design", "visual design", "graphic design",
        "adobe photoshop", "photoshop", "illustrator", "canva",
        "color theory", "typography", "responsive design",
        "information architecture", "heuristic evaluation",
    ],
    "devops": [
        "docker", "kubernetes", "k8s", "aws", "azure", "gcp",
        "google cloud", "amazon web services", "cloud computing",
        "ci/cd", "cicd", "jenkins", "github actions", "gitlab ci",
        "terraform", "ansible", "linux", "bash", "shell scripting",
        "nginx", "apache", "devops", "dev ops", "site reliability",
        "monitoring", "prometheus", "grafana", "elk stack",
        "infrastructure", "deployment", "serverless", "lambda",
        "ec2", "s3", "load balancer", "networking",
        "git", "version control",
    ],
}

ROLE_DISPLAY_NAMES = {
    "frontend": "Frontend",
    "backend": "Backend",
    "fullstack": "Fullstack",
    "ml": "ML/AI",
    "data": "Data",
    "uiux": "UI/UX",
    "devops": "DevOps",
}

ROLE_SYNONYMS = {
    "web developer": "frontend",
    "front end developer": "frontend",
    "frontend developer": "frontend",
    "react developer": "frontend",
    "angular developer": "frontend",
    "ui developer": "frontend",
    "backend developer": "backend",
    "back end developer": "backend",
    "server side developer": "backend",
    "api developer": "backend",
    "java developer": "backend",
    "python developer": "backend",
    "node developer": "backend",
    "software engineer": "fullstack",
    "software developer": "fullstack",
    "sde": "fullstack",
    "full stack developer": "fullstack",
    "fullstack developer": "fullstack",
    "mern stack developer": "fullstack",
    "mean stack developer": "fullstack",
    "ml engineer": "ml",
    "machine learning engineer": "ml",
    "ai engineer": "ml",
    "deep learning engineer": "ml",
    "nlp engineer": "ml",
    "data scientist": "data",
    "data analyst": "data",
    "data engineer": "data",
    "business analyst": "data",
    "ui ux designer": "uiux",
    "ui/ux designer": "uiux",
    "ux designer": "uiux",
    "ui designer": "uiux",
    "graphic designer": "uiux",
    "product designer": "uiux",
    "devops engineer": "devops",
    "cloud engineer": "devops",
    "site reliability engineer": "devops",
    "sre": "devops",
    "system administrator": "devops",
}

EXPERIENCE_KEYWORDS = {
    "internship": 1.0,
    "intern": 1.0,
    "hackathon": 0.8,
    "project": 0.5,
    "freelance": 0.8,
    "work experience": 1.2,
    "professional experience": 1.2,
    "research": 0.7,
    "publication": 0.9,
    "open source": 0.7,
    "contribution": 0.4,
    "certification": 0.6,
    "certified": 0.6,
    "award": 0.5,
    "achievement": 0.4,
    "competition": 0.6,
    "winner": 0.7,
    "runner up": 0.5,
    "lead": 0.8,
    "team lead": 1.0,
    "mentor": 0.6,
}

ALL_ROLES = list(SKILL_DICTIONARY.keys())

# Matching weights
WEIGHT_SKILL = 0.6
WEIGHT_EXPERIENCE = 0.3
WEIGHT_DIVERSITY = 0.1


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text.strip()


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract all text from PDF bytes (for uploaded files)."""
    import io
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF bytes: {e}")
    return text.strip()


def extract_name_from_filename(filename: str) -> str:
    """Extract student name from the PDF filename."""
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r'[-_\s]*(resume|cv)[-_\s]*', ' ', name, flags=re.IGNORECASE)
    name = name.replace('_', ' ').replace('-', ' ')
    name = re.sub(r'\d+', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.title()
    return name if name else "Unknown"


# ============================================================
# SKILL EXTRACTION
# ============================================================

def preprocess_text(text: str) -> str:
    """Clean and normalize resume text."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s\.\-\/\#\+]', ' ', text)
    return text.strip()


def extract_skills(text: str) -> Dict[str, Dict]:
    """Extract skills from resume text using keyword matching."""
    cleaned = preprocess_text(text)
    role_data = {}

    for role, keywords in SKILL_DICTIONARY.items():
        matched = []
        raw_count = 0

        for keyword in keywords:
            if len(keyword) <= 3:
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                pattern = re.escape(keyword)

            count = len(re.findall(pattern, cleaned))
            if count > 0:
                matched.append(keyword)
                raw_count += count

        role_data[role] = {
            "raw_count": raw_count,
            "matched_keywords": matched,
            "unique_skills": len(matched),
        }

    max_possible = max(
        (r["unique_skills"] * 2 + r["raw_count"] * 0.5) for r in role_data.values()
    ) if any(r["raw_count"] > 0 for r in role_data.values()) else 1

    for role in role_data:
        r = role_data[role]
        raw_score = r["unique_skills"] * 2 + r["raw_count"] * 0.5
        normalized = min(10, round((raw_score / max(max_possible, 1)) * 10, 1))
        role_data[role]["score"] = normalized

    return role_data


def calculate_skill_percentages(text: str, skill_data: Dict[str, Dict]) -> Dict[str, int]:
    """
    Calculate percentage proficiency for each individual skill found.
    
    Scoring factors:
    - Base score from mention count (more mentions = higher confidence)
    - Context bonus (appears in skills section, projects, etc.)
    - Variability for realistic distribution
    
    Returns:
        Dict mapping skill name -> percentage (40-95%)
    """
    import random
    import hashlib
    
    cleaned = preprocess_text(text)
    skill_percentages = {}
    
    # Collect all matched skills with their counts
    skill_counts = {}
    for role, data in skill_data.items():
        for keyword in data.get("matched_keywords", []):
            # Count occurrences
            if len(keyword) <= 3:
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                pattern = re.escape(keyword)
            count = len(re.findall(pattern, cleaned))
            skill_counts[keyword] = count
    
    if not skill_counts:
        return {}
    
    # Get max count for normalization
    max_count = max(skill_counts.values())
    
    # Context patterns that indicate proficiency
    proficiency_patterns = [
        r'expert\s+in|proficient\s+in|advanced\s+in|strong\s+in',
        r'\d+\s*\+?\s*years?.*experience',
        r'built|developed|created|implemented|designed|led',
        r'certified|certification',
    ]
    
    for skill, count in skill_counts.items():
        # Base score: 50-75 based on mention frequency
        base_score = 50 + (count / max(max_count, 1)) * 25
        
        # Context bonus: check if skill appears in proficiency context
        context_bonus = 0
        skill_context = re.findall(r'.{0,50}' + re.escape(skill) + r'.{0,50}', cleaned)
        for context in skill_context:
            for pattern in proficiency_patterns:
                if re.search(pattern, context):
                    context_bonus += 5
                    break
        context_bonus = min(context_bonus, 15)  # Cap at 15
        
        # Deterministic "random" variation based on skill name (consistent across runs)
        hash_val = int(hashlib.md5(skill.encode()).hexdigest()[:8], 16)
        variation = (hash_val % 15) - 7  # -7 to +7
        
        # Final score: 40-95 range
        final_score = int(min(95, max(40, base_score + context_bonus + variation)))
        
        # Capitalize skill name nicely
        display_name = skill.title() if len(skill) > 3 else skill.upper()
        # Special cases for common skills
        display_names = {
            "javascript": "JavaScript", "typescript": "TypeScript",
            "nodejs": "Node.js", "node.js": "Node.js", "reactjs": "React.js",
            "react.js": "React.js", "vuejs": "Vue.js", "vue.js": "Vue.js",
            "angularjs": "AngularJS", "mongodb": "MongoDB", "mysql": "MySQL",
            "postgresql": "PostgreSQL", "graphql": "GraphQL", "github": "GitHub",
            "gitlab": "GitLab", "tensorflow": "TensorFlow", "pytorch": "PyTorch",
            "opencv": "OpenCV", "scikit-learn": "Scikit-learn", "sklearn": "Scikit-learn",
            "aws": "AWS", "gcp": "GCP", "ci/cd": "CI/CD", "docker": "Docker",
            "kubernetes": "Kubernetes", "html": "HTML", "css": "CSS",
            "sql": "SQL", "nosql": "NoSQL", "api": "API", "rest api": "REST API",
            "html5": "HTML5", "css3": "CSS3", "jwt": "JWT", "oauth": "OAuth",
            "llm": "LLM", "nlp": "NLP", "cnn": "CNN", "rnn": "RNN", "lstm": "LSTM",
            "ai": "AI", "ml": "ML", "ui": "UI", "ux": "UX", "ui/ux": "UI/UX",
        }
        display_name = display_names.get(skill.lower(), display_name)
        
        skill_percentages[display_name] = final_score
    
    # Sort by percentage descending
    skill_percentages = dict(
        sorted(skill_percentages.items(), key=lambda x: x[1], reverse=True)
    )
    
    return skill_percentages


def determine_primary_role(skill_data: Dict[str, Dict]) -> str:
    """Determine the student's primary role based on highest skill score."""
    best_role = "backend"
    best_score = -1
    best_unique = -1

    for role, data in skill_data.items():
        score = data["score"]
        unique = data["unique_skills"]

        if score > best_score or (score == best_score and unique > best_unique):
            best_score = score
            best_unique = unique
            best_role = role

    if best_role == "fullstack":
        fe_score = skill_data.get("frontend", {}).get("score", 0)
        be_score = skill_data.get("backend", {}).get("score", 0)
        if fe_score > be_score and fe_score > skill_data["fullstack"]["score"]:
            best_role = "frontend"
        elif be_score > fe_score and be_score > skill_data["fullstack"]["score"]:
            best_role = "backend"

    return best_role


def infer_role_from_title(text: str) -> Optional[str]:
    """Check if the resume contains explicit role titles."""
    cleaned = text.lower()
    for title, role in ROLE_SYNONYMS.items():
        if title in cleaned:
            return role
    return None


# ============================================================
# EXPERIENCE SCORING
# ============================================================

def calculate_experience_score(text: str) -> Dict:
    """Calculate experience score from resume text."""
    cleaned = preprocess_text(text)
    matched = []
    weighted_total = 0.0

    for keyword, weight in EXPERIENCE_KEYWORDS.items():
        count = cleaned.count(keyword)
        if count > 0:
            capped_count = min(count, 5)
            weighted_total += capped_count * weight
            matched.append({
                "keyword": keyword,
                "count": count,
                "contribution": round(capped_count * weight, 2)
            })

    normalized = round(min(5.0, (weighted_total / (weighted_total + 5)) * 10), 2)

    return {
        "score": normalized,
        "raw_total": round(weighted_total, 2),
        "matched_keywords": matched,
    }


# ============================================================
# STUDENT PROFILE BUILDER
# ============================================================

def build_student_profile(resume_data: Dict) -> Dict:
    """Build a complete student profile from raw resume data."""
    text = resume_data["raw_text"]

    skill_data = extract_skills(text)
    primary_role = determine_primary_role(skill_data)

    title_role = infer_role_from_title(text)
    if title_role and title_role != primary_role:
        title_score = skill_data.get(title_role, {}).get("score", 0)
        primary_score = skill_data.get(primary_role, {}).get("score", 0)
        if title_score >= primary_score * 0.7:
            primary_role = title_role

    experience = calculate_experience_score(text)

    skill_diversity = sum(
        1 for r in skill_data.values() if r["score"] > 0
    ) / len(ALL_ROLES)

    top_skills = []
    for role in ALL_ROLES:
        for kw in skill_data[role]["matched_keywords"][:3]:
            top_skills.append(kw)

    # Calculate individual skill percentages
    skill_percentages = calculate_skill_percentages(text, skill_data)

    profile = {
        "student_id": resume_data["student_id"],
        "name": resume_data["name"],
        "filename": resume_data["filename"],
        "skills": {role: data["score"] for role, data in skill_data.items()},
        "skill_details": skill_data,
        "skill_percentages": skill_percentages,  # NEW: Individual skill scores
        "primary_role": primary_role,
        "experience_score": experience["score"],
        "experience_details": experience,
        "skill_diversity": round(skill_diversity, 2),
        "top_skills": top_skills,
    }

    return profile


def build_student_profile_with_github(
    resume_data: Dict,
    github_skills: Optional[Dict] = None,
    resume_weight: float = 0.6,
    github_weight: float = 0.4
) -> Dict:
    """
    Build a complete student profile from resume data, optionally enhanced with GitHub data.
    
    Args:
        resume_data: Parsed resume data with raw_text
        github_skills: Skills extracted from GitHub profile (optional)
        resume_weight: Weight for resume skills (default 0.6)
        github_weight: Weight for GitHub skills (default 0.4)
    
    Returns:
        Complete student profile with merged skills
    """
    text = resume_data["raw_text"]

    # Extract skills from resume
    resume_skill_data = extract_skills(text)
    
    # Merge with GitHub skills if provided
    if github_skills:
        merged_skills = {}
        for role in ALL_ROLES:
            resume_score = resume_skill_data.get(role, {}).get("score", 0)
            github_score = github_skills.get(role, {}).get("score", 0)
            
            # Weighted average
            combined_score = (
                resume_score * resume_weight +
                github_score * github_weight
            )
            
            # Merge keywords
            resume_keywords = resume_skill_data.get(role, {}).get("matched_keywords", [])
            github_keywords = github_skills.get(role, {}).get("matched_keywords", [])
            
            merged_skills[role] = {
                "score": round(combined_score, 1),
                "matched_keywords": resume_keywords[:5] + github_keywords[:3],
                "unique_skills": len(set(resume_keywords[:5] + github_keywords[:3])),
                "raw_count": resume_skill_data.get(role, {}).get("raw_count", 0),
                "resume_score": resume_score,
                "github_score": github_score,
            }
        skill_data = merged_skills
    else:
        skill_data = resume_skill_data
    
    primary_role = determine_primary_role(skill_data)

    title_role = infer_role_from_title(text)
    if title_role and title_role != primary_role:
        title_score = skill_data.get(title_role, {}).get("score", 0)
        primary_score = skill_data.get(primary_role, {}).get("score", 0)
        if title_score >= primary_score * 0.7:
            primary_role = title_role

    experience = calculate_experience_score(text)
    
    # Boost experience score if GitHub shows activity
    if github_skills:
        github_activity_bonus = min(0.5, sum(
            1 for r in github_skills.values() 
            if r.get("score", 0) > 3
        ) * 0.1)
        experience["score"] = min(5.0, experience["score"] + github_activity_bonus)

    skill_diversity = sum(
        1 for r in skill_data.values() if r.get("score", 0) > 0
    ) / len(ALL_ROLES)

    top_skills = []
    for role in ALL_ROLES:
        for kw in skill_data.get(role, {}).get("matched_keywords", [])[:3]:
            if kw not in top_skills:
                top_skills.append(kw)

    # Calculate individual skill percentages
    skill_percentages = calculate_skill_percentages(text, skill_data)
    
    # Add GitHub skills to percentages with bonus
    if github_skills:
        for role, gh_data in github_skills.items():
            for kw in gh_data.get("matched_keywords", []):
                display_name = kw.title() if len(kw) > 3 else kw.upper()
                if display_name not in skill_percentages:
                    # GitHub-only skills get a base score
                    skill_percentages[display_name] = 55 + (hash(kw) % 20)
                else:
                    # Boost existing skills found on GitHub
                    skill_percentages[display_name] = min(95, skill_percentages[display_name] + 8)
        # Re-sort
        skill_percentages = dict(sorted(skill_percentages.items(), key=lambda x: x[1], reverse=True))

    profile = {
        "student_id": resume_data["student_id"],
        "name": resume_data["name"],
        "filename": resume_data["filename"],
        "skills": {role: data.get("score", 0) for role, data in skill_data.items()},
        "skill_details": skill_data,
        "skill_percentages": skill_percentages,  # NEW: Individual skill scores
        "primary_role": primary_role,
        "experience_score": experience["score"],
        "experience_details": experience,
        "skill_diversity": round(skill_diversity, 2),
        "top_skills": top_skills,
        "has_github": github_skills is not None,
    }

    return profile


# ============================================================
# PROJECT REQUIREMENT PARSER
# ============================================================

def parse_project_requirements(requirement_text: str, team_size: int = 4) -> Dict:
    """Parse a plain-English project requirement to extract needed roles."""
    cleaned = preprocess_text(requirement_text)
    role_relevance = {}

    for role, keywords in SKILL_DICTIONARY.items():
        matched = []
        for keyword in keywords:
            if keyword in cleaned:
                matched.append(keyword)

        if matched:
            role_relevance[role] = {
                "match_count": len(matched),
                "matched_keywords": matched,
            }

    for title, role in ROLE_SYNONYMS.items():
        if title in cleaned:
            if role not in role_relevance:
                role_relevance[role] = {"match_count": 0, "matched_keywords": []}
            role_relevance[role]["match_count"] += 2
            role_relevance[role]["matched_keywords"].append(f"[role: {title}]")

    sorted_roles = sorted(
        role_relevance.items(),
        key=lambda x: x[1]["match_count"],
        reverse=True
    )

    roles_needed = [role for role, _ in sorted_roles[:team_size]]

    if len(roles_needed) < team_size:
        default_fill = ["backend", "frontend", "fullstack", "data", "devops", "ml", "uiux"]
        for fill_role in default_fill:
            if fill_role not in roles_needed and len(roles_needed) < team_size:
                roles_needed.append(fill_role)

    priority_skills = []
    for role_info in role_relevance.values():
        for kw in role_info["matched_keywords"]:
            if not kw.startswith("[role:"):
                priority_skills.append(kw)

    result = {
        "requirement_text": requirement_text,
        "roles_needed": roles_needed,
        "team_size": team_size,
        "priority_skills": priority_skills,
        "role_relevance": {role: data for role, data in sorted_roles},
    }

    return result


# ============================================================
# TEAM FORMATION ENGINE
# ============================================================

def calculate_optimal_team_sizes(total_students: int, target_size: int) -> List[int]:
    """Calculate optimal team sizes to keep all teams as equal as possible."""
    if total_students <= 0:
        return []

    if target_size <= 0:
        target_size = 4

    num_teams = math.ceil(total_students / target_size)
    num_teams = max(1, num_teams)

    base_size = total_students // num_teams
    remainder = total_students % num_teams

    team_sizes = []
    for i in range(num_teams):
        if i < remainder:
            team_sizes.append(base_size + 1)
        else:
            team_sizes.append(base_size)

    return team_sizes


def calculate_student_overall_score(profile: Dict) -> float:
    """Calculate an overall strength score for a student."""
    max_skill = max(profile["skills"].values()) if profile["skills"] else 0
    exp_score = profile["experience_score"] / 5.0
    diversity = profile["skill_diversity"]
    overall = 0.5 * (max_skill / 10.0) + 0.35 * exp_score + 0.15 * diversity
    return round(overall, 4)


def _calculate_team_balance(team: Dict) -> Dict:
    """Calculate balance metrics for a team."""
    if not team["members"]:
        return {
            "average_score": 0,
            "score_variance": 0,
            "role_diversity": 0,
            "overall_balance": 0
        }

    scores = [m["overall_score"] for m in team["members"]]
    avg = sum(scores) / len(scores)
    variance = sum((s - avg) ** 2 for s in scores) / len(scores)

    unique_roles = len(team.get("roles_filled", set()))
    role_diversity = unique_roles / len(team["members"])

    balance = (
        0.35 * avg +
        0.35 * (1 - min(variance * 10, 1)) +
        0.30 * role_diversity
    )

    return {
        "average_score": round(avg, 3),
        "score_variance": round(variance, 4),
        "role_diversity": round(role_diversity, 2),
        "overall_balance": round(balance, 3),
    }


def form_balanced_teams(
    profiles: List[Dict],
    target_team_size: int = 4,
    project_requirements: Dict = None
) -> List[Dict]:
    """
    Form balanced teams from all students using Snake Draft + Role Balancing.
    """
    if not profiles:
        return []

    total_students = len(profiles)
    profile_lookup = {p["student_id"]: p for p in profiles}

    team_sizes = calculate_optimal_team_sizes(total_students, target_team_size)
    num_teams = len(team_sizes)

    # Initialize empty teams
    teams = []
    for i, size in enumerate(team_sizes):
        teams.append({
            "team_id": f"Team_{i+1:02d}",
            "team_name": f"Team {i+1}",
            "target_size": size,
            "members": [],
            "roles_filled": set(),
            "total_skill_score": 0.0,
        })

    # Add overall score to each profile
    for profile in profiles:
        profile["_overall_score"] = calculate_student_overall_score(profile)

    # Group students by primary role
    role_groups = defaultdict(list)
    for profile in profiles:
        role_groups[profile["primary_role"]].append(profile)

    # Sort each role group by overall score
    for role in role_groups:
        role_groups[role].sort(key=lambda p: p["_overall_score"], reverse=True)

    assigned = set()

    role_priority = ["backend", "frontend", "fullstack", "ml", "data", "uiux", "devops"]
    if project_requirements and "roles_needed" in project_requirements:
        role_priority = project_requirements["roles_needed"] + [
            r for r in role_priority if r not in project_requirements["roles_needed"]
        ]

    # SNAKE DRAFT
    max_size = max(team_sizes)

    for round_num in range(max_size):
        if round_num % 2 == 0:
            draft_order = list(range(num_teams))
        else:
            draft_order = list(range(num_teams - 1, -1, -1))

        current_role_idx = round_num % len(role_priority)

        for team_idx in draft_order:
            team = teams[team_idx]

            if len(team["members"]) >= team["target_size"]:
                continue

            candidate = None
            assigned_role = None

            for role_offset in range(len(role_priority)):
                role = role_priority[(current_role_idx + role_offset) % len(role_priority)]

                if role in team["roles_filled"]:
                    continue

                for student in role_groups.get(role, []):
                    if student["student_id"] not in assigned:
                        candidate = student
                        assigned_role = role
                        break

                if candidate:
                    break

            if not candidate:
                for role in role_priority:
                    for student in role_groups.get(role, []):
                        if student["student_id"] not in assigned:
                            candidate = student
                            assigned_role = student["primary_role"]
                            break
                    if candidate:
                        break

            if candidate:
                member = {
                    "student_id": candidate["student_id"],
                    "name": candidate["name"],
                    "assigned_role": assigned_role,
                    "primary_role": candidate["primary_role"],
                    "overall_score": candidate["_overall_score"],
                    "experience_score": candidate["experience_score"],
                    "skill_scores": candidate["skills"],
                    "top_skills": candidate["top_skills"][:5],
                }
                team["members"].append(member)
                team["roles_filled"].add(assigned_role)
                team["total_skill_score"] += candidate["_overall_score"]
                assigned.add(candidate["student_id"])

    # Calculate balance scores
    for team in teams:
        team["balance_score"] = _calculate_team_balance(team)
        team.pop("total_skill_score", None)

    teams.sort(key=lambda t: t["team_id"])

    return teams


# ============================================================
# EXPLANATION GENERATOR
# ============================================================

def generate_explanation(member: Dict, profile_lookup: Dict) -> str:
    """Generate a human-readable explanation for team assignment."""
    reasons = []
    profile = profile_lookup.get(member["student_id"], {})
    assigned_role = member["assigned_role"]
    role_display = ROLE_DISPLAY_NAMES.get(assigned_role, assigned_role)

    skill_score = member.get("skill_scores", {}).get(assigned_role, 0)
    if skill_score >= 7:
        reasons.append(f"Strong {role_display} skills (score: {skill_score}/10)")
    elif skill_score >= 4:
        reasons.append(f"Moderate {role_display} skills (score: {skill_score}/10)")
    elif skill_score > 0:
        reasons.append(f"Some {role_display} experience (score: {skill_score}/10)")

    if member["primary_role"] == assigned_role:
        reasons.append(f"Primary expertise aligns with {role_display} role")

    if profile and "skill_details" in profile:
        role_details = profile["skill_details"].get(assigned_role, {})
        matched = role_details.get("matched_keywords", [])
        if matched:
            top_kw = matched[:4]
            reasons.append(f"Key skills: {', '.join(top_kw)}")

    exp = member["experience_score"]
    if exp >= 3.5:
        reasons.append(f"Strong practical experience (score: {exp}/5)")
    elif exp >= 2:
        reasons.append(f"Moderate experience (score: {exp}/5)")

    if profile and "experience_details" in profile:
        exp_keywords = profile["experience_details"].get("matched_keywords", [])
        exp_highlights = [
            ek["keyword"] for ek in exp_keywords
            if ek["contribution"] >= 0.5
        ][:3]
        if exp_highlights:
            reasons.append(f"Experience includes: {', '.join(exp_highlights)}")

    score = member.get("overall_score", 0)
    if score >= 0.7:
        reasons.append(f"High overall score ({score:.2f})")
    elif score >= 0.4:
        reasons.append(f"Good overall score ({score:.2f})")

    if not reasons:
        reasons.append("Selected to fill remaining team slot based on overall profile")

    return "; ".join(reasons)


def generate_all_explanations(teams: List[Dict], profiles: List[Dict]) -> List[Dict]:
    """Add explanations to all team members and generate final output."""
    profile_lookup = {p["student_id"]: p for p in profiles}
    explained_teams = []

    for team in teams:
        # Convert set to list for JSON serialization
        roles_filled = list(team.get("roles_filled", set()))
        
        explained_team = {
            "team_name": team["team_name"],
            "team_id": team["team_id"],
            "team_size": len(team["members"]),
            "balance_score": team["balance_score"],
            "members": [],
            "roles_covered": [
                ROLE_DISPLAY_NAMES.get(r, r)
                for r in roles_filled
            ],
        }

        for member in team["members"]:
            explanation = generate_explanation(member, profile_lookup)

            explained_member = {
                "student_id": member["student_id"],
                "name": member["name"],
                "role": ROLE_DISPLAY_NAMES.get(member["assigned_role"], member["assigned_role"]),
                "overall_score": member.get("overall_score", 0),
                "experience_score": member["experience_score"],
                "reason": explanation,
                "top_skills": member.get("top_skills", []),
            }
            explained_team["members"].append(explained_member)

        explained_teams.append(explained_team)

    return explained_teams


# ============================================================
# MAIN PROCESSING FUNCTION (Entry point for API)
# ============================================================

def process_team_formation(
    resumes: List[Dict],  # List of {"filename": str, "content": bytes or str}
    projects: List[Dict],  # List of {"name": str, "description": str, "team_size": int}
    target_team_size: int = 4,
    github_data: Optional[Dict[str, Dict]] = None  # {candidate_name: github_skills}
) -> Dict:
    """
    Main function to process resumes and form teams.
    
    Args:
        resumes: List of resume data with filename and content (bytes for PDF, str for text)
        projects: List of project requirements
        target_team_size: Default team size
        github_data: Optional dict mapping candidate names to their GitHub skill data
    
    Returns:
        Dict with teams, profiles, and summary
    """
    # Step 1: Parse all resumes
    parsed_resumes = []
    for idx, resume in enumerate(resumes, 1):
        filename = resume.get("filename", f"resume_{idx}.pdf")
        content = resume.get("content")
        
        # Extract text based on content type
        if isinstance(content, bytes):
            raw_text = extract_text_from_pdf_bytes(content)
        else:
            raw_text = content if content else ""
        
        if not raw_text:
            continue
        
        name = extract_name_from_filename(filename)
        student_id = f"S{idx:03d}"
        
        parsed_resumes.append({
            "student_id": student_id,
            "filename": filename,
            "name": name,
            "raw_text": raw_text,
        })
    
    if not parsed_resumes:
        return {
            "success": False,
            "error": "No valid resumes could be parsed",
            "teams": [],
            "profiles": [],
        }
    
    # Step 2: Build student profiles (with GitHub data if available)
    student_profiles = []
    for resume in parsed_resumes:
        # Try to match GitHub data by name (case-insensitive)
        candidate_github = None
        if github_data:
            name_lower = resume["name"].lower()
            for github_name, github_skills in github_data.items():
                if github_name.lower() == name_lower or github_name.lower() in name_lower:
                    candidate_github = github_skills.get("skills")
                    break
        
        if candidate_github:
            profile = build_student_profile_with_github(resume, candidate_github)
        else:
            profile = build_student_profile(resume)
        student_profiles.append(profile)
    
    # Step 3: Parse project requirements (if provided)
    parsed_projects = []
    if projects:
        for project in projects:
            parsed = parse_project_requirements(
                project.get("description", ""),
                team_size=project.get("team_size", target_team_size)
            )
            parsed["project_name"] = project.get("name", "Unnamed Project")
            parsed_projects.append(parsed)
    
    # Step 4: Form balanced teams
    project_requirements = parsed_projects[0] if parsed_projects else None
    formed_teams = form_balanced_teams(
        profiles=student_profiles,
        target_team_size=target_team_size,
        project_requirements=project_requirements
    )
    
    # Step 5: Generate explanations
    final_teams = generate_all_explanations(formed_teams, student_profiles)
    
    # Step 6: Prepare profiles for export
    profiles_export = []
    for p in student_profiles:
        profiles_export.append({
            "student_id": p["student_id"],
            "name": p["name"],
            "primary_role": ROLE_DISPLAY_NAMES.get(p["primary_role"], p["primary_role"]),
            "skills": p["skills"],
            "experience_score": p["experience_score"],
            "skill_diversity": p["skill_diversity"],
            "top_skills": p["top_skills"][:8],
        })
    
    # Summary stats
    total_assigned = sum(len(t['members']) for t in final_teams)
    avg_balance = (
        sum(t['balance_score']['overall_balance'] for t in final_teams) / len(final_teams)
        if final_teams else 0
    )
    
    return {
        "success": True,
        "summary": {
            "total_resumes": len(resumes),
            "parsed_resumes": len(parsed_resumes),
            "total_teams": len(final_teams),
            "students_assigned": total_assigned,
            "average_team_balance": round(avg_balance, 3),
        },
        "teams": final_teams,
        "profiles": profiles_export,
        "projects": parsed_projects,
    }
