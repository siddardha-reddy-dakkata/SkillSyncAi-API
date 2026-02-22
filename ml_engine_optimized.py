"""
SkillSyncAI - Optimized ML Engine with Semantic Matching
Uses Sentence Transformers for better skill matching.

IMPROVEMENTS OVER BASIC VERSION:
1. Semantic embeddings for skill matching (not just keyword matching)
2. Cosine similarity for project-student matching
3. Better skill extraction using NLP
4. Configurable matching weights
5. Evaluation metrics for model quality

ACADEMIC TECHNIQUES DEMONSTRATED:
- Transfer Learning (pre-trained BERT/MiniLM embeddings)
- Cosine Similarity for text matching
- Feature Engineering (skill vectors, experience encoding)
- Greedy optimization with constraint satisfaction
"""

import os
import re
import math
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple
import pdfplumber

# ============================================================
# TRY TO IMPORT SENTENCE TRANSFORMERS (fallback to basic if not available)
# ============================================================

USE_SEMANTIC = False
try:
    from sentence_transformers import SentenceTransformer, util
    USE_SEMANTIC = True
    print("✅ Sentence Transformers loaded - using semantic matching")
except ImportError:
    print("⚠️ Sentence Transformers not installed - using keyword matching")
    print("   Install with: pip install sentence-transformers")


# ============================================================
# SEMANTIC SKILL MATCHER (Advanced)
# ============================================================

class SemanticSkillMatcher:
    """
    Uses pre-trained sentence embeddings for semantic skill matching.
    
    Model: all-MiniLM-L6-v2 (fast, good quality)
    - 384-dimensional embeddings
    - Trained on 1B+ sentence pairs
    - Good for semantic similarity tasks
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        self.skill_embeddings = {}
        self.role_embeddings = {}
        
        if USE_SEMANTIC:
            print(f"Loading embedding model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            self._precompute_skill_embeddings()
    
    def _precompute_skill_embeddings(self):
        """Pre-compute embeddings for all known skills for faster matching."""
        all_skills = []
        skill_to_role = {}
        
        for role, skills in SKILL_DICTIONARY.items():
            for skill in skills:
                all_skills.append(skill)
                skill_to_role[skill] = role
        
        if self.model and all_skills:
            embeddings = self.model.encode(all_skills, convert_to_tensor=True)
            for i, skill in enumerate(all_skills):
                self.skill_embeddings[skill] = embeddings[i]
        
        # Also embed role descriptions for project matching
        role_descriptions = {
            "frontend": "frontend web development user interface React Angular Vue HTML CSS JavaScript",
            "backend": "backend server API database Node.js Python Django Flask SQL MongoDB",
            "fullstack": "full stack web development frontend backend complete application",
            "ml": "machine learning artificial intelligence deep learning neural networks NLP computer vision",
            "data": "data science analytics visualization pandas statistics ETL big data",
            "uiux": "UI UX design user experience interface Figma wireframe prototype",
            "devops": "DevOps cloud AWS Docker Kubernetes CI/CD deployment infrastructure",
        }
        
        if self.model:
            for role, desc in role_descriptions.items():
                self.role_embeddings[role] = self.model.encode(desc, convert_to_tensor=True)
    
    def extract_skills_semantic(self, text: str) -> Dict[str, float]:
        """
        Extract skills using semantic similarity.
        Returns role -> similarity score mapping.
        """
        if not self.model:
            return {}
        
        # Encode the resume text
        text_embedding = self.model.encode(text[:5000], convert_to_tensor=True)  # Truncate for speed
        
        role_scores = {}
        for role, role_emb in self.role_embeddings.items():
            similarity = util.cos_sim(text_embedding, role_emb).item()
            role_scores[role] = round(max(0, similarity) * 10, 2)  # Scale to 0-10
        
        return role_scores
    
    def calculate_match_score(self, student_text: str, project_description: str) -> float:
        """
        Calculate semantic similarity between student resume and project requirements.
        """
        if not self.model:
            return 0.0
        
        student_emb = self.model.encode(student_text[:3000], convert_to_tensor=True)
        project_emb = self.model.encode(project_description, convert_to_tensor=True)
        
        similarity = util.cos_sim(student_emb, project_emb).item()
        return round(max(0, similarity), 4)


# ============================================================
# EVALUATION METRICS (for academic presentation)
# ============================================================

class TeamFormationMetrics:
    """
    Evaluation metrics to demonstrate model quality.
    
    Metrics:
    - Team Balance Score: How evenly distributed are skills across teams
    - Role Coverage: Percentage of required roles filled
    - Skill Diversity: Variety of skills in each team
    - Fairness Index: Gini coefficient of team strength distribution
    """
    
    @staticmethod
    def calculate_gini_coefficient(values: List[float]) -> float:
        """
        Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality).
        Lower is better for team balance.
        """
        if not values or len(values) < 2:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        gini = (2 * sum((i + 1) * v for i, v in enumerate(sorted_values))) / (n * sum(sorted_values)) - (n + 1) / n
        return round(abs(gini), 4)
    
    @staticmethod
    def calculate_team_balance(teams: List[Dict]) -> Dict:
        """Calculate overall team formation quality metrics."""
        if not teams:
            return {}
        
        team_strengths = []
        role_coverages = []
        
        for team in teams:
            members = team.get("members", [])
            if members:
                avg_score = sum(m.get("overall_score", 0) for m in members) / len(members)
                team_strengths.append(avg_score)
                
                unique_roles = len(set(m.get("role", m.get("assigned_role", "")) for m in members))
                role_coverages.append(unique_roles / max(len(members), 1))
        
        return {
            "average_team_strength": round(np.mean(team_strengths), 4) if team_strengths else 0,
            "strength_std_dev": round(np.std(team_strengths), 4) if team_strengths else 0,
            "gini_coefficient": TeamFormationMetrics.calculate_gini_coefficient(team_strengths),
            "average_role_coverage": round(np.mean(role_coverages), 4) if role_coverages else 0,
            "num_teams": len(teams),
        }
    
    @staticmethod
    def generate_quality_report(teams: List[Dict], profiles: List[Dict]) -> Dict:
        """Generate a comprehensive quality report for the model."""
        metrics = TeamFormationMetrics.calculate_team_balance(teams)
        
        # Additional metrics
        total_students = len(profiles)
        assigned_students = sum(len(t.get("members", [])) for t in teams)
        
        # Role distribution analysis
        role_counts = Counter(p.get("primary_role", "") for p in profiles)
        
        metrics.update({
            "total_students": total_students,
            "assigned_students": assigned_students,
            "assignment_rate": round(assigned_students / max(total_students, 1), 4),
            "role_distribution": dict(role_counts),
            "model_type": "semantic" if USE_SEMANTIC else "keyword-based",
        })
        
        return metrics


# ============================================================
# SKILL DICTIONARY (same as before, but used as fallback)
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

ALL_ROLES = list(SKILL_DICTIONARY.keys())


# ============================================================
# HYBRID SKILL EXTRACTOR (Combines keyword + semantic)
# ============================================================

class HybridSkillExtractor:
    """
    Combines rule-based keyword matching with semantic embeddings.
    
    Strategy:
    - Use keyword matching for precision (exact skill mentions)
    - Use semantic matching for recall (similar concepts)
    - Weighted combination of both scores
    """
    
    def __init__(self, semantic_weight: float = 0.4):
        self.semantic_weight = semantic_weight if USE_SEMANTIC else 0.0
        self.keyword_weight = 1.0 - self.semantic_weight
        self.semantic_matcher = SemanticSkillMatcher() if USE_SEMANTIC else None
    
    def extract_skills(self, text: str) -> Dict[str, Dict]:
        """Extract skills using hybrid approach."""
        # Keyword-based extraction
        keyword_scores = self._extract_keyword_skills(text)
        
        # Semantic extraction (if available)
        semantic_scores = {}
        if self.semantic_matcher:
            semantic_scores = self.semantic_matcher.extract_skills_semantic(text)
        
        # Combine scores
        combined = {}
        for role in ALL_ROLES:
            kw_score = keyword_scores.get(role, {}).get("score", 0)
            sem_score = semantic_scores.get(role, 0)
            
            combined_score = (
                self.keyword_weight * kw_score +
                self.semantic_weight * sem_score
            )
            
            combined[role] = {
                "score": round(combined_score, 2),
                "keyword_score": kw_score,
                "semantic_score": sem_score,
                "matched_keywords": keyword_scores.get(role, {}).get("matched_keywords", []),
            }
        
        return combined
    
    def _extract_keyword_skills(self, text: str) -> Dict[str, Dict]:
        """Traditional keyword-based skill extraction."""
        cleaned = self._preprocess_text(text)
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
        
        # Normalize scores
        max_possible = max(
            (r["unique_skills"] * 2 + r["raw_count"] * 0.5) for r in role_data.values()
        ) if any(r["raw_count"] > 0 for r in role_data.values()) else 1
        
        for role in role_data:
            r = role_data[role]
            raw_score = r["unique_skills"] * 2 + r["raw_count"] * 0.5
            normalized = min(10, round((raw_score / max(max_possible, 1)) * 10, 1))
            role_data[role]["score"] = normalized
        
        return role_data
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-z0-9\s\.\-\/\#\+]', ' ', text)
        return text.strip()


# ============================================================
# OPTIMIZED TEAM FORMATION (with better balancing)
# ============================================================

class OptimizedTeamFormer:
    """
    Improved team formation with better optimization.
    
    Features:
    - Skill-based matching scores
    - Team balance optimization
    - Role diversity enforcement
    - Fairness constraints
    """
    
    def __init__(self):
        self.skill_extractor = HybridSkillExtractor()
    
    def form_teams(
        self,
        profiles: List[Dict],
        target_team_size: int = 4,
        optimize_for: str = "balance"  # "balance", "skill", "diversity"
    ) -> Tuple[List[Dict], Dict]:
        """
        Form balanced teams with optimization.
        
        Returns:
            Tuple of (teams, metrics)
        """
        if not profiles:
            return [], {}
        
        # Add computed scores to profiles
        for profile in profiles:
            profile["_score"] = self._calculate_overall_score(profile)
        
        # Sort by score for draft
        sorted_profiles = sorted(profiles, key=lambda p: p["_score"], reverse=True)
        
        # Calculate team sizes
        team_sizes = self._calculate_team_sizes(len(profiles), target_team_size)
        num_teams = len(team_sizes)
        
        # Initialize teams
        teams = [
            {
                "team_id": f"Team_{i+1:02d}",
                "team_name": f"Team {i+1}",
                "target_size": size,
                "members": [],
                "roles_filled": set(),
                "total_score": 0.0,
            }
            for i, size in enumerate(team_sizes)
        ]
        
        # Snake draft with role balancing
        assigned = set()
        role_groups = self._group_by_role(sorted_profiles)
        
        max_rounds = max(team_sizes)
        for round_num in range(max_rounds):
            # Snake order
            order = list(range(num_teams)) if round_num % 2 == 0 else list(range(num_teams - 1, -1, -1))
            
            for team_idx in order:
                team = teams[team_idx]
                if len(team["members"]) >= team["target_size"]:
                    continue
                
                # Find best candidate
                candidate = self._find_best_candidate(
                    team, role_groups, assigned, optimize_for
                )
                
                if candidate:
                    self._assign_to_team(team, candidate)
                    assigned.add(candidate["student_id"])
        
        # Calculate final metrics
        for team in teams:
            team["balance_score"] = self._calculate_team_balance(team)
            team["roles_filled"] = list(team["roles_filled"])
        
        metrics = TeamFormationMetrics.generate_quality_report(teams, profiles)
        
        return teams, metrics
    
    def _calculate_overall_score(self, profile: Dict) -> float:
        """Calculate overall strength score."""
        max_skill = max(profile.get("skills", {}).values()) if profile.get("skills") else 0
        exp_score = profile.get("experience_score", 0) / 5.0
        diversity = profile.get("skill_diversity", 0)
        
        return round(0.5 * (max_skill / 10.0) + 0.35 * exp_score + 0.15 * diversity, 4)
    
    def _calculate_team_sizes(self, total: int, target: int) -> List[int]:
        """Calculate optimal team sizes."""
        if total <= 0:
            return []
        
        num_teams = max(1, math.ceil(total / target))
        base = total // num_teams
        remainder = total % num_teams
        
        return [base + 1 if i < remainder else base for i in range(num_teams)]
    
    def _group_by_role(self, profiles: List[Dict]) -> Dict[str, List[Dict]]:
        """Group profiles by primary role."""
        groups = defaultdict(list)
        for p in profiles:
            groups[p.get("primary_role", "backend")].append(p)
        return groups
    
    def _find_best_candidate(
        self,
        team: Dict,
        role_groups: Dict[str, List[Dict]],
        assigned: set,
        optimize_for: str
    ) -> Optional[Dict]:
        """Find the best candidate for a team."""
        role_priority = ["backend", "frontend", "ml", "data", "fullstack", "uiux", "devops"]
        
        # First, try to fill missing roles
        for role in role_priority:
            if role in team["roles_filled"]:
                continue
            
            for candidate in role_groups.get(role, []):
                if candidate["student_id"] not in assigned:
                    return candidate
        
        # Then, take any available
        for role in role_priority:
            for candidate in role_groups.get(role, []):
                if candidate["student_id"] not in assigned:
                    return candidate
        
        return None
    
    def _assign_to_team(self, team: Dict, candidate: Dict):
        """Assign a candidate to a team."""
        member = {
            "student_id": candidate["student_id"],
            "name": candidate["name"],
            "assigned_role": candidate.get("primary_role", "backend"),
            "primary_role": candidate.get("primary_role", "backend"),
            "overall_score": candidate.get("_score", 0),
            "experience_score": candidate.get("experience_score", 0),
            "skill_scores": candidate.get("skills", {}),
            "top_skills": candidate.get("top_skills", [])[:5],
        }
        team["members"].append(member)
        team["roles_filled"].add(candidate.get("primary_role", "backend"))
        team["total_score"] += candidate.get("_score", 0)
    
    def _calculate_team_balance(self, team: Dict) -> Dict:
        """Calculate team balance metrics."""
        members = team.get("members", [])
        if not members:
            return {"average_score": 0, "role_diversity": 0, "overall_balance": 0}
        
        scores = [m.get("overall_score", 0) for m in members]
        avg = sum(scores) / len(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        
        role_diversity = len(team.get("roles_filled", set())) / len(members)
        
        balance = 0.35 * avg + 0.35 * (1 - min(variance * 10, 1)) + 0.30 * role_diversity
        
        return {
            "average_score": round(avg, 3),
            "score_variance": round(variance, 4),
            "role_diversity": round(role_diversity, 2),
            "overall_balance": round(balance, 3),
        }


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SkillSyncAI - Optimized ML Engine")
    print("=" * 60)
    print(f"\nMode: {'Semantic Matching (Advanced)' if USE_SEMANTIC else 'Keyword Matching (Basic)'}")
    
    # Example usage
    extractor = HybridSkillExtractor()
    
    sample_text = """
    Experienced Full Stack Developer with expertise in React, Node.js, and Python.
    Built machine learning models using TensorFlow and scikit-learn.
    Strong background in data analysis with pandas and SQL.
    Completed 3 internships and won 2 hackathons.
    """
    
    print("\n📄 Sample Resume Analysis:")
    skills = extractor.extract_skills(sample_text)
    
    for role, data in sorted(skills.items(), key=lambda x: x[1]["score"], reverse=True):
        if data["score"] > 0:
            print(f"   {ROLE_DISPLAY_NAMES.get(role, role):<12}: {data['score']:.1f}/10")
            if data.get("matched_keywords"):
                print(f"      Keywords: {', '.join(data['matched_keywords'][:5])}")
    
    print("\n✅ Optimized ML Engine ready!")
