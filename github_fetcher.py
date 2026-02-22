"""
SkillSyncAI - GitHub Profile Fetcher
Fetches GitHub data and extracts skills from user profiles.
"""

import os
import httpx
from typing import Dict, List, Optional
from collections import defaultdict
import asyncio

# GitHub Personal Access Token (for higher rate limits: 5000 req/hr vs 60 req/hr)
# Set via environment variable: GITHUB_TOKEN
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# ============================================================
# GITHUB LANGUAGE TO SKILL MAPPING
# ============================================================

GITHUB_LANGUAGE_MAPPING = {
    # Frontend
    "javascript": "frontend",
    "typescript": "frontend",
    "html": "frontend",
    "css": "frontend",
    "vue": "frontend",
    "scss": "frontend",
    "sass": "frontend",
    
    # Backend
    "python": "backend",
    "java": "backend",
    "go": "backend",
    "rust": "backend",
    "ruby": "backend",
    "php": "backend",
    "c#": "backend",
    "kotlin": "backend",
    "scala": "backend",
    
    # Data/ML
    "jupyter notebook": "data",
    "r": "data",
    
    # DevOps
    "shell": "devops",
    "dockerfile": "devops",
    "hcl": "devops",  # Terraform
}

# Topic keywords that indicate specific roles
GITHUB_TOPIC_MAPPING = {
    # Frontend
    "react": "frontend",
    "angular": "frontend",
    "vue": "frontend",
    "nextjs": "frontend",
    "frontend": "frontend",
    "tailwindcss": "frontend",
    "redux": "frontend",
    
    # Backend
    "nodejs": "backend",
    "express": "backend",
    "django": "backend",
    "flask": "backend",
    "fastapi": "backend",
    "spring-boot": "backend",
    "api": "backend",
    "rest-api": "backend",
    "graphql": "backend",
    "mongodb": "backend",
    "postgresql": "backend",
    "mysql": "backend",
    
    # ML/AI
    "machine-learning": "ml",
    "deep-learning": "ml",
    "tensorflow": "ml",
    "pytorch": "ml",
    "nlp": "ml",
    "computer-vision": "ml",
    "neural-network": "ml",
    "artificial-intelligence": "ml",
    "ai": "ml",
    "data-science": "ml",
    "scikit-learn": "ml",
    
    # Data
    "data-analysis": "data",
    "data-visualization": "data",
    "pandas": "data",
    "jupyter": "data",
    "analytics": "data",
    "big-data": "data",
    "etl": "data",
    
    # DevOps
    "docker": "devops",
    "kubernetes": "devops",
    "aws": "devops",
    "devops": "devops",
    "ci-cd": "devops",
    "terraform": "devops",
    "ansible": "devops",
    
    # UI/UX (rare in GitHub but possible)
    "figma": "uiux",
    "design": "uiux",
    "ui": "uiux",
    "ux": "uiux",
}


class GitHubFetcher:
    """Fetches and analyzes GitHub profiles for skill extraction."""
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the GitHub fetcher.
        
        Args:
            token: Optional GitHub personal access token for higher rate limits.
                   Without token: 60 requests/hour
                   With token: 5000 requests/hour
        """
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SkillSyncAI-Bot"
        }
        self.authenticated = bool(token)
        if token:
            self.headers["Authorization"] = f"token {token}"
    
    async def check_rate_limit(self) -> Dict:
        """Check current GitHub API rate limit status."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.BASE_URL}/rate_limit",
                    headers=self.headers
                )
                if response.status_code == 200:
                    data = response.json()
                    core = data.get("resources", {}).get("core", {})
                    return {
                        "limit": core.get("limit", 60),
                        "remaining": core.get("remaining", 0),
                        "reset_at": core.get("reset", 0),
                        "authenticated": self.authenticated,
                    }
        except Exception as e:
            return {"error": str(e), "authenticated": self.authenticated}
        return {"error": "Unknown", "authenticated": self.authenticated}
    
    def extract_username(self, github_input: str) -> str:
        """
        Extract GitHub username from various input formats.
        
        Handles:
        - Full URL: https://github.com/username
        - Username only: username
        - URL with trailing slash: https://github.com/username/
        """
        github_input = github_input.strip()
        
        # Remove trailing slash
        if github_input.endswith("/"):
            github_input = github_input[:-1]
        
        # Extract from URL
        if "github.com/" in github_input:
            parts = github_input.split("github.com/")
            if len(parts) > 1:
                username = parts[1].split("/")[0]
                return username
        
        # Already a username
        return github_input
    
    async def fetch_user_profile(self, username: str) -> Optional[Dict]:
        """Fetch basic user profile information."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.BASE_URL}/users/{username}",
                    headers=self.headers
                )
                if response.status_code == 200:
                    return response.json()
                return None
        except Exception as e:
            print(f"Error fetching profile for {username}: {e}")
            return None
    
    async def fetch_user_repos(self, username: str, max_repos: int = 30) -> List[Dict]:
        """Fetch user's public repositories."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{self.BASE_URL}/users/{username}/repos",
                    params={
                        "sort": "updated",
                        "direction": "desc",
                        "per_page": min(max_repos, 100)
                    },
                    headers=self.headers
                )
                if response.status_code == 200:
                    return response.json()
                return []
        except Exception as e:
            print(f"Error fetching repos for {username}: {e}")
            return []
    
    async def fetch_repo_languages(self, username: str, repo_name: str) -> Dict[str, int]:
        """Fetch languages used in a specific repository."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.BASE_URL}/repos/{username}/{repo_name}/languages",
                    headers=self.headers
                )
                if response.status_code == 200:
                    return response.json()
                return {}
        except Exception:
            return {}
    
    def analyze_repos(self, repos: List[Dict]) -> Dict:
        """
        Analyze repositories to extract skills and activity metrics.
        
        Returns:
            Dict with:
            - languages: Dict of language -> bytes of code
            - topics: List of all topics across repos
            - repo_count: Number of repos
            - total_stars: Sum of stars
            - has_contributions: Whether user contributes actively
        """
        languages = defaultdict(int)
        topics = []
        total_stars = 0
        forked_count = 0
        
        for repo in repos:
            # Skip forked repos for skill analysis (focus on original work)
            if repo.get("fork", False):
                forked_count += 1
                continue
            
            # Collect language
            lang = repo.get("language")
            if lang:
                languages[lang.lower()] += repo.get("size", 0)
            
            # Collect topics
            repo_topics = repo.get("topics", [])
            topics.extend(repo_topics)
            
            # Count stars
            total_stars += repo.get("stargazers_count", 0)
        
        return {
            "languages": dict(languages),
            "topics": list(set(topics)),  # Unique topics
            "repo_count": len(repos) - forked_count,
            "total_stars": total_stars,
            "original_repos": len(repos) - forked_count,
        }
    
    def extract_github_skills(self, analysis: Dict) -> Dict[str, Dict]:
        """
        Convert GitHub analysis to skill scores per role.
        
        Returns same format as resume skill extraction:
        {
            "frontend": {"score": 7.5, "matched_keywords": [...], ...},
            "backend": {"score": 8.0, "matched_keywords": [...], ...},
            ...
        }
        """
        role_scores = {
            "frontend": {"score": 0, "matched_keywords": [], "source": "github"},
            "backend": {"score": 0, "matched_keywords": [], "source": "github"},
            "fullstack": {"score": 0, "matched_keywords": [], "source": "github"},
            "ml": {"score": 0, "matched_keywords": [], "source": "github"},
            "data": {"score": 0, "matched_keywords": [], "source": "github"},
            "uiux": {"score": 0, "matched_keywords": [], "source": "github"},
            "devops": {"score": 0, "matched_keywords": [], "source": "github"},
        }
        
        # Score from languages
        total_bytes = sum(analysis["languages"].values()) or 1
        for lang, bytes_count in analysis["languages"].items():
            role = GITHUB_LANGUAGE_MAPPING.get(lang.lower())
            if role and role in role_scores:
                # Weight by proportion of code
                weight = min(3, (bytes_count / total_bytes) * 5)
                role_scores[role]["score"] += weight
                role_scores[role]["matched_keywords"].append(f"lang:{lang}")
        
        # Score from topics (stronger signal)
        for topic in analysis["topics"]:
            role = GITHUB_TOPIC_MAPPING.get(topic.lower())
            if role and role in role_scores:
                role_scores[role]["score"] += 1.5
                role_scores[role]["matched_keywords"].append(f"topic:{topic}")
        
        # Bonus for repo count (shows activity)
        activity_bonus = min(2, analysis["repo_count"] / 10)
        
        # Normalize scores to 0-10 scale
        max_score = max((r["score"] for r in role_scores.values()), default=1) or 1
        for role in role_scores:
            raw = role_scores[role]["score"]
            if raw > 0:
                role_scores[role]["score"] = round(min(10, (raw / max_score) * 8 + activity_bonus), 1)
            role_scores[role]["unique_skills"] = len(role_scores[role]["matched_keywords"])
        
        return role_scores
    
    async def analyze_github_profile(self, github_input: str) -> Optional[Dict]:
        """
        Full analysis of a GitHub profile.
        
        Args:
            github_input: GitHub username or profile URL
            
        Returns:
            Dict with:
            - username: GitHub username
            - profile: Basic profile info
            - skills: Role-based skill scores
            - metrics: Activity metrics
        """
        username = self.extract_username(github_input)
        
        if not username:
            return None
        
        # Fetch repos (profile is optional, repos are essential)
        repos = await self.fetch_user_repos(username)
        
        if not repos:
            return {
                "username": username,
                "profile": None,
                "skills": {},
                "metrics": {"repo_count": 0, "error": "No public repos found or user doesn't exist"},
            }
        
        # Analyze repos
        analysis = self.analyze_repos(repos)
        
        # Extract skills
        skills = self.extract_github_skills(analysis)
        
        return {
            "username": username,
            "skills": skills,
            "metrics": {
                "repo_count": analysis["repo_count"],
                "total_stars": analysis["total_stars"],
                "top_languages": list(analysis["languages"].keys())[:5],
                "topics": analysis["topics"][:10],
            }
        }


async def fetch_multiple_github_profiles(
    github_mapping: Dict[str, str],
    token: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Fetch GitHub data for multiple candidates.
    
    Args:
        github_mapping: Dict mapping candidate names to GitHub usernames/URLs
        token: Optional GitHub API token. If not provided, uses GITHUB_TOKEN env var.
        
    Returns:
        Dict mapping candidate names to their GitHub analysis
    """
    # Use provided token, or fall back to environment variable
    auth_token = token or GITHUB_TOKEN
    fetcher = GitHubFetcher(token=auth_token)
    results = {}
    
    # Process in batches to avoid rate limiting
    batch_size = 5
    names = list(github_mapping.keys())
    
    for i in range(0, len(names), batch_size):
        batch_names = names[i:i + batch_size]
        tasks = [
            fetcher.analyze_github_profile(github_mapping[name])
            for name in batch_names
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for name, result in zip(batch_names, batch_results):
            if isinstance(result, Exception):
                results[name] = {"error": str(result)}
            else:
                results[name] = result
    
    return results


def merge_skills_with_github(
    resume_skills: Dict[str, Dict],
    github_skills: Dict[str, Dict],
    resume_weight: float = 0.6,
    github_weight: float = 0.4
) -> Dict[str, Dict]:
    """
    Merge resume-extracted skills with GitHub-extracted skills.
    
    Args:
        resume_skills: Skills extracted from resume
        github_skills: Skills extracted from GitHub
        resume_weight: Weight for resume skills (default 0.6)
        github_weight: Weight for GitHub skills (default 0.4)
        
    Returns:
        Merged skill scores
    """
    merged = {}
    all_roles = set(resume_skills.keys()) | set(github_skills.keys())
    
    for role in all_roles:
        resume_data = resume_skills.get(role, {"score": 0, "matched_keywords": []})
        github_data = github_skills.get(role, {"score": 0, "matched_keywords": []})
        
        # Weighted average of scores
        combined_score = (
            resume_data.get("score", 0) * resume_weight +
            github_data.get("score", 0) * github_weight
        )
        
        # Merge keywords
        resume_keywords = resume_data.get("matched_keywords", [])
        github_keywords = github_data.get("matched_keywords", [])
        
        # Handle both list and string formats for keywords
        if isinstance(resume_keywords, list):
            all_keywords = [f"resume:{k}" if isinstance(k, str) else f"resume:{k}" for k in resume_keywords[:5]]
        else:
            all_keywords = []
        
        if isinstance(github_keywords, list):
            all_keywords.extend(github_keywords[:5])
        
        merged[role] = {
            "score": round(combined_score, 1),
            "matched_keywords": all_keywords,
            "unique_skills": len(set(all_keywords)),
            "resume_score": resume_data.get("score", 0),
            "github_score": github_data.get("score", 0),
            "raw_count": resume_data.get("raw_count", 0),
        }
    
    return merged


async def get_github_rate_limit_info() -> Dict:
    """
    Get current GitHub API rate limit info.
    
    Returns:
        Dict with rate limit status and authentication info
    """
    fetcher = GitHubFetcher(token=GITHUB_TOKEN)
    rate_info = await fetcher.check_rate_limit()
    return {
        "authenticated": bool(GITHUB_TOKEN),
        "rate_limit": rate_info.get("limit", 60),
        "remaining": rate_info.get("remaining", "unknown"),
        "message": "Using authenticated token (5000 req/hr)" if GITHUB_TOKEN else "No token - limited to 60 req/hr"
    }


def is_github_authenticated() -> bool:
    """Check if GitHub token is configured."""
    return bool(GITHUB_TOKEN)
