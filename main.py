"""
SkillSyncAI - FastAPI Backend
API for team formation based on resume parsing and skill matching.
"""

import os
import json
import asyncio
import httpx
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ml_engine import process_team_formation

# ============================================================
# Keep-Alive Background Task (prevents Render free tier sleep)
# ============================================================

KEEP_ALIVE_URL = os.getenv("RENDER_EXTERNAL_URL", "")  # Set by Render automatically
KEEP_ALIVE_INTERVAL = 14 * 60  # 14 minutes (Render sleeps after 15 mins)


async def keep_alive_task():
    """Ping the health endpoint periodically to keep the server awake."""
    await asyncio.sleep(60)  # Wait 1 minute before starting
    
    while True:
        try:
            if KEEP_ALIVE_URL:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{KEEP_ALIVE_URL}/health", timeout=30)
                    print(f"[Keep-Alive] Pinged {KEEP_ALIVE_URL}/health - Status: {response.status_code}")
            else:
                print("[Keep-Alive] No RENDER_EXTERNAL_URL set, skipping ping")
        except Exception as e:
            print(f"[Keep-Alive] Error: {e}")
        
        await asyncio.sleep(KEEP_ALIVE_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: Start keep-alive task
    keep_alive = asyncio.create_task(keep_alive_task())
    print("[Startup] Keep-alive task started")
    
    yield
    
    # Shutdown: Cancel keep-alive task
    keep_alive.cancel()
    try:
        await keep_alive
    except asyncio.CancelledError:
        pass
    print("[Shutdown] Keep-alive task stopped")


# ============================================================
# FastAPI App Setup
# ============================================================

app = FastAPI(
    title="SkillSyncAI API",
    description="AI-powered team formation based on resume skill extraction",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - adjust origins for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic Models for Request/Response
# ============================================================

class Project(BaseModel):
    name: str
    description: str
    team_size: int = 4


class ProjectsRequest(BaseModel):
    projects: List[Project]


class TeamMember(BaseModel):
    student_id: str
    name: str
    role: str
    overall_score: float
    experience_score: float
    reason: str
    top_skills: List[str]


class BalanceScore(BaseModel):
    average_score: float
    score_variance: float
    role_diversity: float
    overall_balance: float


class Team(BaseModel):
    team_name: str
    team_id: str
    team_size: int
    balance_score: BalanceScore
    members: List[TeamMember]
    roles_covered: List[str]


class StudentProfile(BaseModel):
    student_id: str
    name: str
    primary_role: str
    skills: dict
    experience_score: float
    skill_diversity: float
    top_skills: List[str]


class Summary(BaseModel):
    total_resumes: int
    parsed_resumes: int
    total_teams: int
    students_assigned: int
    average_team_balance: float


class TeamFormationResponse(BaseModel):
    success: bool
    summary: Optional[Summary] = None
    teams: List[Team] = []
    profiles: List[StudentProfile] = []
    error: Optional[str] = None


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "SkillSyncAI API is running",
        "version": "1.0.0",
        "endpoints": {
            "form_teams": "POST /api/form-teams",
            "health": "GET /health",
            "model_info": "GET /api/model-info",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/model-info")
async def get_model_info():
    """
    Get information about the ML model and techniques used.
    Useful for academic presentation and viva.
    """
    return {
        "model_name": "SkillSyncAI Team Formation Model",
        "version": "1.0.0",
        "techniques": {
            "skill_extraction": {
                "method": "Rule-based Keyword Matching",
                "description": "Pattern matching against 200+ technology keywords across 7 role categories",
                "complexity": "O(n × k) where n=text length, k=keywords"
            },
            "experience_scoring": {
                "method": "Weighted Keyword Frequency",
                "description": "Identifies experience indicators (internship, hackathon, project) with weighted scoring",
                "normalization": "Sigmoid-based normalization to 0-5 scale"
            },
            "matching_algorithm": {
                "method": "Weighted Linear Combination",
                "formula": "score = 0.6×skill + 0.3×experience + 0.1×diversity + bonus",
                "academic_basis": "Multi-criteria decision making (MCDM)"
            },
            "team_formation": {
                "method": "Snake Draft with Role Balancing",
                "description": "Alternating pick order ensures fair distribution of talent",
                "constraints": ["Role diversity", "Skill balance", "Team size equality"]
            },
            "explainability": {
                "method": "Rule-based Explanation Generation",
                "description": "Each assignment includes human-readable reasoning (XAI)"
            }
        },
        "metrics": {
            "team_balance": "Gini coefficient for strength distribution",
            "role_coverage": "Percentage of required roles filled per team",
            "assignment_rate": "Percentage of students successfully assigned"
        },
        "future_improvements": [
            "Semantic embeddings using Sentence Transformers (BERT)",
            "Hungarian Algorithm for optimal assignment",
            "Feedback loop for model refinement",
            "GitHub/LinkedIn profile integration"
        ]
    }


@app.post("/api/form-teams", response_model=TeamFormationResponse)
async def form_teams(
    resumes: List[UploadFile] = File(..., description="Student resume PDF files"),
    projects_json: Optional[str] = Form(None, description="Projects JSON string"),
    team_size: int = Form(4, description="Target team size"),
):
    """
    Form teams from uploaded resumes and project requirements.
    
    **Request:**
    - `resumes`: Multiple PDF files (student resumes)
    - `projects_json`: JSON string containing project requirements (optional)
    - `team_size`: Target team size (default: 4)
    
    **projects_json format:**
    ```json
    [
        {
            "name": "Project Name",
            "description": "We need React, Node.js, and ML skills...",
            "team_size": 4
        }
    ]
    ```
    
    **Response:**
    - `teams`: List of formed teams with members and explanations
    - `profiles`: All parsed student profiles
    - `summary`: Statistics about the team formation
    """
    try:
        # Parse projects JSON
        projects = []
        if projects_json:
            try:
                projects_data = json.loads(projects_json)
                if isinstance(projects_data, list):
                    projects = projects_data
                elif isinstance(projects_data, dict) and "projects" in projects_data:
                    projects = projects_data["projects"]
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid projects JSON: {str(e)}")
        
        # Read all resume files
        resume_data = []
        for resume_file in resumes:
            if not resume_file.filename.lower().endswith('.pdf'):
                continue  # Skip non-PDF files
            
            content = await resume_file.read()
            resume_data.append({
                "filename": resume_file.filename,
                "content": content,
            })
        
        if not resume_data:
            raise HTTPException(status_code=400, detail="No valid PDF resumes provided")
        
        # Process team formation
        result = process_team_formation(
            resumes=resume_data,
            projects=projects,
            target_team_size=team_size,
        )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        return TeamFormationResponse(
            success=False,
            error=str(e),
            teams=[],
            profiles=[],
        )


@app.post("/api/form-teams-json")
async def form_teams_json(
    resumes: List[UploadFile] = File(...),
    projects_file: Optional[UploadFile] = File(None, description="projects.json file"),
    team_size: int = Form(4),
):
    """
    Alternative endpoint that accepts projects as a JSON file upload.
    
    **Request:**
    - `resumes`: Multiple PDF files (student resumes)
    - `projects_file`: Optional projects.json file
    - `team_size`: Target team size (default: 4)
    """
    try:
        # Parse projects file
        projects = []
        if projects_file:
            try:
                projects_content = await projects_file.read()
                projects_data = json.loads(projects_content.decode('utf-8'))
                if isinstance(projects_data, list):
                    projects = projects_data
                elif isinstance(projects_data, dict) and "projects" in projects_data:
                    projects = projects_data["projects"]
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid projects file: {str(e)}")
        
        # Read all resume files
        resume_data = []
        for resume_file in resumes:
            if not resume_file.filename.lower().endswith('.pdf'):
                continue
            
            content = await resume_file.read()
            resume_data.append({
                "filename": resume_file.filename,
                "content": content,
            })
        
        if not resume_data:
            raise HTTPException(status_code=400, detail="No valid PDF resumes provided")
        
        # Process team formation
        result = process_team_formation(
            resumes=resume_data,
            projects=projects,
            target_team_size=team_size,
        )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "teams": [],
            "profiles": [],
        }


@app.post("/api/parse-resumes")
async def parse_resumes(
    resumes: List[UploadFile] = File(..., description="Student resume PDF files"),
):
    """
    Parse resumes and return student profiles without team formation.
    Useful for previewing extracted data before forming teams.
    """
    try:
        resume_data = []
        for resume_file in resumes:
            if not resume_file.filename.lower().endswith('.pdf'):
                continue
            
            content = await resume_file.read()
            resume_data.append({
                "filename": resume_file.filename,
                "content": content,
            })
        
        if not resume_data:
            raise HTTPException(status_code=400, detail="No valid PDF resumes provided")
        
        # Process without team formation - we'll pass empty projects
        result = process_team_formation(
            resumes=resume_data,
            projects=[],
            target_team_size=4,
        )
        
        return {
            "success": True,
            "profiles": result.get("profiles", []),
            "total_parsed": len(result.get("profiles", [])),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "profiles": [],
        }


# ============================================================
# Run with: uvicorn main:app --reload
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
