"""
SkillSyncAI - FastAPI Backend
API for team formation based on resume parsing and skill matching.
"""

import os
import json
import asyncio
import httpx
import zipfile
import io
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ml_engine import process_team_formation
from github_fetcher import (
    GitHubFetcher, 
    fetch_multiple_github_profiles, 
    get_github_rate_limit_info,
    is_github_authenticated
)

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
    participant_id: Optional[str] = None  # NEW: Original participant ID
    skill_percentages: Optional[dict] = None  # NEW: Individual skill percentages


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
    project_id: Optional[str] = None  # NEW: Assigned project ID
    project_name: Optional[str] = None  # NEW: Assigned project name


class StudentProfile(BaseModel):
    student_id: str
    name: str
    primary_role: str
    skills: dict
    experience_score: float
    skill_diversity: float
    top_skills: List[str]
    participant_id: Optional[str] = None  # NEW: Original participant ID
    github_profile: Optional[str] = None  # NEW: GitHub username
    skill_percentages: Optional[dict] = None  # NEW: Individual skill percentages


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
            "form_teams_v2": "POST /api/v2/form-teams (NEW - Structured payload)",
            "form_teams_zip": "POST /api/form-teams-zip",
            "health": "GET /health",
            "model_info": "GET /api/model-info",
            "github_status": "GET /api/github-status",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/github-status")
async def github_status():
    """
    Check GitHub API authentication status and rate limits.
    
    **Response:**
    - `authenticated`: Whether a GitHub token is configured
    - `rate_limit`: Maximum requests per hour (60 without token, 5000 with token)
    - `remaining`: Remaining requests in current window
    - `message`: Human-readable status message
    """
    return await get_github_rate_limit_info()


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
            },
            "github_integration": {
                "method": "GitHub API Profile Analysis",
                "description": "Extracts skills from repository languages and topics",
                "data_sources": ["Repository languages", "Repository topics", "Stars count", "Activity level"],
                "formula": "final_skill = 0.6×resume_skill + 0.4×github_skill"
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
            "LinkedIn profile integration"
        ]
    }


@app.post("/api/form-teams", response_model=TeamFormationResponse)
async def form_teams(
    resumes: List[UploadFile] = File(..., description="Student resume PDF files"),
    projects_json: Optional[str] = Form(None, description="Projects JSON string"),
    team_size: int = Form(4, description="Target team size"),
    github_usernames: Optional[str] = Form(None, description="JSON mapping of candidate names to GitHub usernames"),
):
    """
    Form teams from uploaded resumes and project requirements.
    
    **Request:**
    - `resumes`: Multiple PDF files (student resumes)
    - `projects_json`: JSON string containing project requirements (optional)
    - `team_size`: Target team size (default: 4)
    - `github_usernames`: JSON mapping of candidate names to GitHub usernames (optional)
    
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
    
    **github_usernames format:**
    ```json
    {
        "John Doe": "johndoe",
        "Jane Smith": "janesmith123"
    }
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
        
        # Fetch GitHub data if usernames provided
        github_data = None
        if github_usernames:
            try:
                github_mapping = json.loads(github_usernames)
                if isinstance(github_mapping, dict) and github_mapping:
                    github_data = await fetch_multiple_github_profiles(github_mapping)
            except json.JSONDecodeError as e:
                # Log but don't fail - GitHub data is optional
                print(f"Warning: Invalid github_usernames JSON: {e}")
        
        # Process team formation
        result = process_team_formation(
            resumes=resume_data,
            projects=projects,
            target_team_size=team_size,
            github_data=github_data,
        )
        
        # Add GitHub info to result if available
        if github_data:
            result["github_info"] = {
                "profiles_fetched": len(github_data),
                "profiles": {name: data.get("metrics", {}) for name, data in github_data.items()}
            }
        
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


@app.post("/api/form-teams-zip", response_model=TeamFormationResponse)
async def form_teams_from_zip(
    resumes_zip: UploadFile = File(..., description="ZIP file containing PDF resumes"),
    projects_json: Optional[str] = Form(None, description="Projects JSON string"),
    team_size: int = Form(4, description="Target team size"),
    github_usernames: Optional[str] = Form(None, description="JSON mapping of candidate names to GitHub usernames"),
):
    """
    Form teams from a ZIP file containing PDF resumes.
    
    **Request:**
    - `resumes_zip`: A single ZIP file containing multiple PDF resumes
    - `projects_json`: JSON string containing project requirements (optional)
    - `team_size`: Target team size (default: 4)
    - `github_usernames`: JSON mapping of candidate names to GitHub usernames (optional)
    
    **Example:**
    Upload a file like `resumes.zip` containing:
    - john_doe.pdf
    - jane_smith.pdf
    - bob_wilson.pdf
    
    **github_usernames format:**
    ```json
    {
        "John Doe": "johndoe",
        "Jane Smith": "janesmith123"
    }
    ```
    
    **Response:**
    - `teams`: List of formed teams with members and explanations
    - `profiles`: All parsed student profiles
    - `summary`: Statistics about the team formation
    """
    try:
        # Validate ZIP file
        if not resumes_zip.filename.lower().endswith('.zip'):
            raise HTTPException(status_code=400, detail="Please upload a .zip file")
        
        # Read ZIP content
        zip_content = await resumes_zip.read()
        
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
        
        # Extract PDFs from ZIP
        resume_data = []
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    # Skip directories and non-PDF files
                    if file_info.is_dir():
                        continue
                    
                    filename = file_info.filename
                    # Handle nested folders - get just the filename
                    if '/' in filename:
                        filename = filename.split('/')[-1]
                    if '\\' in filename:
                        filename = filename.split('\\')[-1]
                    
                    # Skip hidden files and non-PDFs
                    if filename.startswith('.') or filename.startswith('__'):
                        continue
                    if not filename.lower().endswith('.pdf'):
                        continue
                    
                    # Read PDF content
                    pdf_content = zip_ref.read(file_info.filename)
                    resume_data.append({
                        "filename": filename,
                        "content": pdf_content,
                    })
                    
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid or corrupted ZIP file")
        
        if not resume_data:
            raise HTTPException(
                status_code=400, 
                detail="No valid PDF files found in the ZIP archive"
            )
        
        # Fetch GitHub data if usernames provided
        github_data = None
        if github_usernames:
            try:
                github_mapping = json.loads(github_usernames)
                if isinstance(github_mapping, dict) and github_mapping:
                    github_data = await fetch_multiple_github_profiles(github_mapping)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid github_usernames JSON: {e}")
        
        # Process team formation
        result = process_team_formation(
            resumes=resume_data,
            projects=projects,
            target_team_size=team_size,
            github_data=github_data,
        )
        
        # Add info about extracted files
        result["zip_info"] = {
            "original_filename": resumes_zip.filename,
            "pdfs_extracted": len(resume_data),
            "pdf_files": [r["filename"] for r in resume_data],
        }
        
        # Add GitHub info if available
        if github_data:
            result["github_info"] = {
                "profiles_fetched": len(github_data),
                "profiles": {name: data.get("metrics", {}) for name, data in github_data.items()}
            }
        
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
# NEW V2 ENDPOINT - Structured Payload Format
# ============================================================

@app.post("/api/v2/form-teams")
async def form_teams_v2(
    resumes: List[UploadFile] = File(..., description="Participant resume PDFs (order must match participantData)"),
    projects: str = Form(..., description="JSON array of projects"),
    participantData: str = Form(..., description="JSON array of participant info"),
):
    """
    Form teams with structured participant and project data.
    
    **Request Format:**
    - `resumes`: PDF files (ORDER MUST MATCH participantData array)
    - `projects`: JSON string
    - `participantData`: JSON string
    
    **projects format:**
    ```json
    [
        {
            "projectId": "P001",
            "projectName": "AI Chatbot",
            "description": "Build an AI-powered chatbot using NLP",
            "techstack": "Python, TensorFlow, React, Node.js"
        }
    ]
    ```
    
    **participantData format:**
    ```json
    [
        {
            "participantId": "PART001",
            "participantName": "John Doe",
            "githubProfile": "johndoe123"
        },
        {
            "participantId": "PART002",
            "participantName": "Jane Smith",
            "githubProfile": "janesmith"
        }
    ]
    ```
    
    **Important:** The order of resumes must match the order of participantData.
    - resumes[0] belongs to participantData[0]
    - resumes[1] belongs to participantData[1]
    - etc.
    
    **Response:** Teams with original participantId, participantName, and projectId preserved.
    """
    try:
        # Parse projects JSON
        try:
            projects_list = json.loads(projects)
            if not isinstance(projects_list, list):
                raise HTTPException(status_code=400, detail="projects must be a JSON array")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid projects JSON: {str(e)}")
        
        # Parse participantData JSON
        try:
            participants_list = json.loads(participantData)
            if not isinstance(participants_list, list):
                raise HTTPException(status_code=400, detail="participantData must be a JSON array")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid participantData JSON: {str(e)}")
        
        # Validate resume count matches participant count
        pdf_resumes = [r for r in resumes if r.filename.lower().endswith('.pdf')]
        if len(pdf_resumes) != len(participants_list):
            raise HTTPException(
                status_code=400, 
                detail=f"Resume count ({len(pdf_resumes)}) must match participantData count ({len(participants_list)})"
            )
        
        # Build resume data with participant info (order-matched)
        resume_data = []
        participant_map = {}  # Map student_id -> participant info
        github_mapping = {}   # Map participant name -> github username
        
        for idx, (resume_file, participant) in enumerate(zip(pdf_resumes, participants_list)):
            content = await resume_file.read()
            
            participant_id = participant.get("participantId", f"P{idx+1:03d}")
            participant_name = participant.get("participantName", f"Participant_{idx+1}")
            github_profile = participant.get("githubProfile", "")
            
            # Use participant name as the student identifier
            student_id = f"S{idx+1:03d}"
            
            resume_data.append({
                "filename": resume_file.filename,
                "content": content,
                "student_id": student_id,
                "name": participant_name,
            })
            
            # Store mapping for later
            participant_map[student_id] = {
                "participant_id": participant_id,
                "participant_name": participant_name,
                "github_profile": github_profile,
            }
            
            # Build GitHub mapping
            if github_profile:
                github_mapping[participant_name] = github_profile
        
        # Fetch GitHub data if any profiles provided
        github_data = None
        if github_mapping:
            try:
                github_data = await fetch_multiple_github_profiles(github_mapping)
            except Exception as e:
                print(f"Warning: GitHub fetch failed: {e}")
        
        # Convert projects to internal format
        internal_projects = []
        project_map = {}  # Map project index -> project info
        
        for idx, proj in enumerate(projects_list):
            project_id = proj.get("projectId", f"PROJ{idx+1:03d}")
            project_name = proj.get("projectName", f"Project {idx+1}")
            description = proj.get("description", "")
            techstack = proj.get("techstack", "")
            
            # Combine description and techstack for requirements
            requirements = f"{description}. Tech stack: {techstack}" if techstack else description
            
            internal_projects.append({
                "name": project_name,
                "description": requirements,
                "team_size": max(1, len(participants_list) // max(len(projects_list), 1)),
            })
            
            project_map[idx] = {
                "project_id": project_id,
                "project_name": project_name,
            }
        
        # Calculate team size based on participants and projects
        if internal_projects:
            target_team_size = max(2, len(participants_list) // len(internal_projects))
        else:
            target_team_size = 4
        
        # Process team formation
        result = process_team_formation(
            resumes=resume_data,
            projects=internal_projects,
            target_team_size=target_team_size,
            github_data=github_data,
        )
        
        # Enrich response with original IDs
        enriched_teams = []
        for idx, team in enumerate(result.get("teams", [])):
            # Add project info
            proj_info = project_map.get(idx, {})
            
            enriched_members = []
            for member in team.get("members", []):
                student_id = member.get("student_id", "")
                part_info = participant_map.get(student_id, {})
                
                enriched_member = {
                    **member,
                    "participant_id": part_info.get("participant_id", student_id),
                    "participant_name": part_info.get("participant_name", member.get("name", "")),
                    "github_profile": part_info.get("github_profile", ""),
                }
                enriched_members.append(enriched_member)
            
            enriched_team = {
                **team,
                "members": enriched_members,
                "project_id": proj_info.get("project_id"),
                "project_name": proj_info.get("project_name"),
            }
            enriched_teams.append(enriched_team)
        
        # Enrich profiles with participant info
        enriched_profiles = []
        for profile in result.get("profiles", []):
            student_id = profile.get("student_id", "")
            part_info = participant_map.get(student_id, {})
            
            enriched_profile = {
                **profile,
                "participant_id": part_info.get("participant_id", student_id),
                "participant_name": part_info.get("participant_name", profile.get("name", "")),
                "github_profile": part_info.get("github_profile", ""),
            }
            enriched_profiles.append(enriched_profile)
        
        return {
            "success": True,
            "summary": result.get("summary"),
            "teams": enriched_teams,
            "profiles": enriched_profiles,
            "projects_received": [
                {"project_id": p.get("projectId"), "project_name": p.get("projectName")}
                for p in projects_list
            ],
            "participants_received": len(participants_list),
            "github_profiles_fetched": len(github_data) if github_data else 0,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "teams": [],
            "profiles": [],
        }


# ============================================================
# Run with: uvicorn main:app --reload
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
