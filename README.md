# SkillSyncAI API

FastAPI backend for the SkillSyncAI team formation system.

## Quick Start

### 1. Install Dependencies

```bash
cd api
pip install -r requirements.txt
```

### 2. Run the Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Access the API

- **API Base URL**: `http://localhost:8000`
- **Interactive Docs (Swagger)**: `http://localhost:8000/docs`
- **Alternative Docs (ReDoc)**: `http://localhost:8000/redoc`

---

## API Endpoints

### 1. Form Teams - `POST /api/form-teams`

Main endpoint for team formation.

**Request (multipart/form-data):**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `resumes` | File[] | Yes | Multiple PDF resume files |
| `projects_json` | String | No | JSON string with project requirements |
| `team_size` | Integer | No | Target team size (default: 4) |

**Example using cURL:**

```bash
curl -X POST "http://localhost:8000/api/form-teams" \
  -F "resumes=@john_doe_resume.pdf" \
  -F "resumes=@jane_smith_resume.pdf" \
  -F "resumes=@bob_wilson_resume.pdf" \
  -F "team_size=4" \
  -F 'projects_json=[{"name": "Web App", "description": "React frontend with Node.js backend", "team_size": 4}]'
```

**Example using JavaScript (Frontend):**

```javascript
async function formTeams(resumeFiles, projects, teamSize = 4) {
  const formData = new FormData();
  
  // Add resume files
  resumeFiles.forEach(file => {
    formData.append('resumes', file);
  });
  
  // Add projects JSON
  formData.append('projects_json', JSON.stringify(projects));
  formData.append('team_size', teamSize);
  
  const response = await fetch('http://localhost:8000/api/form-teams', {
    method: 'POST',
    body: formData,
  });
  
  return await response.json();
}

// Usage
const projects = [
  {
    name: "Smart Campus App",
    description: "React frontend, Node.js backend, ML model for recommendations",
    team_size: 4
  }
];

const result = await formTeams(selectedFiles, projects, 4);
console.log(result.teams);
```

---

### 2. Form Teams (File Upload) - `POST /api/form-teams-json`

Alternative endpoint that accepts `projects.json` as a file upload.

**Request (multipart/form-data):**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `resumes` | File[] | Yes | Multiple PDF resume files |
| `projects_file` | File | No | projects.json file |
| `team_size` | Integer | No | Target team size (default: 4) |

**Example:**

```bash
curl -X POST "http://localhost:8000/api/form-teams-json" \
  -F "resumes=@resume1.pdf" \
  -F "resumes=@resume2.pdf" \
  -F "projects_file=@projects.json" \
  -F "team_size=4"
```

---

### 3. Parse Resumes Only - `POST /api/parse-resumes`

Parse resumes and get student profiles without team formation.

```bash
curl -X POST "http://localhost:8000/api/parse-resumes" \
  -F "resumes=@resume1.pdf" \
  -F "resumes=@resume2.pdf"
```

---

## Request/Response Formats

### Projects JSON Format

```json
[
  {
    "name": "Project Name",
    "description": "We need React frontend, Python Flask backend, and an ML model for predictions. UI/UX design and Docker deployment required.",
    "team_size": 4
  },
  {
    "name": "Another Project",
    "description": "Full-stack e-commerce with payment integration...",
    "team_size": 5
  }
]
```

### Response Format

```json
{
  "success": true,
  "summary": {
    "total_resumes": 20,
    "parsed_resumes": 20,
    "total_teams": 5,
    "students_assigned": 20,
    "average_team_balance": 0.723
  },
  "teams": [
    {
      "team_name": "Team 1",
      "team_id": "Team_01",
      "team_size": 4,
      "balance_score": {
        "average_score": 0.625,
        "score_variance": 0.0024,
        "role_diversity": 1.0,
        "overall_balance": 0.752
      },
      "members": [
        {
          "student_id": "S001",
          "name": "John Doe",
          "role": "Backend",
          "overall_score": 0.72,
          "experience_score": 3.5,
          "reason": "Strong Backend skills (score: 8.5/10); Primary expertise aligns with Backend role; Key skills: node.js, express, mongodb, python",
          "top_skills": ["node.js", "express", "mongodb"]
        }
      ],
      "roles_covered": ["Backend", "Frontend", "ML/AI", "DevOps"]
    }
  ],
  "profiles": [
    {
      "student_id": "S001",
      "name": "John Doe",
      "primary_role": "Backend",
      "skills": {
        "frontend": 3.2,
        "backend": 8.5,
        "fullstack": 2.0,
        "ml": 1.5,
        "data": 2.3,
        "uiux": 0.5,
        "devops": 4.2
      },
      "experience_score": 3.5,
      "skill_diversity": 0.86,
      "top_skills": ["node.js", "express", "mongodb", "docker"]
    }
  ]
}
```

---

## Frontend Integration Guide

### React Example

```jsx
import React, { useState } from 'react';

function TeamFormation() {
  const [resumes, setResumes] = useState([]);
  const [projects, setProjects] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const formData = new FormData();
    
    resumes.forEach(file => {
      formData.append('resumes', file);
    });
    
    formData.append('projects_json', JSON.stringify(projects));
    formData.append('team_size', 4);

    try {
      const response = await fetch('http://localhost:8000/api/form-teams', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          multiple
          accept=".pdf"
          onChange={(e) => setResumes([...e.target.files])}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Form Teams'}
        </button>
      </form>

      {result && (
        <div>
          <h2>Results</h2>
          <p>Teams Formed: {result.summary?.total_teams}</p>
          {result.teams.map(team => (
            <div key={team.team_id}>
              <h3>{team.team_name}</h3>
              <ul>
                {team.members.map(member => (
                  <li key={member.student_id}>
                    {member.name} - {member.role}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

---

## Deployment Options

### 1. Local Development
```bash
uvicorn main:app --reload --port 8000
```

### 2. Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t skillsyncai-api .
docker run -p 8000:8000 skillsyncai-api
```

### 3. Production (Gunicorn)
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 4. Cloud Platforms
- **Railway**: Connect GitHub repo, auto-deploy
- **Render**: Free tier available, easy setup
- **AWS/GCP/Azure**: Use App Service or Cloud Run
- **Heroku**: Add `Procfile` with `web: uvicorn main:app --host 0.0.0.0 --port $PORT`

---

## Environment Variables (Optional)

```bash
# .env file
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
MAX_UPLOAD_SIZE=10485760  # 10MB
```

---

## Project Structure

```
api/
├── main.py           # FastAPI application & endpoints
├── ml_engine.py      # ML logic (skill extraction, team formation)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Notes

- The API accepts **PDF files only** for resumes
- Student names are extracted from filenames (e.g., `john_doe.pdf` → "John Doe")
- Teams are balanced using a snake-draft algorithm for fairness
- Each team assignment includes an explanation (XAI)
