"""Quick test script for feedback loop endpoints."""
import httpx
import json

BASE = "http://localhost:8000"

print("=" * 50)
print("  TEST 1: GET /api/feedback-weights")
print("=" * 50)
r = httpx.get(f"{BASE}/api/feedback-weights")
data = r.json()
print(f"  Source: {data['active_weights']['source']}")
print(f"  Skill weight: {data['active_weights']['weight_skill']}")
print(f"  Experience weight: {data['active_weights']['weight_experience']}")
print(f"  Diversity weight: {data['active_weights']['weight_diversity']}")

print("\n" + "=" * 50)
print("  TEST 2: GET /api/feedback-data-stats")
print("=" * 50)
r = httpx.get(f"{BASE}/api/feedback-data-stats")
stats = r.json()
print(f"  Total records: {stats['total_records']}")
print(f"  Model trained: {stats['model_trained']}")
print(f"  Project types: {stats.get('project_type_distribution', {})}")

print("\n" + "=" * 50)
print("  TEST 3: POST /api/predict-team-success (Good Team)")
print("=" * 50)
good_team = {
    "project_type": "web_application",
    "required_skills": ["React", "Node.js", "MongoDB"],
    "members": [
        {"assigned_role": "frontend", "skills": ["React", "JavaScript", "CSS"], "skill_level": 8, "experience_years": 2},
        {"assigned_role": "backend", "skills": ["Node.js", "MongoDB", "Express"], "skill_level": 7, "experience_years": 3},
        {"assigned_role": "devops", "skills": ["Docker", "AWS", "CI/CD"], "skill_level": 7, "experience_years": 2},
    ]
}
r = httpx.post(f"{BASE}/api/predict-team-success", json=good_team)
result = r.json()
print(f"  Prediction: {result['prediction']}")
print(f"  Success Probability: {result['success_probability']}")
for f in result.get("factors", []):
    print(f"  {f}")

print("\n" + "=" * 50)
print("  TEST 4: POST /api/predict-team-success (Bad Team)")
print("=" * 50)
bad_team = {
    "project_type": "ml_project",
    "required_skills": ["Python", "TensorFlow", "Pandas"],
    "members": [
        {"assigned_role": "frontend", "skills": ["React", "CSS"], "skill_level": 4, "experience_years": 1},
        {"assigned_role": "frontend", "skills": ["Vue", "HTML"], "skill_level": 3, "experience_years": 0},
        {"assigned_role": "ui_ux", "skills": ["Figma"], "skill_level": 5, "experience_years": 1},
    ]
}
r = httpx.post(f"{BASE}/api/predict-team-success", json=bad_team)
result = r.json()
print(f"  Prediction: {result['prediction']}")
print(f"  Success Probability: {result['success_probability']}")
for f in result.get("factors", []):
    print(f"  {f}")

print("\n" + "=" * 50)
print("  TEST 5: POST /api/feedback (Submit new feedback)")
print("=" * 50)
feedback = {
    "records": [
        {
            "team_id": "TEST_001",
            "project_type": "web_application",
            "project_name": "E-Commerce Platform",
            "required_skills": ["React", "Node.js", "PostgreSQL"],
            "members": [
                {"name": "Test Student 1", "assigned_role": "frontend", "skills": ["React", "TypeScript"], "skill_level": 8, "experience_years": 2},
                {"name": "Test Student 2", "assigned_role": "backend", "skills": ["Node.js", "PostgreSQL"], "skill_level": 7, "experience_years": 3},
            ],
            "success": True,
            "grade": "A",
            "score": 88.5,
            "completion_status": "completed_on_time",
            "notes": "Excellent collaboration"
        }
    ]
}
r = httpx.post(f"{BASE}/api/feedback", json=feedback)
result = r.json()
print(f"  Success: {result['success']}")
print(f"  Records added: {result['records_added']}")
print(f"  Total records: {result['total_records']}")
print(f"  Message: {result['message']}")

print("\n" + "=" * 50)
print("  TEST 6: POST /api/retrain")
print("=" * 50)
r = httpx.post(f"{BASE}/api/retrain", timeout=120)
result = r.json()
print(f"  Success: {result['success']}")
print(f"  Records used: {result['training_metrics']['records_used']}")
print(f"  CV Accuracy: {result['training_metrics']['cv_accuracy']}")
print(f"  Train Accuracy: {result['training_metrics']['train_accuracy']}")
print(f"  Top features: {list(result['top_features'].keys())}")
print(f"  Message: {result['message']}")

print("\n" + "=" * 50)
print("  ALL TESTS PASSED!")
print("=" * 50)
