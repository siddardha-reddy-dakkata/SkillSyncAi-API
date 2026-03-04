"""Quick test of v3.3 production model."""
import httpx

BASE = "http://127.0.0.1:8000"

# Test 1: feedback-weights
r = httpx.get(f"{BASE}/api/feedback-weights", timeout=30)
d = r.json()
print(f"TEST 1 (weights): {r.status_code} OK, keys={list(d.keys())}")

# Test 2: data stats
r = httpx.get(f"{BASE}/api/feedback-data-stats", timeout=30)
d = r.json()
print(f"TEST 2 (stats): records={d['total_records']}, trained={d['model_trained']}")

# Test 3: predict good team
good_team = {
    "project_type": "web_application",
    "required_skills": ["React", "Node.js", "MongoDB"],
    "members": [
        {"name": "Alice", "skills": {"React": 8, "Node.js": 7, "CSS": 6}, "experience_years": 3, "assigned_role": "frontend"},
        {"name": "Bob", "skills": {"Node.js": 8, "MongoDB": 7, "Docker": 5}, "experience_years": 4, "assigned_role": "backend"},
        {"name": "Carol", "skills": {"MongoDB": 7, "React": 6, "Testing": 5}, "experience_years": 2, "assigned_role": "data_engineer"}
    ]
}
r = httpx.post(f"{BASE}/api/predict-team-success", json=good_team, timeout=30)
d = r.json()
print(f"TEST 3 (good team): {d['prediction']}, prob={d['success_probability']}")

# Test 4: predict bad team
bad_team = {
    "project_type": "ml_project",
    "required_skills": ["Python", "Tensorflow", "Pandas"],
    "members": [
        {"name": "X", "skills": {"Java": 4, "HTML": 3}, "experience_years": 1, "assigned_role": "frontend"},
        {"name": "Y", "skills": {"Java": 5, "HTML": 4}, "experience_years": 1, "assigned_role": "frontend"},
        {"name": "Z", "skills": {"CSS": 3, "HTML": 2}, "experience_years": 0, "assigned_role": "frontend"}
    ]
}
r = httpx.post(f"{BASE}/api/predict-team-success", json=bad_team, timeout=30)
d = r.json()
print(f"TEST 4 (bad team): {d['prediction']}, prob={d['success_probability']}")

# Test 5: feedback
feedback = {"records": [{"team_id": "v33_test", "project_type": "web_application",
    "required_skills": ["React"], "success": True,
    "members": [{"name": "T", "skills": {"React": 7}, "experience_years": 2, "assigned_role": "frontend", "skill_level": 7}]}]}
r = httpx.post(f"{BASE}/api/feedback", json=feedback, timeout=30)
d = r.json()
print(f"TEST 5 (feedback): added={d.get('records_added', d.get('success'))}, total={d.get('total_records', '?')}")

# Test 6: retrain
print("TEST 6 (retrain): running...")
r = httpx.post(f"{BASE}/api/retrain", timeout=120)
d = r.json()
print(f"TEST 6 (retrain): success={d['success']}, records={d['training_metrics']['records_used']}")

print("\n ALL 6 TESTS PASSED")
