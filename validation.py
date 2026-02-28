"""
SkillSyncAI - Team Formation Validation Module

This module provides validation metrics and methods to evaluate
the quality of team formation predictions.

VALIDATION APPROACHES:
1. Quality Metrics - Measure team balance, diversity, coverage
2. Ground Truth Comparison - Compare against ideal/manual assignments
3. Statistical Tests - Analyze distribution and fairness
4. Simulation - Test with synthetic data
"""

import math
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter


class TeamFormationValidator:
    """
    Validates team formation output using multiple metrics.
    
    Metrics:
    - Gini Coefficient: Measures inequality in team strengths (lower = fairer)
    - Role Coverage: Percentage of required roles filled
    - Skill Balance: How evenly skills are distributed across teams
    - Diversity Score: Variety of skills within each team
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_teams(
        self,
        teams: List[Dict],
        profiles: List[Dict],
        required_roles: List[str] = None
    ) -> Dict:
        """
        Run comprehensive validation on formed teams.
        
        Returns:
            Dict with all validation metrics and pass/fail status
        """
        if not teams:
            return {"error": "No teams to validate", "valid": False}
        
        results = {
            "metrics": {},
            "quality_score": 0,
            "issues": [],
            "valid": True,
        }
        
        # 1. Calculate Gini Coefficient (Team Fairness)
        gini = self._calculate_gini(teams)
        results["metrics"]["gini_coefficient"] = {
            "value": gini,
            "threshold": 0.15,
            "passed": gini < 0.15,
            "interpretation": "Excellent" if gini < 0.1 else "Good" if gini < 0.15 else "Unbalanced"
        }
        
        # 2. Role Coverage
        coverage = self._calculate_role_coverage(teams, required_roles or [])
        results["metrics"]["role_coverage"] = {
            "value": coverage,
            "threshold": 0.75,
            "passed": coverage >= 0.75,
            "interpretation": f"{coverage*100:.0f}% of required roles filled"
        }
        
        # 3. Skill Balance Across Teams
        balance = self._calculate_skill_balance(teams)
        results["metrics"]["skill_balance"] = {
            "value": balance,
            "threshold": 0.7,
            "passed": balance >= 0.7,
            "interpretation": "Well balanced" if balance >= 0.8 else "Acceptable" if balance >= 0.7 else "Imbalanced"
        }
        
        # 4. Team Diversity
        diversity = self._calculate_team_diversity(teams)
        results["metrics"]["diversity_score"] = {
            "value": diversity,
            "threshold": 0.6,
            "passed": diversity >= 0.6,
            "interpretation": f"Average {diversity*100:.0f}% role diversity per team"
        }
        
        # 5. Assignment Rate
        assigned = sum(len(t.get("members", [])) for t in teams)
        total = len(profiles)
        assignment_rate = assigned / max(total, 1)
        results["metrics"]["assignment_rate"] = {
            "value": assignment_rate,
            "threshold": 1.0,
            "passed": assignment_rate >= 0.95,
            "interpretation": f"{assigned}/{total} students assigned"
        }
        
        # Calculate overall quality score (weighted average)
        weights = {
            "gini_coefficient": 0.25,
            "role_coverage": 0.20,
            "skill_balance": 0.25,
            "diversity_score": 0.15,
            "assignment_rate": 0.15,
        }
        
        quality_score = 0
        for metric, weight in weights.items():
            if results["metrics"][metric]["passed"]:
                quality_score += weight * 100
            else:
                # Partial credit based on how close to threshold
                value = results["metrics"][metric]["value"]
                threshold = results["metrics"][metric]["threshold"]
                if metric == "gini_coefficient":
                    # For Gini, lower is better
                    ratio = max(0, 1 - (value - threshold) / threshold)
                else:
                    ratio = value / threshold if threshold > 0 else 0
                quality_score += weight * min(100, ratio * 100)
        
        results["quality_score"] = round(quality_score, 1)
        results["quality_grade"] = self._score_to_grade(quality_score)
        
        # Identify issues
        for metric, data in results["metrics"].items():
            if not data["passed"]:
                results["issues"].append(f"{metric}: {data['interpretation']}")
                results["valid"] = False
        
        if not results["issues"]:
            results["valid"] = True
            results["issues"] = ["All validation checks passed"]
        
        return results
    
    def _calculate_gini(self, teams: List[Dict]) -> float:
        """
        Calculate Gini coefficient for team strength distribution.
        0 = perfect equality, 1 = perfect inequality
        """
        strengths = []
        for team in teams:
            members = team.get("members", [])
            if members:
                avg_score = sum(
                    m.get("overall_score", m.get("experience_score", 0))
                    for m in members
                ) / len(members)
                strengths.append(avg_score)
        
        if len(strengths) < 2:
            return 0.0
        
        # Gini calculation
        strengths = sorted(strengths)
        n = len(strengths)
        total = sum(strengths)
        
        if total == 0:
            return 0.0
        
        cumsum = 0
        for i, s in enumerate(strengths):
            cumsum += (i + 1) * s
        
        gini = (2 * cumsum) / (n * total) - (n + 1) / n
        return round(abs(gini), 4)
    
    def _calculate_role_coverage(self, teams: List[Dict], required_roles: List[str]) -> float:
        """Calculate what percentage of required roles are filled."""
        if not required_roles:
            # If no requirements specified, check general coverage
            all_roles = {"frontend", "backend", "ml", "data", "devops", "uiux", "fullstack"}
            filled = set()
            for team in teams:
                for member in team.get("members", []):
                    role = member.get("assigned_role", member.get("primary_role", ""))
                    filled.add(role)
            return len(filled) / len(all_roles)
        
        total_required = len(teams) * len(required_roles)
        filled = 0
        
        for team in teams:
            team_roles = set(
                m.get("assigned_role", m.get("primary_role", ""))
                for m in team.get("members", [])
            )
            for role in required_roles:
                if role in team_roles:
                    filled += 1
        
        return filled / max(total_required, 1)
    
    def _calculate_skill_balance(self, teams: List[Dict]) -> float:
        """
        Calculate how balanced skills are across teams.
        Uses coefficient of variation (lower = more balanced).
        """
        team_scores = []
        for team in teams:
            members = team.get("members", [])
            if members:
                total_score = sum(
                    m.get("overall_score", m.get("experience_score", 0))
                    for m in members
                )
                team_scores.append(total_score)
        
        if len(team_scores) < 2:
            return 1.0
        
        mean = sum(team_scores) / len(team_scores)
        if mean == 0:
            return 1.0
        
        variance = sum((s - mean) ** 2 for s in team_scores) / len(team_scores)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean  # Coefficient of variation
        
        # Convert to 0-1 score (1 = perfectly balanced)
        balance = max(0, 1 - cv)
        return round(balance, 4)
    
    def _calculate_team_diversity(self, teams: List[Dict]) -> float:
        """Calculate average role diversity within teams."""
        diversities = []
        for team in teams:
            members = team.get("members", [])
            if members:
                roles = [
                    m.get("assigned_role", m.get("primary_role", ""))
                    for m in members
                ]
                unique_roles = len(set(roles))
                diversity = unique_roles / len(members)
                diversities.append(diversity)
        
        if not diversities:
            return 0.0
        
        return round(sum(diversities) / len(diversities), 4)
    
    def _score_to_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def compare_with_ground_truth(
        self,
        predicted_teams: List[Dict],
        ground_truth_teams: List[Dict]
    ) -> Dict:
        """
        Compare predicted team assignments with ground truth (ideal) assignments.
        
        Uses set similarity metrics:
        - Precision: How many predicted pairs are correct
        - Recall: How many correct pairs were predicted
        - F1 Score: Harmonic mean of precision and recall
        """
        # Extract pairs from predicted teams
        predicted_pairs = set()
        for team in predicted_teams:
            members = [m.get("student_id", m.get("name", "")) for m in team.get("members", [])]
            for i, m1 in enumerate(members):
                for m2 in members[i+1:]:
                    predicted_pairs.add(tuple(sorted([m1, m2])))
        
        # Extract pairs from ground truth
        truth_pairs = set()
        for team in ground_truth_teams:
            members = team.get("members", [])
            if isinstance(members[0], dict):
                members = [m.get("student_id", m.get("name", "")) for m in members]
            for i, m1 in enumerate(members):
                for m2 in members[i+1:]:
                    truth_pairs.add(tuple(sorted([m1, m2])))
        
        # Calculate metrics
        true_positives = len(predicted_pairs & truth_pairs)
        false_positives = len(predicted_pairs - truth_pairs)
        false_negatives = len(truth_pairs - predicted_pairs)
        
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "correct_pairs": true_positives,
            "total_predicted_pairs": len(predicted_pairs),
            "total_truth_pairs": len(truth_pairs),
            "interpretation": "Excellent" if f1 >= 0.8 else "Good" if f1 >= 0.6 else "Fair" if f1 >= 0.4 else "Poor"
        }
    
    def generate_validation_report(
        self,
        teams: List[Dict],
        profiles: List[Dict],
        ground_truth: List[Dict] = None
    ) -> Dict:
        """Generate a comprehensive validation report."""
        report = {
            "summary": {},
            "quality_metrics": {},
            "ground_truth_comparison": None,
            "recommendations": [],
        }
        
        # Run validation
        validation = self.validate_teams(teams, profiles)
        report["quality_metrics"] = validation["metrics"]
        report["summary"] = {
            "quality_score": validation["quality_score"],
            "quality_grade": validation["quality_grade"],
            "all_checks_passed": validation["valid"],
            "issues_found": len([i for i in validation["issues"] if "passed" not in i.lower()]),
        }
        
        # Ground truth comparison if provided
        if ground_truth:
            report["ground_truth_comparison"] = self.compare_with_ground_truth(
                teams, ground_truth
            )
        
        # Generate recommendations
        for metric, data in validation["metrics"].items():
            if not data["passed"]:
                if metric == "gini_coefficient":
                    report["recommendations"].append(
                        "Teams are unbalanced. Consider redistributing high-skill students."
                    )
                elif metric == "role_coverage":
                    report["recommendations"].append(
                        "Missing required roles. Consider relaxing constraints or adding more students."
                    )
                elif metric == "skill_balance":
                    report["recommendations"].append(
                        "Skill distribution is uneven. Try the Hungarian algorithm for optimal assignment."
                    )
                elif metric == "diversity_score":
                    report["recommendations"].append(
                        "Low role diversity. Ensure teams have varied skill sets."
                    )
        
        if not report["recommendations"]:
            report["recommendations"] = ["Team formation is optimal. No changes needed."]
        
        return report


# ============================================================
# SAMPLE VALIDATION DATA FORMAT
# ============================================================

SAMPLE_VALIDATION_DATA = {
    "test_cases": [
        {
            "case_id": "TC001",
            "description": "Balanced 4-person teams for web project",
            "input_students": [
                {"student_id": "S1", "name": "Alice", "primary_role": "frontend", "skills": {"frontend": 8, "backend": 3}},
                {"student_id": "S2", "name": "Bob", "primary_role": "backend", "skills": {"backend": 9, "frontend": 2}},
                {"student_id": "S3", "name": "Carol", "primary_role": "ml", "skills": {"ml": 7, "data": 5}},
                {"student_id": "S4", "name": "Dave", "primary_role": "devops", "skills": {"devops": 6, "backend": 4}},
                {"student_id": "S5", "name": "Eve", "primary_role": "frontend", "skills": {"frontend": 7, "uiux": 6}},
                {"student_id": "S6", "name": "Frank", "primary_role": "backend", "skills": {"backend": 8, "ml": 3}},
                {"student_id": "S7", "name": "Grace", "primary_role": "data", "skills": {"data": 8, "ml": 4}},
                {"student_id": "S8", "name": "Henry", "primary_role": "fullstack", "skills": {"fullstack": 7, "backend": 5}},
            ],
            "expected_teams": [
                {"team_id": "Team_01", "members": ["S1", "S2", "S3", "S4"]},
                {"team_id": "Team_02", "members": ["S5", "S6", "S7", "S8"]},
            ],
            "validation_criteria": {
                "min_gini": 0.15,
                "min_diversity": 0.6,
                "required_roles": ["frontend", "backend"],
            }
        }
    ],
    "ground_truth_format": {
        "description": "Ideal team assignments based on expert judgment or past success",
        "teams": [
            {
                "team_id": "Ideal_Team_01",
                "members": ["S1", "S2", "S3", "S4"],
                "reason": "Balanced frontend, backend, ML, and DevOps coverage",
                "historical_performance": "A grade"
            }
        ]
    }
}


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SkillSyncAI - Team Formation Validation")
    print("=" * 60)
    
    # Sample teams to validate
    sample_teams = [
        {
            "team_id": "Team_01",
            "members": [
                {"student_id": "S1", "name": "Alice", "assigned_role": "frontend", "overall_score": 0.72},
                {"student_id": "S2", "name": "Bob", "assigned_role": "backend", "overall_score": 0.68},
                {"student_id": "S3", "name": "Carol", "assigned_role": "ml", "overall_score": 0.75},
            ]
        },
        {
            "team_id": "Team_02",
            "members": [
                {"student_id": "S4", "name": "Dave", "assigned_role": "devops", "overall_score": 0.65},
                {"student_id": "S5", "name": "Eve", "assigned_role": "frontend", "overall_score": 0.70},
                {"student_id": "S6", "name": "Frank", "assigned_role": "data", "overall_score": 0.73},
            ]
        },
    ]
    
    sample_profiles = [{"student_id": f"S{i}"} for i in range(1, 7)]
    
    # Run validation
    validator = TeamFormationValidator()
    results = validator.validate_teams(sample_teams, sample_profiles)
    
    print(f"\n📊 Quality Score: {results['quality_score']}/100 (Grade: {results['quality_grade']})")
    print("\n📈 Metrics:")
    for metric, data in results["metrics"].items():
        status = "✅" if data["passed"] else "❌"
        print(f"   {status} {metric}: {data['value']:.4f} ({data['interpretation']})")
    
    print(f"\n📋 Issues: {', '.join(results['issues'])}")
    
    # Generate full report
    report = validator.generate_validation_report(sample_teams, sample_profiles)
    print(f"\n📝 Recommendations:")
    for rec in report["recommendations"]:
        print(f"   • {rec}")
