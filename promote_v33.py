"""Promote v3.3 to production model."""
import pickle, json, os
from datetime import datetime

MODEL_PATH = "model_weights.pkl"
REPORT_PATH = "training_report.json"

# Load v3.3
with open("model_versions/model_v3.3.pkl", "rb") as f:
    v33 = pickle.load(f)

print(f"v3.3 Model: {type(v33['model']).__name__}")
print(f"Features: {len(v33['feature_names'])}")
print(f"Metrics: acc={v33['training_metrics']['holdout_accuracy']:.3f}, "
      f"F1={v33['training_metrics']['holdout_f1']:.3f}, "
      f"AUC={v33['training_metrics']['holdout_auc']:.3f}")

# Save as production
save_data = {
    "model": v33["model"],
    "scaler": v33["scaler"],
    "learned_weights": v33["learned_weights"],
    "feature_importances": v33["feature_importances"],
    "feature_names": v33["feature_names"],
    "training_metrics": v33["training_metrics"],
    "keep_indices": v33.get("keep_indices"),
    "is_trained": True,
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(save_data, f)

# Update training report
report = {
    "training_date": datetime.now().isoformat(),
    "model_version": "v3.3",
    "model_description": "HistGradientBoosting (16 features, 407 records, real resume augmented)",
    "training_metrics": v33["training_metrics"],
    "learned_weights": v33["learned_weights"],
    "feature_importances": v33["feature_importances"],
}
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\nProduction model updated to v3.3")
print(f"  model_weights.pkl: {os.path.getsize(MODEL_PATH):,} bytes")
print(f"  training_report.json: updated")
