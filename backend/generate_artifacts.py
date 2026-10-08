"""
Script to generate placeholder ML artifacts (scaler, GMM, XGBoost)
so the API can start without running Jupyter notebooks first.

Run: python generate_artifacts.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Number of features and synthetic samples
N_FEATURES = 15
N_SAMPLES = 500
N_CLUSTERS = 8
RANDOM_STATE = 42

# Section types for one-hot
SECTION_TYPES = ["intro", "related_work", "methods", "results", "discussion", "conclusion"]


def generate_gmm_artifacts():
    """Generate synthetic data, fit StandardScaler + GMM, save to disk."""
    rng = np.random.RandomState(RANDOM_STATE)

    # Synthetic feature data: 15 features, each roughly in [0, 1]
    X = rng.rand(N_SAMPLES, N_FEATURES)

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit GMM
    gmm = GaussianMixture(
        n_components=N_CLUSTERS,
        covariance_type="diag",
        random_state=RANDOM_STATE,
    )
    gmm.fit(X_scaled)

    # Save
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
    joblib.dump(gmm, ARTIFACTS_DIR / "gmm.pkl")
    print(f"Saved scaler.pkl and gmm.pkl to {ARTIFACTS_DIR}")


def generate_xgboost_artifact():
    """Generate synthetic training data, fit XGBoost ranker, save to disk."""
    rng = np.random.RandomState(RANDOM_STATE)

    # Feature dimensions: 15 draft features + 6 section_type one-hot + 3 per-section + 8 archetype one-hot = 32
    n_section_features = len(SECTION_TYPES) + 3 + N_CLUSTERS  # one-hot + words + rev_count + days + archetype
    total_features = N_FEATURES + n_section_features

    X = rng.rand(N_SAMPLES, total_features)

    # Synthetic priority labels using a heuristic formula (bootstrap — replace with real feedback)
    # Higher priority for: low revision count, many days since revision, incomplete sections
    # This is a SYNTHETIC BOOTSTRAP — replace with real user feedback (thumbs up/down) once available
    y = (
        0.3 * (1 - X[:, 10])      # low revision_velocity → higher priority
        + 0.2 * X[:, 11]           # abandoned sections → higher priority
        + 0.2 * X[:, 14]           # long time since revision → higher priority
        + 0.15 * (1 - X[:, 4])    # low structural completeness → higher priority
        + 0.15 * X[:, 2]           # overclaiming → higher priority
    )
    y = np.clip(y, 0, 1)

    model = XGBRegressor(
        objective="reg:squarederror",
        max_depth=4,
        n_estimators=200,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)

    joblib.dump(model, ARTIFACTS_DIR / "xgb_ranker.pkl")
    print(f"Saved xgb_ranker.pkl to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    print("Generating placeholder ML artifacts...")
    generate_gmm_artifacts()
    generate_xgboost_artifact()
    print("Done! Artifacts ready for API startup.")
