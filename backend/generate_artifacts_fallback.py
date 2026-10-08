"""
Fallback artifact generator — uses pure Python (no sklearn/xgboost DLLs needed).

Creates pickle-compatible stub objects that mimic sklearn/xgboost interfaces,
so the API can start on machines where native DLLs are blocked.

Run: python generate_artifacts_fallback.py
"""

import pickle
from pathlib import Path

from fallback_models import FallbackScaler, FallbackGMM, FallbackXGBRegressor

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("Generating fallback ML artifacts (pure Python, no DLLs)...")

    scaler = FallbackScaler(n_features=15)
    gmm = FallbackGMM(n_components=8)
    xgb = FallbackXGBRegressor()

    with open(ARTIFACTS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler.pkl")

    with open(ARTIFACTS_DIR / "gmm.pkl", "wb") as f:
        pickle.dump(gmm, f)
    print(f"  Saved gmm.pkl")

    with open(ARTIFACTS_DIR / "xgb_ranker.pkl", "wb") as f:
        pickle.dump(xgb, f)
    print(f"  Saved xgb_ranker.pkl")

    print(f"\nAll artifacts saved to {ARTIFACTS_DIR}")
    print("These are fallback stubs. Replace with real trained models when possible.")
