"""
Linear recalibration for regression models.

Corrects regression-to-the-mean bias by fitting actual = slope * predicted + intercept
on cross-validated out-of-sample predictions, then applying the transform at inference.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

SAVED_DIR = Path(__file__).resolve().parent / "saved"


def fit_calibration(y_pred: np.ndarray, y_actual: np.ndarray) -> dict:
    """Fit linear calibration from OOS predicted → actual.

    Returns:
        {"slope": float, "intercept": float}
    """
    lr = LinearRegression()
    lr.fit(y_pred.reshape(-1, 1), y_actual)
    return {
        "slope": float(lr.coef_[0]),
        "intercept": float(lr.intercept_),
    }


def apply_calibration(y_pred: np.ndarray, cal_params: dict) -> np.ndarray:
    """Apply linear calibration: calibrated = slope * pred + intercept, clamped ≥ 0."""
    calibrated = cal_params["slope"] * np.asarray(y_pred) + cal_params["intercept"]
    return np.maximum(calibrated, 0)


def save_calibration(cal_params: dict, model_id: str) -> Path:
    """Save calibration params to models/saved/{model_id}/calibration.json."""
    out_dir = SAVED_DIR / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "calibration.json"
    path.write_text(json.dumps(cal_params, indent=2))
    print(f"Saved calibration to {path}")
    return path


def load_calibration(model_id: str) -> dict | None:
    """Load calibration params. Returns None if no calibration file exists."""
    path = SAVED_DIR / model_id / "calibration.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
