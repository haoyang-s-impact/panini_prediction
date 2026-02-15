"""Load saved model artifacts for inference.

This module is serving-only — it loads pre-trained models from disk.
For training, see models/train_production_model.py.

Usage:
    from serve.model_registry import load_model
    model, metadata = load_model()
"""

import json
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "saved"


def load_model(model_dir: Path | str | None = None):
    """Load saved model and metadata.

    Returns (model, metadata) tuple.
    """
    model_dir = Path(model_dir or DEFAULT_MODEL_DIR)

    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {model_path}. "
            "Run `python -m models.train_production_model` to train and save."
        )

    model = joblib.load(model_path)

    with open(metadata_path) as f:
        metadata = json.load(f)

    return model, metadata
