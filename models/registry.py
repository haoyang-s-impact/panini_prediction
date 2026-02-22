"""Model registry for tracking trained models and their artifacts.

Manages registry.json and per-model artifact directories under models/saved/.

Usage:
    from models.registry import register_model, list_models, set_active_model

    # Register after training
    register_model(
        model=fitted_model,
        model_id="v5_catboost_tab",
        version="v5",
        framework="catboost",
        pipeline_type="ocr_tabular",
        metrics={"avg_r2": 0.44, "avg_mae": 12500},
        metadata={...},
        set_active=False,
    )

    # List / switch active model
    list_models()
    set_active_model("v5_catboost_tab")

    # Load model artifacts
    model, metadata = load_model()             # active model
    model, metadata = load_model("v5_xgb_tab") # specific model
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = PROJECT_ROOT / "models" / "saved"
REGISTRY_PATH = REGISTRY_DIR / "registry.json"


def _load_registry() -> dict:
    """Load registry.json, creating default if it doesn't exist."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"schema_version": 1, "active_model": None, "models": {}}


def _save_registry(registry: dict):
    """Write registry.json to disk."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def register_model(
    model,
    model_id: str,
    version: str,
    framework: str,
    pipeline_type: str,
    metrics: dict,
    metadata: dict,
    description: str = "",
    set_active: bool = False,
) -> str:
    """Save model artifacts and register in registry.json.

    Args:
        model: Fitted model object (XGBoost, LightGBM, or CatBoost).
        model_id: Unique identifier, e.g. "v5_catboost_tab".
        version: Model version, e.g. "v5".
        framework: "xgboost", "lightgbm", or "catboost".
        pipeline_type: "ocr_tabular", "ocr_tabular_pca30", etc.
        metrics: Evaluation metrics dict (avg_r2, avg_mae, avg_rmse, etc.).
        metadata: Dict with feature_names, feature_dtypes, category_mappings,
                  best_params, target, target_transform, n_samples, n_features.
        description: Human-readable description.
        set_active: If True, set this model as the active serving model.

    Returns:
        model_id
    """
    # Create artifact directory
    artifact_dir = REGISTRY_DIR / model_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save model artifact
    joblib.dump(model, artifact_dir / "model.joblib")

    # Save full metadata (feature schema, params, etc.)
    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Update registry.json with summary entry
    registry = _load_registry()
    registry["models"][model_id] = {
        "model_id": model_id,
        "version": version,
        "framework": framework,
        "pipeline_type": pipeline_type,
        "description": description,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": metadata.get("n_samples"),
        "n_features": metadata.get("n_features"),
        "target": metadata.get("target", "price_cny"),
        "target_transform": metadata.get("target_transform", "none"),
        "metrics": metrics,
        "artifact_dir": model_id,
    }

    if set_active:
        registry["active_model"] = model_id

    _save_registry(registry)

    status = " (set as ACTIVE)" if set_active else ""
    print(f"Registered model '{model_id}'{status}")
    print(f"  Artifacts: {artifact_dir}/")
    return model_id


def load_model(model_id: str | None = None):
    """Load a model's artifacts from disk.

    Args:
        model_id: Specific model to load. If None, loads the active model.

    Returns:
        (model, metadata) tuple.
    """
    registry = _load_registry()

    if model_id is None:
        model_id = registry.get("active_model")
        if model_id is None:
            raise RuntimeError(
                "No active model set in registry. "
                "Register a model with set_active=True, or run: "
                "python -m models.train_production_model"
            )

    if model_id not in registry["models"]:
        available = list(registry["models"].keys())
        raise ValueError(
            f"Model '{model_id}' not found in registry. Available: {available}"
        )

    entry = registry["models"][model_id]
    artifact_dir = REGISTRY_DIR / entry["artifact_dir"]

    model_path = artifact_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. "
            "The model may need to be retrained."
        )

    model = joblib.load(model_path)

    metadata_path = artifact_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    return model, metadata


def get_active_model() -> dict | None:
    """Return registry entry for the active model, or None."""
    registry = _load_registry()
    active_id = registry.get("active_model")
    if active_id and active_id in registry["models"]:
        return registry["models"][active_id]
    return None


def set_active_model(model_id: str):
    """Set the active serving model by model_id."""
    registry = _load_registry()
    if model_id not in registry["models"]:
        available = list(registry["models"].keys())
        raise ValueError(
            f"Model '{model_id}' not found in registry. Available: {available}"
        )
    registry["active_model"] = model_id
    _save_registry(registry)
    print(f"Active model set to '{model_id}'")


def list_models() -> list[dict]:
    """List all registered models with active indicator."""
    registry = _load_registry()
    active_id = registry.get("active_model")

    models = []
    for model_id, entry in registry["models"].items():
        entry_copy = dict(entry)
        entry_copy["is_active"] = model_id == active_id
        models.append(entry_copy)

    return models


def build_metadata(X, y_col="price_cny", target_transform="none", best_params=None):
    """Build standard metadata dict from training DataFrame.

    Args:
        X: Feature DataFrame used for training.
        y_col: Target column name.
        target_transform: "none" or "log1p".
        best_params: Tuned hyperparameters dict.

    Returns:
        metadata dict suitable for register_model().
    """
    cat_cols = [col for col in X.columns if X[col].dtype.name == "category"]
    category_mappings = {col: list(X[col].cat.categories) for col in cat_cols}

    metadata = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "feature_names": X.columns.tolist(),
        "feature_dtypes": {col: str(X[col].dtype) for col in X.columns},
        "category_mappings": category_mappings,
        "target": y_col,
        "target_transform": target_transform,
    }
    if best_params is not None:
        metadata["best_params"] = best_params
    return metadata


def print_models():
    """Print a formatted table of all registered models."""
    models = list_models()
    if not models:
        print("No models registered.")
        return

    print(f"\n{'ID':<30} {'Framework':<10} {'Version':<8} {'Metrics':<30} {'Active'}")
    print("-" * 90)
    for m in models:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in m["metrics"].items())
        active = "*" if m["is_active"] else ""
        print(f"{m['model_id']:<30} {m['framework']:<10} {m['version']:<8} {metrics_str:<30} {active}")
    print()
