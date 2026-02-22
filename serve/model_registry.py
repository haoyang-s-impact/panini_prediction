"""Load saved model artifacts for inference.

Thin wrapper around models.registry — preserves the serve-layer API
so that serve/inference.py and serve/app.py imports work unchanged.

Usage:
    from serve.model_registry import load_model
    model, metadata = load_model()
"""

from models.registry import get_active_model  # noqa: F401
from models.registry import load_model as _registry_load


def load_model(model_id: str | None = None):
    """Load saved model and metadata from the registry.

    Args:
        model_id: Specific model to load. If None, loads the active model.

    Returns (model, metadata) tuple.
    """
    return _registry_load(model_id)
