"""Panini Card Price Prediction — Streamlit App.

Upload a card auction screenshot and get an AI-powered price prediction.

Usage:
    streamlit run serve/app.py                                  # use active model
    streamlit run serve/app.py -- --model v5_catboost_tab       # use specific model
    streamlit run serve/app.py -- --model v4_xgb_ocr_tabular   # use V4 XGBoost
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `models` and `data` packages are importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

# --- Parse --model argument before Streamlit consumes argv ---
_model_id_override = None
if "--model" in sys.argv:
    idx = sys.argv.index("--model")
    if idx + 1 < len(sys.argv):
        _model_id_override = sys.argv[idx + 1]

st.set_page_config(
    page_title="Panini Card Price Predictor",
    page_icon="🏀",
    layout="wide",
)


def _resolve_model_id() -> str | None:
    """Return the model ID to serve (from CLI arg or registry active model)."""
    if _model_id_override:
        return _model_id_override
    from models.registry import get_active_model
    entry = get_active_model()
    return entry["model_id"] if entry else None


def _get_model_entry(model_id: str) -> dict | None:
    """Get registry entry for the given model ID."""
    from models.registry import list_models
    for m in list_models():
        if m["model_id"] == model_id:
            return m
    return None


@st.cache_resource
def load_ocr_reader():
    """Load EasyOCR reader once, keep in memory."""
    import easyocr
    import torch
    use_gpu = torch.cuda.is_available()
    return easyocr.Reader(["ch_sim", "en"], gpu=use_gpu, verbose=False)


@st.cache_resource
def load_prediction_model(model_id: str):
    """Load a specific model once, keep in memory."""
    from serve.model_registry import load_model
    return load_model(model_id)


def _get_predict_fn(framework: str, model_id: str):
    """Return the right predict_from_image function for the model framework."""
    if framework == "catboost":
        import serve.inference_v5_catboost as mod
        mod.MODEL_ID = model_id
        return mod.predict_from_image
    else:
        import serve.inference as mod
        mod.MODEL_ID = model_id
        return mod.predict_from_image


def run_prediction(image_bytes: bytes, framework: str, model_id: str) -> dict:
    """Run the full prediction pipeline on uploaded image bytes."""
    load_ocr_reader()
    load_prediction_model(model_id)

    predict_fn = _get_predict_fn(framework, model_id)
    return predict_fn(image_bytes)


def format_feature_value(key: str, value) -> str:
    """Format a feature value for display."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


# --- UI ---

st.title("Panini Card Price Predictor")
st.markdown("Upload a card auction screenshot to get an estimated sale price.")

# Resolve which model to serve
serving_model_id = _resolve_model_id()

if serving_model_id is None:
    st.error(
        "No active model in registry. Run `python -m models.train_production_model` "
        "first to train and register a model."
    )
    st.stop()

model_entry = _get_model_entry(serving_model_id)
if model_entry is None:
    st.error(f"Model '{serving_model_id}' not found in registry.")
    st.stop()

framework = model_entry["framework"]
source = "CLI override" if _model_id_override else "registry active"
st.caption(f"Model: **{serving_model_id}** ({framework}) — {source}")

uploaded = st.file_uploader(
    "Upload card screenshot",
    type=["jpg", "jpeg", "png"],
    help="Upload a Chinese auction screenshot (from WeChat/Xianyu) of a Panini NBA card.",
)

if uploaded is not None:
    image_bytes = uploaded.read()

    col_img, col_results = st.columns([1, 1])

    with col_img:
        st.image(image_bytes, caption=uploaded.name, use_container_width=True)

    with col_results:
        with st.spinner("Analyzing card... (first run may take ~10s to load OCR model)"):
            result = run_prediction(image_bytes, framework, serving_model_id)

        if result.get("error"):
            st.warning(result["error"])
        else:
            # Price prediction
            price = result["predicted_price_cny"]
            st.metric(
                label="Estimated Price",
                value=f"¥{price:,.0f} CNY",
            )

            # Extracted features
            features = result.get("extracted_features", {})
            if features:
                st.subheader("Card Details")

                # Group features for readable display
                identity_keys = ["player_name", "team", "card_year", "card_series"]
                attribute_keys = [
                    "is_rookie", "is_autograph", "has_patch", "is_refractor",
                    "parallel_type", "card_features",
                ]
                auction_keys = ["price_cny", "bid_times", "end_time"]
                number_keys = ["serial_max", "grade", "grading_company"]

                def show_features(keys, label):
                    items = {k: features[k] for k in keys if k in features}
                    if items:
                        st.markdown(f"**{label}**")
                        for k, v in items.items():
                            display_key = k.replace("_", " ").title()
                            st.text(f"  {display_key}: {format_feature_value(k, v)}")

                show_features(identity_keys, "Identity")
                show_features(attribute_keys, "Attributes")
                show_features(number_keys, "Numbers & Grading")
                show_features(auction_keys, "Auction Data (from OCR)")

            # Claude analysis
            from dotenv import load_dotenv
            load_dotenv()

            from serve.claude_reasoning import get_analysis
            with st.spinner("Generating AI analysis..."):
                analysis = get_analysis(result)

            if analysis:
                st.subheader("AI Analysis")
                st.info(analysis)

            # OCR debug
            ocr_lines = result.get("ocr_lines", [])
            if ocr_lines:
                with st.expander(f"Raw OCR Text ({len(ocr_lines)} lines)"):
                    for line in ocr_lines:
                        conf = line["confidence"]
                        st.text(f"[{conf:.2f}] {line['text']}")
