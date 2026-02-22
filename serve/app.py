"""Panini Card Price Prediction — Streamlit App.

Upload a card auction screenshot and get an AI-powered price prediction.

Usage:
    streamlit run serve/app.py
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Panini Card Price Predictor",
    page_icon="🏀",
    layout="wide",
)


@st.cache_resource
def load_ocr_reader():
    """Load EasyOCR reader once, keep in memory."""
    import easyocr
    import torch
    use_gpu = torch.cuda.is_available()
    return easyocr.Reader(["ch_sim", "en"], gpu=use_gpu, verbose=False)


@st.cache_resource
def load_prediction_model():
    """Load saved XGBoost model once, keep in memory."""
    from serve.model_registry import load_model
    return load_model()


def run_prediction(image_bytes: bytes) -> dict:
    """Run the full prediction pipeline on uploaded image bytes."""
    # Pre-warm cached resources
    load_ocr_reader()
    load_prediction_model()

    from serve.inference import predict_from_image
    return predict_from_image(image_bytes)


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

# Check if model is saved
model_path = Path("models/saved/model.joblib")
if not model_path.exists():
    st.error(
        "No trained model found. Run `python -m models.train_production_model` "
        "first to train and save the model."
    )
    st.stop()

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
            result = run_prediction(image_bytes)

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
