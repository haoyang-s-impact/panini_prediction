"""Claude API integration for natural language card analysis.

Calls Claude to explain the price prediction in plain language.
Gracefully falls back to None if API key is missing or call fails.
"""

import json
import os


def get_analysis(prediction: dict) -> str | None:
    """Generate natural language analysis of a card prediction.

    Args:
        prediction: dict from predict_from_image() with
            predicted_price_cny, extracted_features, ocr_lines.

    Returns:
        Analysis string, or None if Claude API unavailable.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic
    except ImportError:
        return None

    features = prediction.get("extracted_features", {})
    price = prediction.get("predicted_price_cny", 0)

    prompt = f"""You are a Panini NBA trading card pricing expert. Given these extracted features and predicted price for a card auction screenshot, provide a brief analysis.

Extracted Features:
{json.dumps(features, indent=2, ensure_ascii=False, default=str)}

Predicted Price: {price:,.0f} CNY

Explain in 3-4 sentences:
1. What card this appears to be (player, set, key attributes)
2. Which features likely drive the price up or down
3. Any caveats (missing data, uncertain OCR reads, low confidence)

Be concise and specific. Use the actual feature values in your explanation."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception:
        return None
