"""
Test script to evaluate feature extractors on sample OCR data
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from panini_card_ocr_etl import (
    load_ocr_data,
    extract_player_name,
    extract_card_series,
    extract_serial_numbers,
    extract_grading,
    extract_temporal,
    extract_descriptors,
    extract_team,
    extract_price,
    extract_bid_times,
)

# Load OCR data
print("Loading OCR data...")
df = load_ocr_data()
print(f"Loaded {len(df)} OCR entries from {df['image'].nunique()} images\n")

# Test on first 3 images
test_images = df['image'].unique()[:3]

for img_name in test_images:
    print("=" * 80)
    print(f"Testing on image: {img_name}")
    print("=" * 80)

    # Get OCR entries for this image
    img_data = df[df['image'] == img_name]
    ocr_entries = list(zip(img_data['text'], img_data['confidence'], img_data['bbox']))

    print(f"\nTotal OCR entries: {len(ocr_entries)}")
    print(f"Average confidence: {img_data['confidence'].mean():.3f}")

    # Test each extractor
    print("\n--- Player Name ---")
    player_name, conf = extract_player_name(ocr_entries)
    print(f"Player: {player_name} (confidence: {conf:.3f})")

    print("\n--- Card Series ---")
    card_year, card_series = extract_card_series(ocr_entries)
    print(f"Year: {card_year}, Series: {card_series}")

    print("\n--- Serial Numbers ---")
    serials = extract_serial_numbers(ocr_entries)
    print(f"Card #: {serials['card_number']}, Serial: {serials['serial_number']}, Max: {serials['serial_max']}")

    print("\n--- Grading ---")
    grading = extract_grading(ocr_entries)
    print(f"Company: {grading['grading_company']}, Grade: {grading['grade']}, Subtype: {grading['grade_subtype']}")

    print("\n--- Temporal Data ---")
    end_time, product_id = extract_temporal(ocr_entries)
    print(f"End Time: {end_time}")
    print(f"Product ID: {product_id}")

    print("\n--- Descriptors ---")
    descriptors = extract_descriptors(ocr_entries)
    print(f"Rookie: {descriptors['is_rookie']}")
    print(f"Autograph: {descriptors['is_autograph']}")
    print(f"Patch: {descriptors['has_patch']}")
    print(f"Refractor: {descriptors['is_refractor']}")
    print(f"Parallel: {descriptors['parallel_type']}")
    print(f"Features: {descriptors['card_features']}")

    print("\n--- Team ---")
    team = extract_team(ocr_entries)
    print(f"Team: {team}")

    print("\n--- Prices & Bids ---")
    price_cny, price_usd, shipping = None, None, None
    bid_times = None
    for text in img_data['text']:
        if not price_cny and not price_usd and not shipping:
            prices = extract_price(text)
            if prices[0] or prices[1] or prices[2]:
                price_cny, price_usd, shipping = prices
        if not bid_times:
            bid_times = extract_bid_times(text)

    print(f"CNY: ¥{price_cny}, USD: ${price_usd}, Shipping: ¥{shipping}")
    print(f"Bids: {bid_times}")

    print("\n")

print("=" * 80)
print("Test complete!")
