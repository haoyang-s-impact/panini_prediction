# %% [markdown]
# # Panini Card OCR Data ETL
# Load and process OCR text extraction results for machine learning

# %%
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz import fuzz, process
import ast

# Import NBA player data
import sys
sys.path.insert(0, str(Path(__file__).parent))
from data.nba_players import (
    NBA_PLAYERS,
    CHINESE_NAME_MAP,
    TEAM_MAPPINGS,
    PARALLEL_MAPPINGS,
    DESCRIPTOR_KEYWORDS
)

# %% [markdown]
# ## Configuration

# %%
# Spatial zones for position-based extraction (Y-axis ranges)
SPATIAL_ZONES = {
    'timestamp': (0, 250),
    'card_image': (250, 700),
    'player_info': (700, 1250),
    'description': (1250, 1600),
    'pricing': (1600, 1900),
    'metadata': (2000, 2600),
}

# Confidence thresholds per feature type
CONFIDENCE_THRESHOLDS = {
    'player_name': 0.50,
    'card_series': 0.60,
    'price': 0.70,
    'product_id': 0.85,
    'grades': 0.75,
    'general': 0.50,
}

# Regex patterns
PATTERNS = {
    'card_series': r'(20\d{2}[-/]?\d{2})\s+(Panini\s+[A-Za-z\s]+?)(?=\s+[A-Z]|签|$|\d|RC|新秀)',
    'card_year': r'(20\d{2}[-/]?\d{2})',
    'serial_number': r'(\d+)?/(\d+)编?',
    'card_number': r'#(\d+)',
    'simple_serial': r'(\d+)编',
    'end_time': r'结束时间[:：]\s*(\d{4}[-/]\d{2}[-/]\d{2}\s*\d{2}:\d{2}:\d{2})',
    'product_id': r'商品编号[:：]\s*(\d+)',
    'sold_status': r'商品[己已]售出',
    'psa_grade': r'PSA\s*(\d+\.?\d*)',
    'bgs_grade': r'BGS\s*(\d+\.?\d*)',
    'sgc_grade': r'SGC\s*(\d+\.?\d*)',
    'csg_grade': r'CSG\s*(\d+\.?\d*)',
    'grade_subtype': r'(金标|银标|黑标)',
    'player_english': r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
}

# %% [markdown]
# ## Helper Functions

# %%
def parse_bbox(bbox_str):
    """Parse bounding box string to extract coordinates."""
    try:
        bbox = ast.literal_eval(bbox_str)
        return bbox
    except:
        return None

def get_text_zone(bbox):
    """Determine spatial zone from bounding box Y-coordinate."""
    if bbox is None:
        return 'unknown'

    try:
        # Extract Y-coordinates
        y_coords = []
        for point in bbox:
            if len(point) >= 2:
                y_coords.append(float(point[1]))

        if not y_coords:
            return 'unknown'

        y_center = sum(y_coords) / len(y_coords)

        for zone, (y_min, y_max) in SPATIAL_ZONES.items():
            if y_min <= y_center < y_max:
                return zone
    except:
        pass

    return 'unknown'

def filter_by_confidence(ocr_entries, threshold=0.5):
    """Filter OCR entries by confidence threshold."""
    return [(text, conf, bbox) for text, conf, bbox in ocr_entries if conf >= threshold]

def filter_by_zone(ocr_entries, zone):
    """Filter OCR entries by spatial zone."""
    result = []
    for text, conf, bbox in ocr_entries:
        parsed_bbox = parse_bbox(bbox)
        if get_text_zone(parsed_bbox) == zone:
            result.append((text, conf, bbox))
    return result

# %% [markdown]
# ## Feature Extraction Functions

# %%
def fuzzy_match_player(text, player_list, threshold=85):
    """
    Fuzzy match OCR text to player name.

    Args:
        text: OCR extracted text
        player_list: List of valid player names
        threshold: Minimum similarity score (0-100)

    Returns:
        Matched player name or None
    """
    # Try exact substring match first (fast)
    text_upper = text.upper()
    for player in player_list:
        if player.upper() in text_upper or text_upper in player.upper():
            return player

    # Fuzzy match using token_sort_ratio (handles word order)
    result = process.extractOne(
        text,
        player_list,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold
    )

    return result[0] if result else None

def extract_player_name(ocr_entries):
    """Extract player name from OCR entries."""
    # Filter by confidence
    entries = filter_by_confidence(ocr_entries, CONFIDENCE_THRESHOLDS['player_name'])

    # 1. Check Chinese name mapping first
    for text, conf, bbox in entries:
        for cn_name, en_name in CHINESE_NAME_MAP.items():
            if cn_name in text:
                return en_name, conf

    # 2. Fuzzy match English names
    for text, conf, bbox in entries:
        # Skip very short text
        if len(text) < 5:
            continue
        matched = fuzzy_match_player(text, NBA_PLAYERS, threshold=85)
        if matched:
            return matched, conf

    # 3. Fallback: check player_info zone
    zone_entries = filter_by_zone(entries, 'player_info')
    for text, conf, bbox in zone_entries:
        matched = fuzzy_match_player(text, NBA_PLAYERS, threshold=80)
        if matched:
            return matched, conf

    return None, 0.0

def extract_card_series(ocr_entries):
    """Extract card year and series from OCR entries."""
    entries = filter_by_confidence(ocr_entries, CONFIDENCE_THRESHOLDS['card_series'])

    card_year = None
    card_series = None
    max_conf = 0.0

    for text, conf, bbox in entries:
        # Try to match full pattern
        match = re.search(PATTERNS['card_series'], text, re.IGNORECASE)
        if match and conf > max_conf:
            card_year = match.group(1)
            series = match.group(2).strip()
            # Normalize capitalization
            card_series = series.title()
            max_conf = conf

    return card_year, card_series

def extract_serial_numbers(ocr_entries):
    """Extract card number and serial number from OCR entries."""
    entries = filter_by_confidence(ocr_entries, CONFIDENCE_THRESHOLDS['general'])

    card_number = None
    serial_number = None
    serial_max = None

    for text, conf, bbox in entries:
        # Card number pattern (#184)
        if not card_number:
            match = re.search(PATTERNS['card_number'], text)
            if match:
                card_number = match.group(1)

        # Serial number pattern (05/25编)
        if not serial_number:
            match = re.search(PATTERNS['serial_number'], text)
            if match:
                num = match.group(1) if match.group(1) else ""
                max_val = match.group(2)
                serial_number = f"{num}/{max_val}" if num else f"/{max_val}"
                try:
                    serial_max = int(max_val)
                except:
                    pass

        # Simple serial (75编)
        if not serial_number:
            match = re.search(PATTERNS['simple_serial'], text)
            if match:
                serial_number = match.group(1)

    return {
        'card_number': card_number,
        'serial_number': serial_number,
        'serial_max': serial_max
    }

def extract_grading(ocr_entries):
    """Extract grading information from OCR entries."""
    entries = filter_by_confidence(ocr_entries, CONFIDENCE_THRESHOLDS['grades'])

    grading_company = None
    grade = None
    grade_subtype = None

    for text, conf, bbox in entries:
        # Check each grading company
        if not grading_company:
            for company in ['psa', 'bgs', 'sgc', 'csg']:
                pattern_key = f'{company}_grade'
                match = re.search(PATTERNS[pattern_key], text, re.IGNORECASE)
                if match:
                    grading_company = company.upper()
                    try:
                        grade = float(match.group(1))
                    except:
                        pass
                    break

        # Check for grade subtype (金标/黑标/银标)
        if not grade_subtype:
            match = re.search(PATTERNS['grade_subtype'], text)
            if match:
                subtype_map = {
                    '黑标': 'Black Label',
                    '金标': 'Gold Label',
                    '银标': 'Silver Label'
                }
                grade_subtype = subtype_map.get(match.group(1))

    return {
        'grading_company': grading_company,
        'grade': grade,
        'grade_subtype': grade_subtype
    }

def extract_temporal(ocr_entries):
    """Extract end time and product ID from OCR entries."""
    entries = filter_by_confidence(ocr_entries, CONFIDENCE_THRESHOLDS['general'])

    end_time = None
    product_id = None

    for text, conf, bbox in entries:
        # End time
        if not end_time:
            match = re.search(PATTERNS['end_time'], text)
            if match:
                end_time = match.group(1).strip()
                # Normalize spacing
                end_time = re.sub(r'\s+', ' ', end_time)

        # Product ID
        if not product_id:
            match = re.search(PATTERNS['product_id'], text)
            if match:
                product_id = match.group(1)

    return end_time, product_id

def extract_descriptors(ocr_entries):
    """Extract card descriptors (autograph, rookie, patch, parallel, etc.)."""
    entries = filter_by_confidence(ocr_entries, CONFIDENCE_THRESHOLDS['general'])

    # Initialize boolean flags
    is_autograph = False
    is_rookie = False
    has_patch = False
    is_refractor = False
    parallel_type = None
    features_list = []

    # Combine all text for comprehensive search
    all_text = ' '.join([text for text, conf, bbox in entries])

    # Check each descriptor type
    for desc_type, patterns in DESCRIPTOR_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, all_text, re.IGNORECASE):
                if desc_type == 'autograph':
                    is_autograph = True
                    features_list.append('Autograph')
                elif desc_type == 'rookie':
                    is_rookie = True
                    features_list.append('Rookie')
                elif desc_type == 'patch':
                    has_patch = True
                    features_list.append('Patch')
                elif desc_type == 'refractor':
                    is_refractor = True
                    features_list.append('Refractor')
                elif desc_type == 'rpa':
                    is_autograph = True
                    is_rookie = True
                    has_patch = True
                    features_list.append('RPA')
                break

    # Extract parallel type
    for text, conf, bbox in entries:
        for cn_color, en_color in PARALLEL_MAPPINGS.items():
            if cn_color in text or cn_color.upper() in text.upper():
                parallel_type = f"Panini {en_color}"
                features_list.append(f"{en_color} Parallel")
                break
        if parallel_type:
            break

    # Create combined features string
    card_features = ', '.join(list(dict.fromkeys(features_list))) if features_list else None

    return {
        'is_autograph': is_autograph,
        'is_rookie': is_rookie,
        'has_patch': has_patch,
        'is_refractor': is_refractor,
        'parallel_type': parallel_type,
        'card_features': card_features
    }

def extract_team(ocr_entries):
    """Extract team name from OCR entries."""
    entries = filter_by_confidence(ocr_entries, CONFIDENCE_THRESHOLDS['general'])

    for text, conf, bbox in entries:
        for team_keyword, team_name in TEAM_MAPPINGS.items():
            if team_keyword in text or team_keyword.upper() in text.upper():
                return team_name

    return None

def extract_sold_status(ocr_entries):
    """Check if card is marked as sold."""
    for text, conf, bbox in ocr_entries:
        if re.search(PATTERNS['sold_status'], text):
            return 'Sold'
    return None

# %% [markdown]
# ## Load OCR Data

# %%
def load_ocr_data(directory="output/raw_ocr_output"):
    data = []

    for file_path in Path(directory).glob('*.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Get image name
        image_name = content.split('\n')[0].replace('OCR Results for: ', '').strip()

        # Parse each OCR entry
        for entry in content.split('----------------------------------------')[1:]:
            text_match = re.search(r'Text:\s*(.+)', entry)
            conf_match = re.search(r'Confidence:\s*([\d.]+)', entry)
            bbox_match = re.search(r'Bounding Box:\s*(\[.+\])', entry, re.DOTALL)

            if text_match and conf_match and bbox_match:
                data.append({
                    'image': image_name,
                    'text': text_match.group(1).strip(),
                    'confidence': float(conf_match.group(1)),
                    'bbox': bbox_match.group(1).strip()
                })

    return pd.DataFrame(data)

# %%
# Load the data
df = load_ocr_data()
print(f"Loaded {len(df)} OCR entries from {df['image'].nunique()} images")

# %% [markdown]
# ## Aggregation Function

# %%
def aggregate_card_features(df):
    """
    Aggregate per-image OCR data into one row per card.

    Input: Long-format DataFrame (multiple rows per image)
    Output: Wide-format DataFrame (one row per image)
    """
    results = []

    for image_name, group in df.groupby('image'):
        # Prepare OCR entries as list of tuples (text, confidence, bbox)
        ocr_entries = list(zip(group['text'], group['confidence'], group['bbox']))

        # Extract all features
        player_name, player_conf = extract_player_name(ocr_entries)
        card_year, card_series = extract_card_series(ocr_entries)
        serial_data = extract_serial_numbers(ocr_entries)
        grading_data = extract_grading(ocr_entries)
        end_time, product_id = extract_temporal(ocr_entries)
        descriptor_data = extract_descriptors(ocr_entries)
        team = extract_team(ocr_entries)
        sold_status = extract_sold_status(ocr_entries)

        # Extract prices and bids using existing functions
        price_cny = None
        price_usd = None
        shipping_price = None
        bid_times = None

        for text in group['text']:
            if not price_cny and not price_usd and not shipping_price:
                prices = extract_price(text)
                if prices[0] or prices[1] or prices[2]:
                    price_cny, price_usd, shipping_price = prices

            if not bid_times:
                bid_times = extract_bid_times(text)

        # Calculate average OCR confidence
        ocr_avg_confidence = group['confidence'].mean()

        # Quality flag: needs_review if confidence is low or many fields missing
        low_confidence = ocr_avg_confidence < 0.6
        missing_count = sum([
            player_name is None,
            card_series is None,
            price_cny is None and price_usd is None,
        ])
        needs_review = low_confidence or (missing_count >= 2)

        # Combine all features
        features = {
            'image': image_name,
            'product_id': product_id,

            # Player info
            'player_name': player_name,
            'team': team,

            # Card identity
            'card_year': card_year,
            'card_series': card_series,

            # Numbers
            'card_number': serial_data['card_number'],
            'serial_number': serial_data['serial_number'],
            'serial_max': serial_data['serial_max'],

            # Features (boolean flags)
            'is_rookie': descriptor_data['is_rookie'],
            'is_autograph': descriptor_data['is_autograph'],
            'has_patch': descriptor_data['has_patch'],
            'is_refractor': descriptor_data['is_refractor'],

            # Descriptors
            'parallel_type': descriptor_data['parallel_type'],
            'card_features': descriptor_data['card_features'],

            # Grading
            'grading_company': grading_data['grading_company'],
            'grade': grading_data['grade'],
            'grade_subtype': grading_data['grade_subtype'],

            # Auction data
            'price_cny': price_cny,
            'price_usd': price_usd,
            'shipping_price': shipping_price,
            'bid_times': bid_times,
            'end_time': end_time,
            'sold_status': sold_status,

            # Quality metadata
            'ocr_avg_confidence': ocr_avg_confidence,
            'needs_review': needs_review,
        }

        results.append(features)

    # Create DataFrame with proper dtypes
    output_df = pd.DataFrame(results)

    # Set proper nullable dtypes
    dtype_map = {
        'image': 'string',
        'product_id': 'string',
        'player_name': 'string',
        'team': 'string',
        'card_year': 'string',
        'card_series': 'string',
        'card_number': 'string',
        'serial_number': 'string',
        'serial_max': 'Int64',
        'is_rookie': 'boolean',
        'is_autograph': 'boolean',
        'has_patch': 'boolean',
        'is_refractor': 'boolean',
        'parallel_type': 'string',
        'card_features': 'string',
        'grading_company': 'string',
        'grade': 'Float64',
        'grade_subtype': 'string',
        'price_cny': 'Float64',
        'price_usd': 'Float64',
        'shipping_price': 'Float64',
        'bid_times': 'Int64',
        'end_time': 'string',
        'sold_status': 'string',
        'ocr_avg_confidence': 'Float64',
        'needs_review': 'boolean',
    }

    for col, dtype in dtype_map.items():
        if col in output_df.columns:
            output_df[col] = output_df[col].astype(dtype)

    return output_df

# %% [markdown]
# ## Extract Target Variables (Legacy - for reference)

# %%
def extract_price(text):
    """Extract price from text starting with 半 (CNY) or $
    Returns a tuple: (cny_price, usd_price, shipping_price)
    """
    cny_price = None
    usd_price = None
    shipping_price = None

    # Match shipping price: any text containing "运费"
    # If text contains "运费", extract any price from it
    if '运费' in text:
        # Try to extract price: 半10.00, 10.00, or $10.00
        price_match = re.search(r'[半$]?([\d,]+\.?\d*)', text)
        if price_match:
            shipping_price = float(price_match.group(1).replace(',', ''))
    else:
        # Match Chinese Yuan: 半10,750.00 or 半18.00
        # Only extract as card price if NOT a shipping entry
        cny_match = re.search(r'半([\d,]+\.?\d*)', text)
        if cny_match:
            cny_price = float(cny_match.group(1).replace(',', ''))

        # Match USD: $1,472.75
        usd_match = re.search(r'\$([\d,]+\.?\d*)', text)
        if usd_match:
            usd_price = float(usd_match.group(1).replace(',', ''))

    return cny_price, usd_price, shipping_price

def extract_bid_times(text):
    """Extract number of bids from text ending with 次竞价"""
    # Match pattern like: 念38 次竞价 or 38次竞价
    match = re.search(r'(\d+)\s*次竞价', text)
    if match:
        return int(match.group(1))
    return None

# %% [markdown]
# ## Aggregate Features Per Card

# %%
# Aggregate features per card image
print("\nAggregating features per card...")
cards_df = aggregate_card_features(df)
print(f"Extracted features for {len(cards_df)} cards")

# %% [markdown]
# ## Feature Extraction Results

# %%
# Show extraction statistics
print("\n" + "="*60)
print("FEATURE EXTRACTION SUMMARY")
print("="*60)
print(f"Total cards processed: {len(cards_df)}")
print(f"\nFeature completeness:")
print(f"  Player names found: {cards_df['player_name'].notna().sum()} ({cards_df['player_name'].notna().sum()/len(cards_df)*100:.1f}%)")
print(f"  Card series found: {cards_df['card_series'].notna().sum()} ({cards_df['card_series'].notna().sum()/len(cards_df)*100:.1f}%)")
print(f"  Team names found: {cards_df['team'].notna().sum()} ({cards_df['team'].notna().sum()/len(cards_df)*100:.1f}%)")
print(f"  CNY prices found: {cards_df['price_cny'].notna().sum()} ({cards_df['price_cny'].notna().sum()/len(cards_df)*100:.1f}%)")
print(f"  USD prices found: {cards_df['price_usd'].notna().sum()} ({cards_df['price_usd'].notna().sum()/len(cards_df)*100:.1f}%)")
print(f"  Shipping prices found: {cards_df['shipping_price'].notna().sum()} ({cards_df['shipping_price'].notna().sum()/len(cards_df)*100:.1f}%)")
print(f"  Bid times found: {cards_df['bid_times'].notna().sum()} ({cards_df['bid_times'].notna().sum()/len(cards_df)*100:.1f}%)")
print(f"  Serial numbers found: {cards_df['serial_number'].notna().sum()} ({cards_df['serial_number'].notna().sum()/len(cards_df)*100:.1f}%)")
print(f"  Graded cards found: {cards_df['grading_company'].notna().sum()} ({cards_df['grading_company'].notna().sum()/len(cards_df)*100:.1f}%)")
print(f"\nCard features:")
print(f"  Rookie cards: {cards_df['is_rookie'].sum()} ({cards_df['is_rookie'].sum()/len(cards_df)*100:.1f}%)")
print(f"  Autographs: {cards_df['is_autograph'].sum()} ({cards_df['is_autograph'].sum()/len(cards_df)*100:.1f}%)")
print(f"  Patches: {cards_df['has_patch'].sum()} ({cards_df['has_patch'].sum()/len(cards_df)*100:.1f}%)")
print(f"  Refractors: {cards_df['is_refractor'].sum()} ({cards_df['is_refractor'].sum()/len(cards_df)*100:.1f}%)")
print(f"  Parallels: {cards_df['parallel_type'].notna().sum()} ({cards_df['parallel_type'].notna().sum()/len(cards_df)*100:.1f}%)")
print(f"\nData quality:")
print(f"  Average OCR confidence: {cards_df['ocr_avg_confidence'].mean():.3f}")
print(f"  Cards needing review: {cards_df['needs_review'].sum()} ({cards_df['needs_review'].sum()/len(cards_df)*100:.1f}%)")
print("="*60)

# %%
# View sample extracted cards
print("\nSample extracted card data:")
display_cols = ['player_name', 'card_year', 'card_series', 'team', 'is_rookie', 'is_autograph',
                'parallel_type', 'grade', 'price_cny', 'bid_times']
cards_df[display_cols].head(10)

# %%
# View cards with player names
player_cards = cards_df[cards_df['player_name'].notna()][['image', 'player_name', 'team', 'card_year', 'card_series', 'card_features', 'price_cny']]
print(f"\nCards with player names ({len(player_cards)} cards):")
player_cards.head(15)

# %%
# View rookie cards
rookie_cards = cards_df[cards_df['is_rookie'] == True][['image', 'player_name', 'card_series', 'serial_number', 'is_autograph', 'has_patch', 'price_cny']]
print(f"\nRookie cards ({len(rookie_cards)} cards):")
rookie_cards

# %%
# View graded cards
graded_cards = cards_df[cards_df['grading_company'].notna()][['image', 'player_name', 'grading_company', 'grade', 'card_series', 'price_cny']]
print(f"\nGraded cards ({len(graded_cards)} cards):")
graded_cards

# %%
# View cards needing manual review
review_cards = cards_df[cards_df['needs_review'] == True][['image', 'player_name', 'card_series', 'price_cny', 'ocr_avg_confidence', 'needs_review']]
print(f"\nCards needing manual review ({len(review_cards)} cards):")
review_cards.head(20)

# %% [markdown]
# ## Data Quality Report

# %%
def generate_quality_report(cards_df):
    """Generate comprehensive data quality report."""
    report = {
        'total_cards': len(cards_df),
        'avg_ocr_confidence': cards_df['ocr_avg_confidence'].mean(),
        'cards_needing_review': cards_df['needs_review'].sum(),
        'completeness': {}
    }

    # Calculate completeness for each column
    for col in cards_df.columns:
        if col not in ['image', 'needs_review', 'ocr_avg_confidence']:
            non_null = cards_df[col].notna().sum()
            report['completeness'][col] = {
                'count': int(non_null),
                'percentage': round((non_null / len(cards_df)) * 100, 2)
            }

    return report

# Generate quality report
quality_report = generate_quality_report(cards_df)

# Create quality report DataFrame
quality_df = pd.DataFrame([
    {'field': k, 'count': v['count'], 'percentage': v['percentage']}
    for k, v in quality_report['completeness'].items()
]).sort_values('percentage', ascending=False)

print("\nData Quality Report:")
print(quality_df.to_string(index=False))

# %% [markdown]
# ## Export Data

# %%
# Save main card data to CSV
output_file = 'output/panini_cards_extracted.csv'
cards_df.to_csv(output_file, index=False)
print(f"\n✓ Saved extracted card data to {output_file}")
print(f"  Total cards: {len(cards_df)}")
print(f"  Total columns: {len(cards_df.columns)}")

# %%
# Save to JSON
json_file = 'output/panini_cards_extracted.json'
cards_df.to_json(json_file, orient='records', force_ascii=False, indent=2)
print(f"✓ Saved to {json_file}")

# %%
# Save quality report
quality_report_file = 'output/ocr_quality_report.csv'
quality_df.to_csv(quality_report_file, index=False)
print(f"✓ Saved quality report to {quality_report_file}")

# %%
# Save cards needing review
review_file = 'output/cards_needing_review.csv'
review_cards = cards_df[cards_df['needs_review'] == True]
review_cards.to_csv(review_file, index=False)
print(f"✓ Saved {len(review_cards)} cards needing review to {review_file}")

# %%
# Export summary statistics
print("\n" + "="*60)
print("EXPORT SUMMARY")
print("="*60)
print(f"Main data:        {output_file}")
print(f"JSON format:      {json_file}")
print(f"Quality report:   {quality_report_file}")
print(f"Review queue:     {review_file}")
print(f"\nTop features by completeness:")
for idx, row in quality_df.head(10).iterrows():
    print(f"  {row['field']:20s} {row['percentage']:5.1f}% ({row['count']:3d} cards)")
print("="*60)
