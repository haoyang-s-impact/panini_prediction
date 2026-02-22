#!/usr/bin/env python3
"""
Panini Card Bidding Information Extractor - Interactive Version
This script supports cell-by-cell execution in VS Code, PyCharm, or Spyder

Features:
1. Text Extraction - Extract all text using EasyOCR (optimized for Chinese & English)
2. Text Categorization - Parse and categorize into structured fields
3. Data Export - Export to CSV, JSON, and Excel formats
4. Data Visualization - View statistics and insights

To run cell-by-cell:
- VS Code: Use # %% to mark cells, then click "Run Cell" or use Shift+Enter
- PyCharm: Same as VS Code
- Spyder: Use # %% or # <codecell> markers
- Jupyter: Can also run this file directly in Jupyter
"""

# %%
# =============================================================================
# 1. SETUP AND INSTALLATION
# =============================================================================
# Uncomment the line below to install packages (run once)
!pip install -q easyocr Pillow opencv-python pandas openpyxl python-dateutil numpy matplotlib seaborn

# %%
# =============================================================================
# 2. IMPORT LIBRARIES
# =============================================================================
import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import easyocr
from PIL import Image
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 50)
plt.style.use('default')

print("✓ All libraries imported successfully!")

# %%
# =============================================================================
# 3. CONFIGURATION
# =============================================================================
# Set your folder paths here
PICS_FOLDER = "pics"  # Folder containing your screenshots
OUTPUT_FOLDER = "output"  # Where to save results

# Create output folder if it doesn't exist
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
Path(OUTPUT_FOLDER, "raw_ocr_output").mkdir(exist_ok=True)

print(f"✓ Configuration set:")
print(f"  - Input folder: {PICS_FOLDER}")
print(f"  - Output folder: {OUTPUT_FOLDER}")

# %%
# =============================================================================
# 4. INITIALIZE OCR ENGINE
# =============================================================================
# This will load the EasyOCR model with Chinese and English support
# Note: First run will download language models (~200MB). This is normal and only happens once.

print("Initializing EasyOCR (first run will download models, ~200MB)...")
print("This may take 1-2 minutes...")

reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)

print("✓ OCR engine initialized successfully!")

# %%
# =============================================================================
# 5. DEFINE HELPER FUNCTIONS
# =============================================================================

def extract_text_from_image(image_path: Path, reader) -> List[tuple]:
    """
    Extract text from a single image using EasyOCR

    Returns: List of (text, confidence, bounding_box) tuples
    """
    try:
        # Read image
        result = reader.readtext(str(image_path))

        if not result:
            return []

        # EasyOCR returns: (bbox, text, confidence)
        extracted_data = [(text, confidence, bbox) for bbox, text, confidence in result]

        return extracted_data

    except Exception as e:
        print(f"Error extracting text from {image_path.name}: {e}")
        return []


def save_raw_ocr(image_name: str, ocr_data: List[tuple], output_folder: str) -> None:
    """Save raw OCR output to text file"""
    raw_folder = Path(output_folder) / "raw_ocr_output"
    output_file = raw_folder / f"{image_name}.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"OCR Results for: {image_name}\n")
        f.write("=" * 80 + "\n\n")

        for text, confidence, bbox in ocr_data:
            f.write(f"Text: {text}\n")
            f.write(f"Confidence: {confidence:.4f}\n")
            f.write(f"Bounding Box: {bbox}\n")
            f.write("-" * 40 + "\n")


def categorize_text(ocr_data: List[tuple], image_name: str) -> Dict[str, Any]:
    """
    Categorize extracted text into structured fields
    """
    # Combine all text for easier pattern matching
    all_text = " ".join([text for text, _, _ in ocr_data])

    categorized = {
        "source_file": image_name,
        "card_identity": {},
        "grading_info": {},
        "auction_data": {},
        "product_metadata": {},
        "raw_text": all_text
    }

    # Extract auction end time
    end_time_pattern = r'结束时间[:：]\s*(\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2})'
    end_time_match = re.search(end_time_pattern, all_text)
    if end_time_match:
        categorized["auction_data"]["end_time"] = end_time_match.group(1)

    # Extract price (CNY)
    price_pattern = r'¥\s*([\d,]+\.?\d*)'
    price_matches = re.findall(price_pattern, all_text)
    if price_matches:
        prices = [float(p.replace(',', '')) for p in price_matches]
        categorized["auction_data"]["final_price_cny"] = max(prices)

    # Extract USD price
    usd_pattern = r'\$\s*([\d,]+\.?\d*)'
    usd_match = re.search(usd_pattern, all_text)
    if usd_match:
        categorized["auction_data"]["final_price_usd"] = float(usd_match.group(1).replace(',', ''))

    # Extract bid count
    bid_pattern = r'(\d+)\s*次竞价'
    bid_match = re.search(bid_pattern, all_text)
    if bid_match:
        categorized["auction_data"]["bid_count"] = int(bid_match.group(1))

    # Extract product ID
    product_id_pattern = r'商品编号[:：]\s*(\d+)'
    product_id_match = re.search(product_id_pattern, all_text)
    if product_id_match:
        categorized["product_metadata"]["product_id"] = product_id_match.group(1)

    # Extract shipping cost
    shipping_pattern = r'运费[:：]\s*¥?\s*(\d+\.?\d*)'
    shipping_match = re.search(shipping_pattern, all_text)
    if shipping_match:
        categorized["product_metadata"]["shipping_cost"] = float(shipping_match.group(1))

    # Extract sport type
    if '篮球' in all_text or 'Basketball' in all_text:
        categorized["product_metadata"]["sport_type"] = "Basketball"

    # Check if sold
    if '商品已售出' in all_text or '已售出' in all_text:
        categorized["auction_data"]["sold_status"] = "Sold"

    # Extract grading information
    grading_patterns = [
        (r'PSA\s*(\d+\.?\d*)', 'PSA'),
        (r'BGS\s*(\d+\.?\d*)', 'BGS'),
        (r'SGC\s*(\d+\.?\d*)', 'SGC')
    ]

    for pattern, company in grading_patterns:
        match = re.search(pattern, all_text, re.IGNORECASE)
        if match:
            categorized["grading_info"]["company"] = company
            categorized["grading_info"]["grade"] = match.group(1)
            break

    # Extract year/season
    year_pattern = r'(20\d{2}[-/]\d{2})'
    year_match = re.search(year_pattern, all_text)
    if year_match:
        categorized["card_identity"]["season"] = year_match.group(1)

    # Extract Panini series
    if 'Prizm' in all_text:
        categorized["card_identity"]["series"] = "Panini Prizm"
    elif 'Select' in all_text:
        categorized["card_identity"]["series"] = "Panini Select"
    elif 'Optic' in all_text:
        categorized["card_identity"]["series"] = "Panini Optic"
    elif 'Panini' in all_text:
        categorized["card_identity"]["series"] = "Panini"

    # Extract card type markers
    card_types = []
    if 'RC' in all_text or 'Rookie' in all_text or '新秀' in all_text:
        card_types.append("Rookie Card")
    if 'Auto' in all_text or '签名' in all_text:
        card_types.append("Autograph")
    if 'Patch' in all_text or '球衣' in all_text:
        card_types.append("Patch")
    if card_types:
        categorized["card_identity"]["card_type"] = ", ".join(card_types)

    # Extract player name - common NBA players
    player_keywords = [
        'Stephen Curry', 'Curry', 'LeBron James', 'LeBron', 'Giannis',
        'Luka Doncic', 'Luka', 'Ja Morant', 'Morant', 'Zion Williamson',
        'Zion', 'Trae Young', 'Damian Lillard', 'Kevin Durant', 'Durant',
        'Shai Gilgeous-Alexander', 'Shai', 'Anthony Edwards', 'Lamelo Ball',
        'Jayson Tatum', 'Tatum', 'Joel Embiid', 'Embiid', 'Nikola Jokic',
        'Jokic', 'Kawhi Leonard', 'Jimmy Butler', 'Devin Booker'
    ]

    for player in player_keywords:
        if player in all_text:
            categorized["card_identity"]["player_name"] = player
            break

    # If no player found by keyword, try capitalized English words
    if "player_name" not in categorized["card_identity"]:
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        name_matches = re.findall(name_pattern, all_text)
        if name_matches:
            categorized["card_identity"]["player_name"] = name_matches[0]

    # Calculate average OCR confidence
    if ocr_data:
        avg_confidence = sum(conf for _, conf, _ in ocr_data) / len(ocr_data)
        categorized["ocr_confidence"] = round(avg_confidence, 4)

    return categorized


print("✓ Helper functions defined successfully!")

# %%
# =============================================================================
# 6. PREVIEW SAMPLE IMAGE (OPTIONAL)
# =============================================================================
# Get list of images
pics_path = Path(PICS_FOLDER)
image_files = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
    image_files.extend(pics_path.glob(ext))

print(f"Found {len(image_files)} images in {PICS_FOLDER}")

# Display first image
if image_files:
    sample_image = Image.open(image_files[0])
    plt.figure(figsize=(10, 12))
    plt.imshow(sample_image)
    plt.title(f"Sample Screenshot: {image_files[0].name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print(f"Dimensions: {sample_image.size}")

# %%
# =============================================================================
# 7. TEST OCR ON SINGLE IMAGE
# =============================================================================
if image_files:
    print(f"Testing OCR on: {image_files[0].name}")
    print("=" * 80)

    # Extract text
    test_ocr_data = extract_text_from_image(image_files[0], reader)

    print(f"\n✓ Extracted {len(test_ocr_data)} text elements\n")

    # Display first 10 extracted texts
    print("Sample extracted text:")
    print("-" * 80)
    for i, (text, conf, _) in enumerate(test_ocr_data[:10], 1):
        print(f"{i:2d}. {text:40s} (confidence: {conf:.3f})")

    if len(test_ocr_data) > 10:
        print(f"... and {len(test_ocr_data) - 10} more")

    # Categorize the text
    print("\n" + "=" * 80)
    print("Categorized Information:")
    print("=" * 80)

    test_categorized = categorize_text(test_ocr_data, image_files[0].name)

    # Display categorized data
    for category, data in test_categorized.items():
        if category != "raw_text" and data:
            print(f"\n{category.upper().replace('_', ' ')}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {data}")

# %%
# =============================================================================
# 8. PROCESS ALL IMAGES
# =============================================================================
# This may take several minutes depending on the number of images
# Estimated time: ~2-3 seconds per image (total ~3-5 minutes for 97 images)

print(f"Processing {len(image_files)} images...")
print("=" * 80)

all_results = []

for idx, image_path in enumerate(image_files, 1):
    print(f"[{idx}/{len(image_files)}] Processing: {image_path.name}", end=" ... ")

    # Extract text
    ocr_data = extract_text_from_image(image_path, reader)

    if not ocr_data:
        print("⚠ No text extracted")
        continue

    # Save raw OCR output
    save_raw_ocr(image_path.name, ocr_data, OUTPUT_FOLDER)

    # Categorize text
    categorized = categorize_text(ocr_data, image_path.name)
    all_results.append(categorized)

    print(f"✓ Extracted {len(ocr_data)} elements")

print("\n" + "=" * 80)
print(f"✓ Processing complete! Successfully processed {len(all_results)} images")
print("=" * 80)

# %%
# =============================================================================
# 9. CONVERT TO DATAFRAME
# =============================================================================
# Flatten nested dictionaries for DataFrame
flattened_data = []
for item in all_results:
    flat_item = {
        "source_file": item.get("source_file", ""),
        "player_name": item.get("card_identity", {}).get("player_name", ""),
        "season": item.get("card_identity", {}).get("season", ""),
        "series": item.get("card_identity", {}).get("series", ""),
        "card_type": item.get("card_identity", {}).get("card_type", ""),
        "grading_company": item.get("grading_info", {}).get("company", ""),
        "grade": item.get("grading_info", {}).get("grade", ""),
        "end_time": item.get("auction_data", {}).get("end_time", ""),
        "final_price_cny": item.get("auction_data", {}).get("final_price_cny", None),
        "final_price_usd": item.get("auction_data", {}).get("final_price_usd", None),
        "bid_count": item.get("auction_data", {}).get("bid_count", None),
        "sold_status": item.get("auction_data", {}).get("sold_status", ""),
        "product_id": item.get("product_metadata", {}).get("product_id", ""),
        "shipping_cost": item.get("product_metadata", {}).get("shipping_cost", None),
        "sport_type": item.get("product_metadata", {}).get("sport_type", ""),
        "ocr_confidence": item.get("ocr_confidence", None)
    }
    flattened_data.append(flat_item)

df = pd.DataFrame(flattened_data)

print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
print("\nFirst 5 rows:")
print(df.head())

# %%
# =============================================================================
# 10. DATA SUMMARY AND STATISTICS
# =============================================================================
print("=" * 80)
print("DATA SUMMARY")
print("=" * 80)

print(f"\nTotal records: {len(df)}")
print(f"\nData completeness:")
print("-" * 40)
for col in df.columns:
    non_null = df[col].notna().sum()
    pct = (non_null / len(df)) * 100 if len(df) > 0 else 0
    print(f"{col:20s}: {non_null:3d}/{len(df)} ({pct:5.1f}%)")

# Price statistics
if df['final_price_cny'].notna().any():
    print(f"\nPrice Statistics (CNY):")
    print("-" * 40)
    print(f"Minimum:  ¥{df['final_price_cny'].min():,.2f}")
    print(f"Maximum:  ¥{df['final_price_cny'].max():,.2f}")
    print(f"Mean:     ¥{df['final_price_cny'].mean():,.2f}")
    print(f"Median:   ¥{df['final_price_cny'].median():,.2f}")

# Bid statistics
if df['bid_count'].notna().any():
    print(f"\nBid Count Statistics:")
    print("-" * 40)
    print(f"Minimum:  {int(df['bid_count'].min())}")
    print(f"Maximum:  {int(df['bid_count'].max())}")
    print(f"Mean:     {df['bid_count'].mean():.1f}")
    print(f"Median:   {df['bid_count'].median():.0f}")

# Top players
if df['player_name'].notna().any():
    print(f"\nTop 10 Players (by card count):")
    print("-" * 40)
    player_counts = df['player_name'].value_counts().head(10)
    for player, count in player_counts.items():
        print(f"{player:30s}: {count} cards")

# Grading distribution
if df['grading_company'].notna().any():
    print(f"\nGrading Company Distribution:")
    print("-" * 40)
    grading_counts = df['grading_company'].value_counts()
    for company, count in grading_counts.items():
        pct = (count / len(df)) * 100
        print(f"{company:10s}: {count} cards ({pct:.1f}%)")

# %%
# =============================================================================
# 11. DATA VISUALIZATIONS
# =============================================================================
# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Price distribution
if df['final_price_cny'].notna().any():
    axes[0, 0].hist(df['final_price_cny'].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Price Distribution (CNY)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Price (¥)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Bid count distribution
if df['bid_count'].notna().any():
    axes[0, 1].hist(df['bid_count'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_title('Bid Count Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Bids')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Top players by count
if df['player_name'].notna().any():
    top_players = df['player_name'].value_counts().head(10)
    axes[1, 0].barh(range(len(top_players)), top_players.values, color='steelblue')
    axes[1, 0].set_yticks(range(len(top_players)))
    axes[1, 0].set_yticklabels(top_players.index)
    axes[1, 0].set_title('Top 10 Players by Card Count', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Number of Cards')
    axes[1, 0].grid(axis='x', alpha=0.3)
    axes[1, 0].invert_yaxis()

# 4. Grading distribution
if df['grading_company'].notna().any():
    grading_counts = df['grading_company'].value_counts()
    colors = plt.cm.Set3(range(len(grading_counts)))
    axes[1, 1].pie(grading_counts.values, labels=grading_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors)
    axes[1, 1].set_title('Grading Company Distribution', fontsize=14, fontweight='bold')
else:
    axes[1, 1].text(0.5, 0.5, 'No grading data available',
                    ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/data_visualization.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ Visualization saved to {OUTPUT_FOLDER}/data_visualization.png")

# %%
# =============================================================================
# 12. EXPORT DATA
# =============================================================================
print("Exporting data...")
print("=" * 80)

# 1. Export to CSV
csv_path = Path(OUTPUT_FOLDER) / "panini_cards_data.csv"
df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig for Excel compatibility
print(f"✓ CSV exported to: {csv_path}")

# 2. Export to JSON
json_path = Path(OUTPUT_FOLDER) / "panini_cards_data.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"✓ JSON exported to: {json_path}")

# 3. Export to Excel with formatting
excel_path = Path(OUTPUT_FOLDER) / "panini_cards_data.xlsx"
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Panini Cards', index=False)

    # Auto-adjust column widths
    worksheet = writer.sheets['Panini Cards']
    for idx, col in enumerate(df.columns):
        max_length = max(
            df[col].astype(str).apply(len).max(),
            len(col)
        )
        col_letter = chr(65 + idx) if idx < 26 else chr(65 + idx // 26 - 1) + chr(65 + idx % 26)
        worksheet.column_dimensions[col_letter].width = min(max_length + 2, 50)

print(f"✓ Excel exported to: {excel_path}")

print("\n" + "=" * 80)
print("EXPORT COMPLETE!")
print("=" * 80)
print(f"\nAll files saved in: {OUTPUT_FOLDER}/")
print(f"  - CSV: panini_cards_data.csv")
print(f"  - JSON: panini_cards_data.json")
print(f"  - Excel: panini_cards_data.xlsx")
print(f"  - Visualization: data_visualization.png")
print(f"  - Raw OCR output: raw_ocr_output/ folder")

# %%
# =============================================================================
# 13. ADVANCED ANALYSIS - PRICE VS BID CORRELATION (OPTIONAL)
# =============================================================================
# Create correlation plot
if df['final_price_cny'].notna().any() and df['bid_count'].notna().any():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out null values
    plot_df = df[['final_price_cny', 'bid_count']].dropna()

    if len(plot_df) > 0:
        ax.scatter(plot_df['bid_count'], plot_df['final_price_cny'], alpha=0.6, s=100)
        ax.set_xlabel('Number of Bids', fontsize=12)
        ax.set_ylabel('Final Price (CNY)', fontsize=12)
        ax.set_title('Bid Count vs Final Price', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        # Add trend line
        z = np.polyfit(plot_df['bid_count'], plot_df['final_price_cny'], 1)
        p = np.poly1d(z)
        ax.plot(plot_df['bid_count'], p(plot_df['bid_count']), "r--", alpha=0.8, linewidth=2)

        # Calculate correlation
        corr = plot_df['bid_count'].corr(plot_df['final_price_cny'])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_FOLDER}/price_vs_bids.png", dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\n✓ Correlation analysis saved to {OUTPUT_FOLDER}/price_vs_bids.png")
        print(f"Correlation coefficient: {corr:.3f}")
    else:
        print("Not enough data for correlation analysis")

# %%
# =============================================================================
# 14. VIEW SAMPLE RESULTS
# =============================================================================
# Display sample results with most complete data
print("Sample results with most complete data:")
print("=" * 80)

# Count non-null values per row
df['completeness'] = df.notna().sum(axis=1)
top_complete = df.nlargest(min(5, len(df)), 'completeness')

for idx, row in top_complete.iterrows():
    print(f"\nRecord {idx + 1}:")
    print("-" * 80)
    print(f"File: {row['source_file']}")
    if row['player_name']:
        print(f"Player: {row['player_name']}")
    if row['season']:
        print(f"Season: {row['season']}")
    if row['series']:
        print(f"Series: {row['series']}")
    if row['card_type']:
        print(f"Type: {row['card_type']}")
    if row['grading_company']:
        print(f"Grade: {row['grading_company']} {row['grade']}")
    if pd.notna(row['final_price_cny']):
        print(f"Price: ¥{row['final_price_cny']:,.2f}")
    if pd.notna(row['bid_count']):
        print(f"Bids: {row['bid_count']:.0f}")
    if row['end_time']:
        print(f"End Time: {row['end_time']}")

df = df.drop('completeness', axis=1)  # Remove temporary column

# %%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
This script has successfully:
1. ✓ Extracted text from all Panini card auction screenshots using EasyOCR
2. ✓ Categorized the text into structured fields (card info, auction data, grading, etc.)
3. ✓ Exported data to CSV, JSON, and Excel formats
4. ✓ Generated visualizations and statistical analysis
5. ✓ Saved raw OCR output for verification

All output files are available in the 'output/' folder.

TROUBLESHOOTING:
- Import errors: Make sure all packages are installed (see cell 1)
- Slow processing: EasyOCR takes ~2-3 seconds per image, which is normal
- Poor extraction: Try adjusting image preprocessing in the extract function
- Missing data: Check the raw OCR output files to see what text was extracted
""")

# %%
