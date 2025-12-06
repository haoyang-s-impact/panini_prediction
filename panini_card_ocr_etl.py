# %% [markdown]
# # Panini Card OCR Data ETL
# Load and process OCR text extraction results for machine learning

# %%
import os
import re
import pandas as pd
from pathlib import Path

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
# ## Extract Target Variables

# %%
def extract_price(text):
    """Extract price from text starting with 半 (CNY) or $"""
    # Match Chinese Yuan: 半10,750.00 or 半18.00
    cny_match = re.search(r'半([\d,]+\.?\d*)', text)
    if cny_match:
        return float(cny_match.group(1).replace(',', ''))

    # Match USD: $1,472.75
    usd_match = re.search(r'\$([\d,]+\.?\d*)', text)
    if usd_match:
        return float(usd_match.group(1).replace(',', ''))

    return None

def extract_bid_times(text):
    """Extract number of bids from text ending with 次竞价"""
    # Match pattern like: 念38 次竞价 or 38次竞价
    match = re.search(r'(\d+)\s*次竞价', text)
    if match:
        return int(match.group(1))
    return None

# %%
# Extract target variables
df['price'] = df['text'].apply(extract_price)
df['bid_times'] = df['text'].apply(extract_bid_times)

# Show extraction results
print(f"Prices found: {df['price'].notna().sum()}")
print(f"Bid times found: {df['bid_times'].notna().sum()}")

# %%
# View extracted prices
price_df = df[df['price'].notna()][['image', 'text', 'price', 'confidence']]
print(f"\nExtracted prices ({len(price_df)} entries):")
price_df

# %%
# View extracted bid times
bid_df = df[df['bid_times'].notna()][['image', 'text', 'bid_times', 'confidence']]
print(f"\nExtracted bid times ({len(bid_df)} entries):")
bid_df

# %% [markdown]
# ## Explore the Data

# %%
# Show basic statistics
df.info()

# %%
# Confidence statistics
df['confidence'].describe()

# %%
# View first few entries
df.head(10)

# %%
# Filter high confidence entries (>0.8)
high_conf = df[df['confidence'] > 0.8]
print(f"High confidence entries: {len(high_conf)} ({len(high_conf)/len(df)*100:.1f}%)")
high_conf.head()

# %% [markdown]
# ## Export Data

# %%
# Save to CSV
df.to_csv('output/ocr_data.csv', index=False)
print("Saved to output/ocr_data.csv")

# %%
# Save to JSON
df.to_json('output/ocr_data.json', orient='records', force_ascii=False, indent=2)
print("Saved to output/ocr_data.json")
