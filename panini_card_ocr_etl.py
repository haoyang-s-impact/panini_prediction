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

# %%
# Extract target variables
df[['price_cny', 'price_usd', 'shipping_price']] = df['text'].apply(lambda x: pd.Series(extract_price(x)))
df['bid_times'] = df['text'].apply(extract_bid_times)

# Show extraction results
print(f"CNY prices found: {df['price_cny'].notna().sum()}")
print(f"USD prices found: {df['price_usd'].notna().sum()}")
print(f"Shipping prices found: {df['shipping_price'].notna().sum()}")
print(f"Bid times found: {df['bid_times'].notna().sum()}")

# %%
# View extracted prices
price_df = df[(df['price_cny'].notna()) | (df['price_usd'].notna())][['image', 'text', 'price_cny', 'price_usd', 'confidence']]
print(f"\nExtracted prices ({len(price_df)} entries):")
price_df

# %%
# View extracted shipping prices
shipping_df = df[df['shipping_price'].notna()][['image', 'text', 'shipping_price', 'confidence']]
print(f"\nExtracted shipping prices ({len(shipping_df)} entries):")
shipping_df

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
