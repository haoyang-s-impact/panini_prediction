# Implementation Plan: Comprehensive Feature Extraction for Panini Card OCR Data

## Overview

Transform the current basic OCR ETL pipeline into a comprehensive feature extraction system that extracts player names, card metadata, grades, serial numbers, and descriptors from Panini trading card images. The system will handle variable OCR data quality, fuzzy player name matching, and convert sparse OCR data into wide-format structured data suitable for ML.

## Critical Files

- **`panini_card_ocr_etl.py`** - Main ETL script to modify (lines 50-125)
- **`data/nba_players.py`** - NEW file with player names and mappings (to create)
- **`requirements.txt`** - Add `rapidfuzz>=3.0.0` dependency

## Implementation Steps

### 1. Setup Infrastructure

**Create NBA player data module** (`data/nba_players.py`):
- Hardcode list of top 200-300 NBA players (2018-2025 focus)
- Add Chinese-to-English player name mappings for ~50 major players
- Add NBA team name mappings (Chinese/English → full team names)
- Store parallel color mappings ("银折" → "Silver", etc.)

**Update dependencies**:
- Add `rapidfuzz>=3.0.0` to `requirements.txt` for fuzzy player name matching

### 2. Add Configuration Section

**In `panini_card_ocr_etl.py`, add after imports**:
- Regex patterns for: player names, card series, serial numbers, grades, end times, product IDs, descriptors
- Spatial zone definitions (Y-coordinate ranges for position-based extraction)
- Confidence thresholds per feature type (player: 0.50, price: 0.70, ID: 0.85, etc.)
- Import NBA player data from new module

### 3. Implement Pattern-Based Extractors

**Add extraction functions** (one per feature type):

1. `extract_player_name(ocr_entries)`:
   - Check Chinese name mapping first (exact match)
   - Apply regex to find English names
   - Fuzzy match against NBA_PLAYERS list using `rapidfuzz` (threshold: 85+)
   - Return highest confidence match
   - Returns: (player_name, confidence)

2. `extract_card_series(ocr_entries)`:
   - Regex pattern: `(20\d{2}[-/]?\d{2})\s+(Panini\s+[A-Za-z\s]+)`
   - Extract year and series separately
   - Keep full "Panini + Series" (e.g., "Panini Prizm")
   - Returns: (card_year, card_series)

3. `extract_serial_numbers(ocr_entries)`:
   - Patterns: `(\d+)?/(\d+)编?` for serials, `#(\d+)` for card numbers
   - Parse into structured format: "05/25" and max=25
   - Returns: dict with serial_number, serial_max, card_number

4. `extract_grading(ocr_entries)`:
   - Patterns for PSA, BGS, SGC, CSG
   - Extract grade value (float) and subtype (黑标/金标)
   - Returns: dict with grading_company, grade, grade_subtype

5. `extract_temporal(ocr_entries)`:
   - Extract 结束时间 (end time) and 商品编号 (product ID)
   - Parse datetime for validation
   - Returns: (end_time, product_id)

6. `extract_descriptors(ocr_entries)`:
   - Search for: autograph (签字), rookie (RC/新秀), patch (球衣), refractor (折射)
   - Extract parallel type (银折/紫折/红碎冰 → "Panini Silver"/"Panini Purple"/"Panini Red Ice")
   - Create boolean flags: is_autograph, is_rookie, has_patch
   - Returns: dict with all descriptor fields

7. `extract_team(ocr_entries)`:
   - Map Chinese/English team names to full names
   - "勇士"/"WARRIORS" → "Golden State Warriors"
   - Returns: team name string

### 4. Add Position-Based Helpers

**Spatial zone functions** (fallback when patterns fail):

- `get_text_zone(bbox)`: Determine spatial zone from Y-coordinate
  - Zones: timestamp (0-250), card_image (250-700), player_info (700-1250), description (1250-1600), pricing (1600-1900), metadata (2000-2600)

- `filter_by_zone(ocr_data, zone)`: Get text entries in specific spatial area

**Use as fallback** when pattern matching returns None

### 5. Implement Aggregation Function

**`aggregate_card_features(df)` function**:
- Group DataFrame by `image` (one row per card)
- For each image, extract OCR entries list: (text, confidence, bbox)
- Call all extractor functions
- Combine results into single dict per image
- Handle missing values gracefully (None/NaN)
- Calculate average OCR confidence
- Return wide-format DataFrame with schema:

```python
Columns:
- image, product_id
- player_name, team, card_year, card_series, card_number
- is_rookie, is_autograph, has_patch, parallel_type
- serial_number, serial_max, card_features (combined descriptors)
- grading_company, grade, grade_subtype
- price_cny, price_usd, shipping_price, bid_times
- end_time, sold_status, ocr_avg_confidence
```

### 6. Integrate with Existing Pipeline

**Modify main execution flow** (lines 77-100):
- Keep existing `extract_price()` and `extract_bid_times()` functions
- After loading OCR data, call `aggregate_card_features(df)` to get wide-format output
- This replaces the current per-text-entry extraction with per-image aggregation
- Update print statements to show extraction counts for new features
- Update display cells to show extracted player names, card series, grades, etc.

### 7. Update Data Export

**Modify export sections** (lines 117-125):
- Ensure new columns are included in CSV/JSON exports
- Add data quality report showing completeness % per column
- Optional: Add Excel export with formatted output

### 8. Testing & Validation

**Test on sample files**:
- Use 5 diverse OCR files for unit testing each extractor
- Sample files: Stephen Curry (autograph + silver), Ja Morant (rookie + patch), Shai Gilgeous-Alexander (graded PSA10)
- Validate output format and missing data handling
- Check fuzzy matching works for OCR errors ("STEPHENCURR" → "Stephen Curry")

**Full dataset processing**:
- Run on all 97 OCR files
- Generate data quality report (completeness metrics)
- Manually validate 10-20 random samples against original images

## DataFrame Schema (Output)

Wide-format with nullable types:

```
Identification: image (str), product_id (str)
Player: player_name (str), team (str)
Card Identity: card_year (str), card_series (str), card_number (str)
Features: is_rookie (bool), is_autograph (bool), has_patch (bool),
          parallel_type (str), serial_number (str), serial_max (Int64),
          card_features (str - comma-separated)
Grading: grading_company (str), grade (Float64), grade_subtype (str)
Auction: price_cny (Float64), price_usd (Float64), shipping_price (Float64),
         bid_times (Int64), end_time (str), sold_status (str)
Metadata: ocr_avg_confidence (Float64)
```

## Key Design Decisions

1. **Hybrid Extraction**: Pattern-based (primary) + position-based (fallback)
2. **Fuzzy Matching**: Use `rapidfuzz` with 85+ threshold for player names
3. **Hardcoded NBA Data**: Top 200-300 players (avoids API dependency, covers 90%+ of cards)
4. **Confidence Filtering**: Different thresholds per feature type (player: 0.50, ID: 0.85)
5. **Panini Branding**: Store parallels as "Panini Silver" not just "Silver"
6. **Wide Format**: One row per card (aggregated from multiple OCR entries)
7. **Graceful Degradation**: All fields optional (NaN if not found)

## Expected Challenges

- **OCR Quality**: Some text has confidence <0.5 (handle with thresholds)
- **Player Name Variations**: Need fuzzy matching for OCR errors
- **Missing Features**: Not all cards have all fields (use nullable types)
- **Language Mixing**: Chinese + English text (dual pattern sets)

## Success Metrics

- **Completeness**: 90%+ cards have player_name, card_series extracted
- **Accuracy**: 95%+ player name matches verified correct (manual sample)
- **Coverage**: 80%+ cards have at least 5 feature fields populated
- **Processing**: Handle all 97 OCR files without errors
