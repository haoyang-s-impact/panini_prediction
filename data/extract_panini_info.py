#!/usr/bin/env python3
"""
Panini Card Bidding Information Extractor
Extracts and categorizes text information from Chinese auction screenshots
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from paddleocr import PaddleOCR
from PIL import Image
import pandas as pd
import cv2
import numpy as np


class PaniniCardExtractor:
    """Extract and categorize information from Panini card auction screenshots"""

    def __init__(self, pics_folder: str = "pics", output_folder: str = "output"):
        """
        Initialize the extractor

        Args:
            pics_folder: Path to folder containing screenshots
            output_folder: Path to save output files
        """
        self.pics_folder = Path(pics_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)

        # Initialize PaddleOCR with Chinese and English support
        print("Initializing PaddleOCR (this may take a moment)...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='ch',  # Chinese
            use_gpu=False,
            show_log=False
        )

        self.raw_ocr_folder = self.output_folder / "raw_ocr_output"
        self.raw_ocr_folder.mkdir(exist_ok=True)

    def extract_text_from_image(self, image_path: Path) -> List[tuple]:
        """
        Extract text from a single image using PaddleOCR

        Args:
            image_path: Path to the image file

        Returns:
            List of tuples: (text, confidence, bounding_box)
        """
        try:
            result = self.ocr.ocr(str(image_path), cls=True)

            if not result or not result[0]:
                return []

            extracted_data = []
            for line in result[0]:
                bbox = line[0]  # Bounding box coordinates
                text = line[1][0]  # Extracted text
                confidence = line[1][1]  # Confidence score
                extracted_data.append((text, confidence, bbox))

            return extracted_data

        except Exception as e:
            print(f"Error extracting text from {image_path.name}: {e}")
            return []

    def save_raw_ocr(self, image_name: str, ocr_data: List[tuple]) -> None:
        """Save raw OCR output to text file"""
        output_file = self.raw_ocr_folder / f"{image_name}.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"OCR Results for: {image_name}\n")
            f.write("=" * 80 + "\n\n")

            for text, confidence, bbox in ocr_data:
                f.write(f"Text: {text}\n")
                f.write(f"Confidence: {confidence:.4f}\n")
                f.write(f"Bounding Box: {bbox}\n")
                f.write("-" * 40 + "\n")

    def categorize_text(self, ocr_data: List[tuple], image_name: str) -> Dict[str, Any]:
        """
        Categorize extracted text into structured fields

        Args:
            ocr_data: List of (text, confidence, bbox) tuples
            image_name: Name of the source image

        Returns:
            Dictionary with categorized information
        """
        # Combine all text for easier pattern matching
        all_text = " ".join([text for text, _, _ in ocr_data])
        text_lines = [text for text, _, _ in ocr_data]

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
            # Usually the first or largest price is the final price
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

        # Extract grading information (PSA, BGS, SGC)
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

        # Extract year/season (e.g., 2022-23, 2019-20)
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

        # Extract player name (look for capitalized English words, typically player names)
        # Common NBA players - this is a heuristic approach
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

        # If no player found by keyword, try to extract capitalized English words
        if "player_name" not in categorized["card_identity"]:
            # Look for sequences of capitalized words (likely player names)
            name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
            name_matches = re.findall(name_pattern, all_text)
            if name_matches:
                # Take the first match as potential player name
                categorized["card_identity"]["player_name"] = name_matches[0]

        # Calculate average OCR confidence
        if ocr_data:
            avg_confidence = sum(conf for _, conf, _ in ocr_data) / len(ocr_data)
            categorized["ocr_confidence"] = round(avg_confidence, 4)

        return categorized

    def process_all_images(self) -> List[Dict[str, Any]]:
        """
        Process all images in the pics folder

        Returns:
            List of categorized data dictionaries
        """
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            image_files.extend(self.pics_folder.glob(ext))

        if not image_files:
            print(f"No images found in {self.pics_folder}")
            return []

        print(f"Found {len(image_files)} images to process")

        all_results = []

        for idx, image_path in enumerate(image_files, 1):
            print(f"Processing {idx}/{len(image_files)}: {image_path.name}")

            # Extract text
            ocr_data = self.extract_text_from_image(image_path)

            if not ocr_data:
                print(f"  Warning: No text extracted from {image_path.name}")
                continue

            # Save raw OCR output
            self.save_raw_ocr(image_path.name, ocr_data)

            # Categorize text
            categorized = self.categorize_text(ocr_data, image_path.name)
            all_results.append(categorized)

            print(f"  Extracted {len(ocr_data)} text elements")

        return all_results

    def export_to_csv(self, data: List[Dict[str, Any]], filename: str = "panini_cards_data.csv") -> None:
        """Export categorized data to CSV"""
        output_path = self.output_folder / filename

        # Flatten nested dictionaries for CSV
        flattened_data = []
        for item in data:
            flat_item = {
                "source_file": item.get("source_file", ""),
                "player_name": item.get("card_identity", {}).get("player_name", ""),
                "season": item.get("card_identity", {}).get("season", ""),
                "series": item.get("card_identity", {}).get("series", ""),
                "card_type": item.get("card_identity", {}).get("card_type", ""),
                "grading_company": item.get("grading_info", {}).get("company", ""),
                "grade": item.get("grading_info", {}).get("grade", ""),
                "end_time": item.get("auction_data", {}).get("end_time", ""),
                "final_price_cny": item.get("auction_data", {}).get("final_price_cny", ""),
                "final_price_usd": item.get("auction_data", {}).get("final_price_usd", ""),
                "bid_count": item.get("auction_data", {}).get("bid_count", ""),
                "sold_status": item.get("auction_data", {}).get("sold_status", ""),
                "product_id": item.get("product_metadata", {}).get("product_id", ""),
                "shipping_cost": item.get("product_metadata", {}).get("shipping_cost", ""),
                "sport_type": item.get("product_metadata", {}).get("sport_type", ""),
                "ocr_confidence": item.get("ocr_confidence", "")
            }
            flattened_data.append(flat_item)

        df = pd.DataFrame(flattened_data)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')  # utf-8-sig for Excel compatibility
        print(f"\nCSV saved to: {output_path}")

    def export_to_json(self, data: List[Dict[str, Any]], filename: str = "panini_cards_data.json") -> None:
        """Export categorized data to JSON"""
        output_path = self.output_folder / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"JSON saved to: {output_path}")

    def export_to_excel(self, data: List[Dict[str, Any]], filename: str = "panini_cards_data.xlsx") -> None:
        """Export categorized data to Excel"""
        output_path = self.output_folder / filename

        # Flatten nested dictionaries for Excel
        flattened_data = []
        for item in data:
            flat_item = {
                "Source File": item.get("source_file", ""),
                "Player Name": item.get("card_identity", {}).get("player_name", ""),
                "Season": item.get("card_identity", {}).get("season", ""),
                "Series": item.get("card_identity", {}).get("series", ""),
                "Card Type": item.get("card_identity", {}).get("card_type", ""),
                "Grading Company": item.get("grading_info", {}).get("company", ""),
                "Grade": item.get("grading_info", {}).get("grade", ""),
                "End Time": item.get("auction_data", {}).get("end_time", ""),
                "Final Price (CNY)": item.get("auction_data", {}).get("final_price_cny", ""),
                "Final Price (USD)": item.get("auction_data", {}).get("final_price_usd", ""),
                "Bid Count": item.get("auction_data", {}).get("bid_count", ""),
                "Sold Status": item.get("auction_data", {}).get("sold_status", ""),
                "Product ID": item.get("product_metadata", {}).get("product_id", ""),
                "Shipping Cost": item.get("product_metadata", {}).get("shipping_cost", ""),
                "Sport Type": item.get("product_metadata", {}).get("sport_type", ""),
                "OCR Confidence": item.get("ocr_confidence", "")
            }
            flattened_data.append(flat_item)

        df = pd.DataFrame(flattened_data)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Panini Cards', index=False)

            # Auto-adjust column widths
            worksheet = writer.sheets['Panini Cards']
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(col)
                )
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)

        print(f"Excel saved to: {output_path}")


def main():
    """Main execution function"""
    print("=" * 80)
    print("Panini Card Bidding Information Extractor")
    print("=" * 80)
    print()

    # Initialize extractor
    extractor = PaniniCardExtractor(pics_folder="pics", output_folder="output")

    # Process all images
    print("\nStarting OCR extraction...")
    results = extractor.process_all_images()

    if not results:
        print("\nNo data extracted. Please check your images.")
        return

    print(f"\n{'=' * 80}")
    print(f"Successfully processed {len(results)} images")
    print(f"{'=' * 80}\n")

    # Export to all formats
    print("Exporting results...")
    extractor.export_to_csv(results)
    extractor.export_to_json(results)
    extractor.export_to_excel(results)

    print(f"\n{'=' * 80}")
    print("Extraction complete!")
    print(f"{'=' * 80}")
    print(f"\nOutput files saved in: {extractor.output_folder}")
    print(f"Raw OCR text saved in: {extractor.raw_ocr_folder}")
    print()

    # Print summary statistics
    print("Summary Statistics:")
    print("-" * 40)

    players = [r.get("card_identity", {}).get("player_name", "") for r in results if r.get("card_identity", {}).get("player_name")]
    if players:
        print(f"Unique players found: {len(set(players))}")

    prices = [r.get("auction_data", {}).get("final_price_cny", 0) for r in results if r.get("auction_data", {}).get("final_price_cny")]
    if prices:
        print(f"Price range: ¥{min(prices):,.2f} - ¥{max(prices):,.2f}")
        print(f"Average price: ¥{sum(prices)/len(prices):,.2f}")

    bids = [r.get("auction_data", {}).get("bid_count", 0) for r in results if r.get("auction_data", {}).get("bid_count")]
    if bids:
        print(f"Bid count range: {min(bids)} - {max(bids)}")
        print(f"Average bids: {sum(bids)/len(bids):.1f}")

    print()


if __name__ == "__main__":
    main()
