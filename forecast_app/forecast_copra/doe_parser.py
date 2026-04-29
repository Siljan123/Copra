# doe_parser.py — Compatible version (returns both 'price' and 'average')

import re
import io
from decimal import Decimal
import pdfplumber


SURIGAO_MUNICIPALITIES = [
    "Bislig City", "Tandag City", "Barobo", "Carmen", "Lianga",
    "Lanuza", "Lingig", "Tagbina", "Hinaluan", "Hinatuan",
    "Tago", "Cagwait", "Cantilan", "Carrascal", "Cortes",
    "Madrid", "Marihatag", "San Agustin", "San Miguel",
    "Sta. Maria", "Santa Maria", "Bayabas",
]

PROVINCE_MARKERS = ["surigao del sur"]
NEXT_PROVINCE_MARKERS = [
    "surigao del norte", "agusan", "davao", "bukidnon",
    "misamis", "zamboanga", "lanao", "cotabato",
    "sarangani", "south cotabato", "sultan kudarat",
]


def _extract_diesel_price(text):
    """Extract a single diesel price from text."""
    if not text:
        return None
    
    text = str(text).strip()
    
    # Look for common price first (usually labeled)
    common_match = re.search(r'common\s+price\s*:?\s*(\d{2,3}\.\d{2})', text, re.IGNORECASE)
    if common_match:
        return Decimal(common_match.group(1))
    
    # Look for average price
    avg_match = re.search(r'avg\.?\s*:?\s*(\d{2,3}\.\d{2})', text, re.IGNORECASE)
    if avg_match:
        return Decimal(avg_match.group(1))
    
    # Look for range (low-high) - take the average of low and high
    range_match = re.search(r'(\d{2,3}\.\d{2})\s*[-–]\s*(\d{2,3}\.\d{2})', text)
    if range_match:
        low = Decimal(range_match.group(1))
        high = Decimal(range_match.group(2))
        return round((low + high) / 2, 2)
    
    # Look for single price value
    price_match = re.search(r'\b(\d{2,3}\.\d{2})\b', text)
    if price_match:
        return Decimal(price_match.group(1))
    
    return None


def parse_doe_pdf(pdf_bytes: bytes, period_label: str = "") -> list:
    """
    Extract diesel price for each municipality in Surigao del Sur.
    Returns: [{ municipality, price, average, period_label }, ...]
    """
    # Try table extraction first
    try:
        results = _parse_via_tables(pdf_bytes, period_label)
        if results:
            print(f"[DOE] Found {len(results)} municipalities")
            return results
    except Exception as e:
        print(f"[DOE] Table extraction failed: {e}")
    
    # Try text extraction
    text = _extract_text(pdf_bytes)
    if text.strip():
        results = _parse_via_text(text, period_label)
        if results:
            print(f"[DOE] Found {len(results)} municipalities via text")
            return results
    
    # Try OCR as last resort
    print("[DOE] Trying OCR...")
    text = _extract_text_ocr(pdf_bytes)
    if text.strip():
        results = _parse_via_text(text, period_label)
        print(f"[DOE] Found {len(results)} municipalities via OCR")
        return results
    
    return []


def parse_provincial_average(pdf_bytes: bytes, period_label: str = "") -> dict | None:
    """
    Calculate provincial average from all municipalities.
    Returns: { province, average, period_label, municipality_count, details }
    """
    municipalities = parse_doe_pdf(pdf_bytes, period_label)
    
    if not municipalities:
        return None
    
    prices = [m["average"] for m in municipalities if m["average"] is not None]
    if not prices:
        return None
    
    provincial_avg = round(sum(prices) / len(prices), 2)
    
    return {
        "province": "Surigao del Sur",
        "average": provincial_avg,
        "period_label": period_label,
        "municipality_count": len(prices),
        "details": municipalities,
    }


def _extract_text(pdf_bytes):
    full_text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                full_text += t + "\n"
    return full_text


def _extract_text_ocr(pdf_bytes):
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except ImportError:
        print("[DOE] OCR not available")
        return ""
    
    full_text = ""
    images = convert_from_bytes(pdf_bytes, dpi=200)
    for i, img in enumerate(images):
        print(f"[DOE] OCR page {i+1}/{len(images)}")
        full_text += pytesseract.image_to_string(img, config='--psm 6') + "\n"
    return full_text


def _parse_via_tables(pdf_bytes, period_label):
    """Extract from PDF tables."""
    results = {}
    in_surigao = False
    
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables:
                continue
            
            for table in tables:
                current_municipality = None
                
                for row in table:
                    if not row:
                        continue
                    
                    cells = [str(c).strip() if c else "" for c in row]
                    row_text = " ".join(cells).lower()
                    
                    # Check if we're in Surigao del Sur section
                    if any(k in row_text for k in PROVINCE_MARKERS):
                        in_surigao = True
                    elif any(k in row_text for k in NEXT_PROVINCE_MARKERS):
                        in_surigao = False
                    
                    if not in_surigao:
                        continue
                    
                    # Find municipality name
                    for mun in SURIGAO_MUNICIPALITIES:
                        if mun.lower() in row_text:
                            current_municipality = mun
                            break
                    
                    if not current_municipality:
                        continue
                    
                    # Look for diesel price in this row
                    if 'diesel' in row_text:
                        # Combine all cells to find price
                        full_row = " ".join(cells)
                        price = _extract_diesel_price(full_row)
                        
                        if price and current_municipality not in results:
                            results[current_municipality] = {
                                "municipality": current_municipality,
                                "price": price,      # For backward compatibility
                                "average": price,    # For backward compatibility
                                "period_label": period_label,
                            }
    
    return list(results.values())


def _parse_via_text(text, period_label):
    """Extract from plain text."""
    results = {}
    in_surigao = False
    current_municipality = None
    
    for line in text.split('\n'):
        ll = line.lower()
        
        # Track province boundaries
        if any(k in ll for k in PROVINCE_MARKERS):
            in_surigao = True
            continue
        if any(k in ll for k in NEXT_PROVINCE_MARKERS):
            in_surigao = False
            continue
        
        if not in_surigao:
            continue
        
        # Look for municipality
        for mun in SURIGAO_MUNICIPALITIES:
            if mun.lower() in ll:
                current_municipality = mun
                break
        
        if not current_municipality:
            continue
        
        # Look for diesel price
        if 'diesel' in ll:
            price = _extract_diesel_price(line)
            
            if price and current_municipality not in results:
                results[current_municipality] = {
                    "municipality": current_municipality,
                    "price": price,      # For backward compatibility
                    "average": price,    # For backward compatibility
                    "period_label": period_label,
                }
    
    return list(results.values())