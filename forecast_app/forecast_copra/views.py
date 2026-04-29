from django.core.cache import cache
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # Required for background rendering
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render
from django.core.paginator import Paginator
from .models import TrainedModel  
import pandas as pd
import numpy as np
from django.contrib import messages
from django.utils import timezone
import base64
import io
import os
from datetime import date, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from django.shortcuts import render
from django.contrib import messages
from .models import TrainingData, ForecastLog
import pdfplumber
import json
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from datetime import date, datetime
import re
import requests
from .models import DOEDieselUpload
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from .forms import ExcelUploadForm, LoginForm, TrainingDataForm, ForecastForm
from .models import TrainingData, TrainedModel, ForecastLog, ExcelUpload
from .utils.arimax_model import ARIMAXModel


def parse_period_to_date(period_label):
    """
    Parse period label like 'April 21-27, 2026' into a date (start of period).
    Returns date object or None if parsing fails.
    """
    if not period_label:
        return None
    try:
        # Split by '-' to separate start and end
        parts = period_label.split('-')
        if len(parts) >= 2:
            start_part = parts[0].strip()  # e.g., "April 21"
            # Get year from the end part
            end_part = parts[1].strip()
            if ',' in end_part:
                year = end_part.split(',')[1].strip()
            else:
                # Fallback, assume current year or something, but for now None
                return None
            start_date_str = f"{start_part}, {year}"
            return datetime.strptime(start_date_str, "%B %d, %Y").date()
    except (ValueError, IndexError):
        pass
    return None


# ── Shared headers ────────────────────────────────────────────────────────────
_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}
 
 
# ==============================================================================
# SCRAPER 1: ICC Coconut Oil Price (PDF — no Selenium, no browser)
# PDF URL pattern: coconutcommunity.org/files/document/wpu{YEAR}-{MONTH:02d}.pdf
# Contains: "Philippines (Domestic, Millgate Price)" — exact data you need
# Speed: ~1–2s (was ~15–20s with Selenium)
# ==============================================================================
def get_live_coconut_oil_price():
    """
    Downloads ICC weekly price PDF directly.
    Tries current month first, then falls back up to 4 months.
    Returns: { price (USD/MT), date (str), change (str) }
 
    FIX: Strips commas before regex so "2,433" is read as 2433 not 433.
    """
    data = {
        "price":  None,
        "date":   date.today().strftime('%b %d, %Y'),
        "change": "0.00"
    }
 
    today = date.today()
 
    # Build list of months to try: current → up to 4 months back
    months_to_try = []
    for i in range(5):
        year  = today.year
        month = today.month - i
        if month <= 0:
            month += 12
            year  -= 1
        months_to_try.append((year, month))
 
    for year, month in months_to_try:
        url = f"https://coconutcommunity.org/files/document/wpu{year}-{month:02d}.pdf"
        print(f"[ICC PDF] Trying: {url}")
 
        try:
            response = requests.get(url, headers=_HEADERS, timeout=12)
 
            if response.status_code != 200:
                print(f"[ICC PDF] HTTP {response.status_code} — trying previous month")
                continue
 
            # ── Extract full text from all PDF pages ──────────────────────
            full_text = ""
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
 
            if not full_text.strip():
                print("[ICC PDF] Empty PDF — skipping")
                continue
 
            print(f"[ICC PDF] Text snippet:\n{full_text[:600]}")
 
            # ── Find the Philippines (Domestic, Millgate Price) row ───────
            lines = full_text.split('\n')
 
            for i, line in enumerate(lines):
                if 'Philippines' in line and ('Domestic' in line or 'Millgate' in line):
                    print(f"[ICC PDF] Found target line [{i}]: {line}")

                    # ── Only use THIS line, not next lines ────────────────────────
                    # Joining next lines pulls in OTHER product prices (e.g. 682)

                    # Step 1: remove "2 ,433" style spaces before comma → "2433"
                    line_clean = re.sub(r'(\d)\s*,\s*(\d)', r'\1\2', line)
                    # Step 2: remove any remaining commas
                    line_clean = line_clean.replace(',', '')

                    print(f"[ICC PDF] Cleaned line: {line_clean}")

                    # Match 4+ digit numbers only (2433, 2796 etc — avoids small stray numbers)
                    raw_numbers = re.findall(r'\b(\d{4,5}(?:\.\d+)?)\b', line_clean)

                    prices = []
                    for raw in raw_numbers:
                        try:
                            val = float(raw)
                            if 500 < val < 8000:
                                prices.append(val)
                        except ValueError:
                            continue

                    print(f"[ICC PDF] Prices found: {prices}")
 
                    if not prices:
                        continue
 
                    # Latest price = last value in the row (most recent week)
                    latest_price = prices[-1]
 
                    # Compute week-over-week change
                    if len(prices) >= 2:
                        prev_price = prices[-2]
                        change_val = latest_price - prev_price
                        data['change'] = f"{'+' if change_val >= 0 else ''}{change_val:.2f}"
                    else:
                        data['change'] = "0.00"
 
                    data['price'] = latest_price
                    data['date']  = date(year, month, 1).strftime('%b %Y')
 
                    print(f"[ICC PDF] Success: price={latest_price}, change={data['change']}, date={data['date']}")
                    return data  # Found — exit immediately
 
            print(f"[ICC PDF] Philippines row not found in {url}")
 
        except requests.exceptions.Timeout:
            print(f"[ICC PDF] Timeout on {url}")
        except Exception as e:
            print(f"[ICC PDF] Error on {url}: {e}")
            continue
 
    print("[ICC PDF] All months failed — price unavailable")
    return data
 
 
# ==============================================================================
# SCRAPER 2: BSP Peso-Dollar Rate
# Source: bsp.gov.ph/statistics/external/day99_data.aspx
# Reads exact column for current month (e.g. "Mar-26"), most recent day
# Speed: ~0.5–1s
# ==============================================================================
def get_live_peso_rate():
    """
    Fetches PHP/USD rate from BSP daily rate table.
    Finds the exact column for the current month and reads the latest available day.
    Returns: { rate (float), date (str) }
    """
    data = {
        "rate": None,
        "date": date.today().strftime('%b %d, %Y')
    }
 
    try:
        response = requests.get(
            "https://www.bsp.gov.ph/statistics/external/day99_data.aspx",
            headers=_HEADERS,
            timeout=10
        )
 
        if response.status_code != 200:
            print(f"[BSP] Failed: HTTP {response.status_code}")
            return data
 
        soup = BeautifulSoup(response.content, 'lxml')
 
        today     = date.today()
        today_day = today.day
        today_mth = today.strftime('%b-%y')  # e.g. "Mar-26"
 
        all_rows = soup.find_all('tr')
 
        # ── Step 1: Find header row and locate current month column index ──
        target_col   = None
        header_texts = []
 
        for row in all_rows:
            cells = row.find_all('td')
            texts = [c.get_text(strip=True) for c in cells]
 
            # Header row has "Date" and month labels like "Mar-26"
            if 'Date' in texts and any(re.match(r'[A-Za-z]{3}-\d{2}', t) for t in texts):
                header_texts = texts
                print(f"[BSP] Header: {header_texts}")
 
                if today_mth in header_texts:
                    target_col = header_texts.index(today_mth)
                    print(f"[BSP] Target column: {target_col} ({today_mth})")
                else:
                    # Fall back to the most recent month column available
                    month_indices = [
                        i for i, t in enumerate(header_texts)
                        if re.match(r'[A-Za-z]{3}-\d{2}', t)
                    ]
                    if month_indices:
                        target_col = month_indices[-1]
                        fallback_mth = header_texts[target_col]
                        print(f"[BSP] Fallback column: {target_col} ({fallback_mth})")
                break
 
        if target_col is None:
            print("[BSP] Could not find target month column")
            return data
 
        # ── Step 2: Scan rows — get most recent day <= today ──────────────
        best_rate = None
        best_day  = None
 
        for row in all_rows:
            cells = row.find_all('td')
 
            if len(cells) <= target_col:
                continue
 
            # First non-empty cell = day number
            non_empty = [c.get_text(strip=True) for c in cells if c.get_text(strip=True)]
            if not non_empty:
                continue
 
            try:
                day_num = int(non_empty[0])
            except ValueError:
                continue  # skip AVERAGE, header rows etc.
 
            if day_num > today_day:
                continue  # future day — skip
 
            # Read value at exact target column
            rate_text = cells[target_col].get_text(strip=True).replace(',', '').strip()
 
            if not rate_text:
                continue  # holiday / no trading that day
 
            try:
                val = float(rate_text)
                if 40 < val < 100:  # sanity check: PHP/USD always in this range
                    best_rate = val
                    best_day  = day_num
                    print(f"[BSP] Day {day_num}: {val}")
            except ValueError:
                continue
 
        # ── Step 3: Return result ─────────────────────────────────────────
        if best_rate:
            data['rate'] = best_rate
            try:
                data['date'] = date(today.year, today.month, best_day).strftime('%b %d, %Y')
            except Exception:
                data['date'] = today.strftime('%b %d, %Y')
            print(f"[BSP] Success: {data}")
        else:
            print("[BSP] No rate found")
 
    except requests.exceptions.Timeout:
        print("[BSP] Timeout")
    except Exception as e:
        print(f"[BSP] Error: {e}")
 
    return data
 
def get_latest_doe_diesel_average():
    """
    Get the latest provincial average diesel price from the most recent DOE upload.
    Returns (average_price, period_label) or (None, None) if not found.
    """
    from .models import DOEDieselUpload
    
    # Get the most recent processed upload
    latest_upload = DOEDieselUpload.objects.filter(processed=True).first()
    
    if not latest_upload:
        return None, None
    
    # Calculate provincial average from all municipality prices
    prices = latest_upload.prices.all()
    if not prices:
        return None, None
    
    averages = [p.average for p in prices if p.average is not None]
    if not averages:
        return None, None
    
    provincial_average = sum(averages) / len(averages)
    
    return provincial_average, (latest_upload.start_date, latest_upload.end_date) 
# ==============================================================================
# PARALLEL FETCHER — both scrapers run at the SAME time
# Total time = slowest single scraper (~1–2s), NOT the sum of both
# ==============================================================================
def get_all_live_data():
    """
    Runs both scrapers in parallel using threads.
    Usage in views.py:
        live_market, live_peso = get_all_live_data()
    """
    results = {"oil": None, "peso": None, "wage": None}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(get_live_coconut_oil_price): "oil",
            executor.submit(get_live_peso_rate):         "peso",
            executor.submit(scrape_caraga_min_wage):     "wage",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                print(f"[PARALLEL] {key} scraper failed: {e}")
                if key == "oil":
                    results[key] = {"price": None, "date": date.today().strftime('%b %d, %Y'), "change": "0.00"}
                else:
                    results[key] = {"rate": None, "date": date.today().strftime('%b %d, %Y')}
 
    print(f"[PARALLEL] Oil:  {results['oil']}")
    print(f"[PARALLEL] Peso: {results['peso']}")
    return results["oil"], results["peso"]
# ====================
# PUBLIC VIEWS
# ====================

def scrape_caraga_min_wage():
    url = "https://nwpc.dole.gov.ph/summary-of-daily-minimum-wage-rates-per-wage-order-by-region-non-agriculture-1989-present/"
    
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Find all table rows
    rows = soup.find_all("tr")
    
    caraga_rows = []
    in_caraga = False
    
    for row in rows:
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        
        if not cells:
            continue
        
        # Detect CARAGA section
        if any("CARAGA" in c or "XIII" in c for c in cells):
            in_caraga = True
        
        # Stop at next region
        if in_caraga and any("ARMM" in c or "BARMM" in c for c in cells):
            break
        
        if in_caraga and cells:
            caraga_rows.append(cells)
    
    # Get the last row with a wage value
    latest_wage = None
    for row in reversed(caraga_rows):
        for cell in reversed(row):
            try:
                val = float(cell.replace(",", ""))
                if 300 < val < 1000:  # reasonable wage range
                    latest_wage = val
                    break
            except ValueError:
                continue
        if latest_wage:
            break
    
    return latest_wage  # returns 475.00

# Usage
# wage = scrape_caraga_min_wage()
# print(f"Current Caraga Min Wage: ₱{wage}")


def home(request):
    """Home page with forecast form"""

    # ── Fetch Latest Farmgate Price ──────────────────────────────────────────
    latest_data = TrainingData.objects.order_by('-date').first()
    latest_farmgate_price = float(latest_data.farmgate_price) if latest_data else None
    latest_farmgate_date  = latest_data.date if latest_data else None
    
    live_market = cache.get('live_market')
    if not live_market:
        live_market = get_live_coconut_oil_price()
        cache.set('live_market', live_market, timeout=3600)

    live_peso = cache.get('live_peso')
    if not live_peso:
        live_peso = get_live_peso_rate()
        cache.set('live_peso', live_peso, timeout=3600)

    doe_avg_price, doe_period = get_latest_doe_diesel_average()
    # Diesel price: use DOE average as default, farmer can override
    live_diesel_price = doe_avg_price    # e.g. 98.38
    if isinstance(doe_period, tuple) and len(doe_period) == 2:
        start_date, end_date = doe_period
        if hasattr(start_date, 'strftime') and hasattr(end_date, 'strftime'):
            live_diesel_date = f"{start_date.strftime('%B %d')} to {end_date.strftime('%B %d, %Y')}" if start_date and end_date else "Unavailable"
        else:
            live_diesel_date = f"{start_date} to {end_date}" if start_date and end_date else "Unavailable"
    else:
        if hasattr(doe_period, 'strftime'):
            live_diesel_date = doe_period.strftime('%B %d, %Y') if doe_period else "Unavailable"
        else:
            live_diesel_date = str(doe_period) if doe_period else "Unavailable"
    context = {
        'live_diesel_price': live_diesel_price,
        'live_diesel_date':  live_diesel_date,
    }

    # Labor min wage — cached 24 hours (changes rarely)
    live_labor_wage = cache.get('live_labor_wage')
    if not live_labor_wage:
        live_labor_wage = scrape_caraga_min_wage()
        cache.set('live_labor_wage', live_labor_wage, timeout=86400)  # 24 hours
    # -------- Handle form submission --------
    if request.method == 'POST':
        form = ForecastForm(request.POST)

        if form.is_valid():
            active_model = TrainedModel.objects.filter(is_active=True).first()

            if not active_model:
                messages.error(request, 'No trained model available. Please check back later.')
                return redirect('home')

            try:
                # ── Load Model ───────────────────────────────────────────
                arimax = ARIMAXModel()
                arimax.load_model(active_model.model_file_path)

                # ── Get User Inputs ──────────────────────────────────────
                oil_price        = float(form.cleaned_data['oil_price_trend'])
                peso_dollar      = float(form.cleaned_data['peso_dollar_rate'])
                diesel_price     = form.cleaned_data.get('diesel_price')
                labor_min_wage   = form.cleaned_data.get('labor_min_wage')
                forecast_horizon = int(form.cleaned_data['forecast_horizon'])

                # ── Run Forecast ─────────────────────────────────────────
                forecast_result = arimax.forecast(
                    steps=forecast_horizon,
                    use_latest_values=True,
                    latest_oil=oil_price,
                    latest_peso=peso_dollar,
                    latest_diesel=float(diesel_price) if diesel_price else None,
                    latest_labor=float(labor_min_wage) if labor_min_wage else None,
                )

                # ── Extract Final Predicted Price ────────────────────────
                if hasattr(forecast_result, 'iloc'):
                    predicted_price = float(forecast_result.iloc[-1])
                else:
                    predicted_price = float(forecast_result[-1])

                # ── Log Forecast ─────────────────────────────────────────
                ForecastLog.objects.create(
                    model_used=active_model, 
                    forecast_horizon=forecast_horizon,
                    farmer_input_oil_price_trend=oil_price,
                    farmer_input_peso_dollar_rate=peso_dollar,
                    farmer_input_diesel_price=float(diesel_price) if diesel_price else None,
                    farmer_input_labor_min_wage=float(labor_min_wage) if labor_min_wage else None,
                    price_predicted=predicted_price,
                )

            
                forecast_start = (
                    datetime.now().date() + timedelta(days=1)
                    if latest_farmgate_date
                    else datetime.now().date()
                )

                forecast_dates = pd.date_range(
                    start=forecast_start,
                    periods=forecast_horizon,
                    freq='D',
                ).strftime('%Y-%m-%d').tolist()

                # ── Forecast Values ──────────────────────────────────────
                if hasattr(forecast_result, 'tolist'):
                    forecast_values = forecast_result.tolist()
                elif hasattr(forecast_result, 'values'):
                    forecast_values = forecast_result.values.tolist()
                else:
                    forecast_values = list(forecast_result)

                forecast_data = list(zip(forecast_dates, forecast_values))

                # ── Initialize Output Variables ──────────────────────────
                trend                  = None
                volatility             = None
                price_range            = None
                summary_recommendation = None
                recommendations        = []

                # ── Compute Statistics ───────────────────────────────────
                if len(forecast_values) >= 2:
                    start_price  = float(forecast_values[0])
                    end_price    = float(forecast_values[-1])
                    prices_arr   = np.array(forecast_values, dtype=float)
                    mean_price   = float(prices_arr.mean())
                    std_price    = float(prices_arr.std())
                    volatility   = (std_price / mean_price * 100.0) if mean_price > 0 else 0.0

                    total_change_pct = (
                        ((end_price - start_price) / start_price) * 100.0
                        if start_price > 0 else 0.0
                    )

                    # Trend classification
                    if total_change_pct > 3:
                        trend = 'increasing'
                    elif total_change_pct < -3:
                        trend = 'decreasing'
                    else:
                        trend = 'stable'

                    price_range = {
                        'min': float(prices_arr.min()),
                        'max': float(prices_arr.max()),
                        'avg': mean_price,
                    }

                    # Optimal selling day
                    best_day_index = int(np.argmax(prices_arr))
                    best_day_date  = forecast_dates[best_day_index]
                    best_day_price = float(prices_arr[best_day_index])

                    # ── SUMMARY RECOMMENDATION ───────────────────────────
                    if trend == 'increasing':
                        summary_recommendation = (
                            f"Prices are projected to RISE by {abs(total_change_pct):.1f}% "
                            f"over {forecast_horizon} days. "
                            f"It is recommended to WAIT and sell closer to {best_day_date} "
                            f"for better returns."
                        )
                    elif trend == 'decreasing':
                        summary_recommendation = (
                            f"Prices are projected to DROP by {abs(total_change_pct):.1f}% "
                            f"over {forecast_horizon} days. "
                            f"It is recommended to SELL SOON to avoid further price decline."
                        )
                    else:
                        summary_recommendation = (
                            f"Prices are STABLE over the next {forecast_horizon} days. "
                            f"You have flexibility to sell based on your logistics and cash-flow needs."
                        )

                    # ── RECOMMENDATIONS ──────────────────────────────────

                    # 1. OPTIMAL SELLING TIME
                    recommendations.append(
                        f"📅OPTIMAL SELLING TIME: Based on the forecast, the best time to sell "
                        f"your copra is on <strong>{best_day_date}</strong> with an estimated price of "
                        f"<strong>₱{best_day_price:.2f}/kg</strong>. "
                        f"This is the highest projected price within your {forecast_horizon}-day forecast window."
                    )

                    # 2. SELL NOW OR WAIT?
                    if trend == 'increasing':
                        recommendations.append(
                            f"SELL OR WAIT: Prices are trending <strong>upward</strong>. "
                            f"Waiting until <strong>{best_day_date}</strong> could give you "
                            f"₱{best_day_price - start_price:.2f}/kg more than selling today. "
                            f"Only wait if your copra is properly dried and stored."
                        )
                    elif trend == 'decreasing':
                        recommendations.append(
                            f"SELL OR WAIT: Prices are trending <strong>downward</strong>. "
                            f"It is advised to <strong>sell as soon as possible</strong> to protect your income. "
                            f"Delaying may result in ₱{start_price - end_price:.2f}/kg loss."
                        )
                    else:
                        recommendations.append(
                            f"SELL OR WAIT: Prices are <strong>stable</strong> with minimal change expected. "
                            f"You can sell at your convenience. "
                        )

                    # 3. RISK ADVISORY
                    if volatility > 15:
                        recommendations.append(
                            f"⚠️ RISK ADVISORY: Price volatility is <strong>HIGH ({volatility:.1f}%)</strong>. "
                            f"Avoid selling all your copra on a single day. "
                        )
                    elif volatility > 7:
                        recommendations.append(
                            f"⚠️ RISK ADVISORY: Moderate price fluctuations detected ({volatility:.1f}% volatility). "
                            f"Monitor weekly coconut oil price and daily peso-dollar rate changes before finalizing "
                            f"your selling schedule."
                        )
                    else:
                        recommendations.append(
                            f"✅ RISK ADVISORY: Price forecast is <strong>stable "
                            f"(low volatility: {volatility:.1f}%)</strong>. "
                        )

                    # 4. PRICE RANGE AWARENESS
                    recommendations.append(
                        f"💰 PRICE RANGE: Over the next {forecast_horizon} days, copra prices are expected "
                        f"to range between <strong>₱{price_range['min']:.2f}</strong> and "
                        f"<strong>₱{price_range['max']:.2f}</strong>, with an average of "
                        f"<strong>₱{price_range['avg']:.2f}/kg</strong>. "
                        f"Use this range to negotiate better deals with traders."
                    )

                    # 5. MARKET FACTORS REMINDER
                    recommendations.append(
                        f"🌍 MARKET FACTORS: This forecast is based on your current oil price trend "
                        f"(₱{oil_price:.2f}) and peso-dollar rate (₱{peso_dollar:.2f}). "
                        f"Sudden changes in global oil prices or exchange rates may shift actual "
                        f"farmgate prices. Re-check the forecast if major market events occur."
                    )

                # ── Render Result Page ───────────────────────────────────
                return render(request, 'forecast_copra/forecast_result.html', {
                    'predicted_price':        predicted_price,
                    'oil_price':              oil_price,
                    'peso_dollar_rate':       peso_dollar,
                    'forecast_horizon':       forecast_horizon,
                    'model_name':             active_model.name,
                    'forecast_data':          forecast_data,
                    'trend':                  trend,
                    'volatility':             volatility,
                    'price_range':            price_range,
                    'summary_recommendation': summary_recommendation,
                    'recommendations':        recommendations,
                    'forecast_start': forecast_start,
                    'latest_farmgate_price':  latest_farmgate_price,
                    'latest_farmgate_date':   latest_farmgate_date,
                })
            
            except Exception as e:
                messages.error(request, f'Forecast error: {str(e)}')
                return redirect('home')
    else:
        form = ForecastForm(initial={
            'diesel_price': live_diesel_price if live_diesel_price else None
        })

    # -------- Page display section --------
    active_model = TrainedModel.objects.filter(is_active=True).first()

    if active_model:
        model_available = True
        model_info      = f"Active Model: {active_model.name}"
    else:
        model_available = False
        model_info      = "No trained model available. Forecasts cannot be made."

    recent_forecasts = ForecastLog.objects.all().order_by('-created_at')[:5]

    return render(request, 'forecast_copra/home.html', {
        'form':            form,
        'recent_forecasts': recent_forecasts,
        'model_available': model_available if model_available else None,
        'active_model':    active_model if active_model else None,
        'model_info':      model_info if model_info else None,
        'suggested_oil':  latest_data.oil_price_trend if latest_data else None,
        'suggested_peso': latest_data.peso_dollar_rate if latest_data else None,
        'is_negative':   "-" in str(live_market['change']),
        'live_oil_price': f"{live_market['price']:.2f}" if live_market['price'] else "Unavailable",
        'live_oil_date':  live_market['date'] if live_market['date'] else "Unavailable",
        'live_oil_change': live_market['change'] if live_market['change'] else "Unavailable",
        'latest_date':    latest_data.date if latest_data else None,
        'live_peso_rate':  f"{live_peso['rate']:.2f}" if live_peso['rate'] else None,
        'live_peso_date':  live_peso['date'] if live_peso['date'] else "Unavailable",
        'latest_farmgate_price':  latest_farmgate_price if latest_farmgate_price else None,
        'latest_farmgate_date':   latest_farmgate_date if latest_farmgate_date else None,
        'live_diesel_price':      f"{live_diesel_price:.2f}" if live_diesel_price else None,
        'live_diesel_date':       live_diesel_date,
        'live_labor_wage':        live_labor_wage['wage'] if live_labor_wage else None,
        'live_labor_wage_date':   live_labor_wage['period'] if live_labor_wage else None,
    })

def recent_forecasts(request):
    """View all recent forecasts with calculated target dates"""
    forecasts = ForecastLog.objects.all().order_by('-created_at')[:100]
    
    # We calculate the 'target_date' for each forecast dynamically
    for f in forecasts:
        # created_at + horizon = the day the prediction is actually for
        f.target_date = f.created_at + timedelta(days=f.forecast_horizon)
    
    return render(request, 'forecast_copra/recent_forecasts.html', {
        'forecasts': forecasts
    })

# ====================
# ADMIN VIEWS
# ====================

def admin_login(request):
    """Admin login page"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        if not username or not password:
            messages.error(request, 'Please enter both username and password.')
            return render(request, 'forecast_copra/admin_login.html')
        
        # Authenticate user
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            if user.is_staff:  # Check if user is admin/staff
                login(request, user)
                messages.success(request, f'Welcome, {username}!')
                return redirect('admin_dashboard')
            else:
                messages.error(request, 'This user is not an admin.')
        else:
            messages.error(request, 'Invalid username or password.')
    
    return render(request, 'forecast_copra/admin_login.html')

@login_required
def admin_logout(request):
    """Admin logout"""
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('home')

@login_required
def admin_dashboard(request):
    """Admin dashboard - only accessible by staff users"""
    if not request.user.is_staff:
        messages.error(request, 'Only admin users can access this page.')
        return redirect('home')
    
    # Get statistics
    total_data = TrainingData.objects.count()
    total_models = TrainedModel.objects.count()
    active_model = TrainedModel.objects.filter(is_active=True).first()
    total_forecasts = ForecastLog.objects.count()
    
    return render(request, 'forecast_copra/admin_dashboard.html', {
        'total_data': total_data,
        'total_models': total_models,
        'active_model': active_model,
        'total_forecasts': total_forecasts
    })

@login_required
def manage_data(request):
    """Manage training data"""
    if not request.user.is_staff:
        messages.error(request, 'Only admin users can access this page.')
        return redirect('home')
    
    excel_form = ExcelUploadForm()
    manual_form = TrainingDataForm()
    data = TrainingData.objects.all().order_by('-date')
    excel_uploads = ExcelUpload.objects.all().order_by('-uploaded_at')
    
    if request.method == 'POST':
        if 'excel_submit' in request.POST:
            excel_form = ExcelUploadForm(request.POST, request.FILES)
            if excel_form.is_valid():
                excel_file = excel_form.cleaned_data['excel_file']
                
                # Save uploaded file
                fs = FileSystemStorage()
                filename = fs.save(f'excel_uploads/{excel_file.name}', excel_file)
                file_path = fs.path(filename)
                
                # Create upload record
                upload = ExcelUpload.objects.create(file=filename)
                
                try:
                    # Process Excel file
                    processed_data, message = process_excel_file(file_path)
                    
                    if processed_data:
                        # Save data to database
                        saved_count = 0
                        for item in processed_data:
                            if not TrainingData.objects.filter(date=item['date']).exists():
                                TrainingData.objects.create(
                                    date=item['date'],
                                    farmgate_price=item['farmgate_price'],
                                    oil_price_trend=item['oil_price_trend'],
                                    peso_dollar_rate=item['peso_dollar_rate'],
                                    diesel_price=item.get('diesel_price'),
                                    labor_min_wage=item.get('labor_min_wage')
                                )
                                saved_count += 1
                        
                        # Update upload record
                        upload.processed = True
                        upload.rows_imported = saved_count
                        upload.save()
                        
                        messages.success(request, f' Successfully imported {saved_count} rows from Excel file.')
                    else:
                        messages.error(request, f' No data was imported. {message}')
                        
                except Exception as e:
                    messages.error(request, f' Error processing Excel file: {str(e)}')
                
                return redirect('manage_data')
        
        elif 'manual_submit' in request.POST:
            manual_form = TrainingDataForm(request.POST)
            if manual_form.is_valid():
                date = manual_form.cleaned_data['date']
                if TrainingData.objects.filter(date=date).exists():
                    messages.warning(request, f'⚠️ Data for date {date} already exists.')
                else:
                    manual_form.save()
                    messages.success(request, ' Data added successfully')
            else:
                messages.error(request, ' Please correct the errors in the form.')
            return redirect('manage_data')
    
    return render(request, 'forecast_copra/manage_data.html', {
        'excel_form': excel_form,
        'manual_form': manual_form,
        'data': data,
        'excel_uploads': excel_uploads
    })

@login_required
def train_model(request):
    """Train ARIMAX model with ACF/PACF Diagnostics & Model Saving"""
    graph_base64 = None 
    diagnostic_graph = None
    metrics = {}
    raw_series_graph = None
    comparison_rows = []
    
    p, d, q = None, None, None

    if request.method == 'POST':
        # 1. Capture Parameters (p, d, q) - REQUIRED from user input
        try:
            p = int(request.POST.get('p', 1))
            d = int(request.POST.get('d', 1))
            q = int(request.POST.get('q', 1))
        except (ValueError, TypeError):
            p, d, q = 1, 1, 1

        # NEW: Capture train/val/test ratios (with defaults)
        try:
            train_ratio = float(request.POST.get('train_ratio', 0.7))
            val_ratio = float(request.POST.get('val_ratio', 0.15))
            test_ratio = float(request.POST.get('test_ratio', 0.15))
        except (ValueError, TypeError):
            train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

        # 2. Identify and Load Data Source
        processed_data = None
        use_full_data = False
        
        if 'excel_file' in request.FILES:
            excel_file = request.FILES.get('excel_file')
            fs = FileSystemStorage()
            filename = fs.save(f'temp_training/{excel_file.name}', excel_file)
            file_path = fs.path(filename)
            processed_data, _ = process_excel_file(file_path)
            use_full_data = False 
            if os.path.exists(file_path): os.remove(file_path)
        else:
            processed_data = list(TrainingData.objects.all().values())
            use_full_data = True
            
        if processed_data and len(processed_data) > 0 and 'diagnose' in request.POST:
            try:
                df_raw = pd.DataFrame(processed_data)
                df_raw['date'] = pd.to_datetime(df_raw['date'])
                df_raw = df_raw.sort_values('date')

                fig, axes = plt.subplots(2, 1, figsize=(12, 6))

                # Plot 1: Raw Series
                axes[0].plot(df_raw['date'], df_raw['farmgate_price'],
                    color='#2980b9', linewidth=1.5)
                axes[0].set_title('Raw Time Series: Farmgate Price (Before Differencing)')
                axes[0].set_ylabel('Price (₱)')
                axes[0].grid(True, alpha=0.3)

                # Plot 2: Differenced Series
                differenced = df_raw['farmgate_price'].diff(d).dropna()
                axes[1].plot(differenced.values, color='#e67e22', linewidth=1.5)
                axes[1].set_title(f'Differenced Series (d={d}): After Differencing')
                axes[1].set_ylabel('Differenced Price')
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                plt.close()
                raw_series_graph = base64.b64encode(buf.getvalue()).decode('utf-8')
            except Exception as e:
                print(f"Raw series plot error: {e}")

        # 3. Generate Diagnostic Graph (ACF/PACF) ONLY for Excel evaluation training
        if processed_data and len(processed_data) > 0 and 'diagnose' in request.POST:
            try:
                df = pd.DataFrame(processed_data)
                df['date'] = pd.to_datetime(df['date'])       
                df = df.sort_values('date').reset_index(drop=True) 
                series = df['farmgate_price'].diff(d).dropna()

                if not series.empty:
                    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    lags = min(20, len(series)//2 - 1)
                    if lags > 0:
                        plot_acf(series, ax=ax1, lags=lags)
                        ax1.set_title(f"ACF: Autocorrelation (MA Identification) or q | d={d}")
                        plot_pacf(series, ax=ax2, lags=lags)
                        ax2.set_title(f"PACF: Partial Autocorrelation (AR Identification) or p | d={d}")

                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight')
                        plt.close()
                        diagnostic_graph = base64.b64encode(buf.getvalue()).decode('utf-8')
            except Exception as e:
                print(f"Diagnostic error: {e}")

        # 4. ACTION: TRAIN (Only if train button clicked)
        if 'excel_train' in request.POST or 'db_train' in request.POST:
            if processed_data and len(processed_data) >= 10:
                try:
                    print(f"[TRAINING] Using ARIMA order: ({p}, {d}, {q})")
                    print(f"[TRAINING] Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
                    
                    arimax = ARIMAXModel(order=(p, d, q))
                    # Pass the ratios to train method
                    metrics = arimax.train(
                        processed_data, 
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio,
                        is_deployment=use_full_data
                    )

                    if 'error' in metrics:
                        messages.error(request, f"Training failed: {metrics['error']}")
                    else:
                        # --- GENERATE PERFORMANCE GRAPH (ONLY FOR EVALUATION / EXCEL TRAINING) ---
                        actual = metrics.get('plot_actual', [])
                        preds = metrics.get('plot_preds', [])
                        is_deployment = metrics.get('is_deployment', False)
                        
                        if actual and preds:
                            plt.figure(figsize=(10, 4))
                            sns.set_style("whitegrid")
                            plt.plot(actual, label='Actual Price', color='#2ecc71', linewidth=2, marker='o')
                            plt.plot(preds, label='Predicted Price', color='#e74c3c', linestyle='--', linewidth=2, marker='x')
                            
                            # NEW: Show both validation and test metrics in title
                            val_mape = metrics.get('val_mape', 0)
                            test_mape = metrics.get('mape', 0)
                            plt.title(f"Model Performance (p={p}, d={d}, q={q}) | Val MAPE: {val_mape:.2f}% | Test MAPE: {test_mape:.2f}%")
                            plt.xlabel('Test Sample Index')
                            plt.ylabel('Farmgate Price')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                            plt.close()
                            graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                            comparison_rows = [
                                {
                                    "index": i + 1,
                                    "actual": float(a),
                                    "predicted": float(pv),
                                    "error": float(abs(a - pv)),
                                    "error_pct": float(abs(a - pv) / (a + 1e-10) * 100)
                                }
                                for i, (a, pv) in enumerate(zip(actual, preds))
                            ]
                        
                        # --- SAVE MODEL RECORD ---
                        model_prefix = "model" if 'excel_train' in request.POST else "db_model"
                        model_name = f"{model_prefix}_{p}_{d}_{q}_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
                        model_path = arimax.save_model(model_name)

                        # Store metrics appropriately
                        if is_deployment:
                            mae_val = None
                            rmse_val = None
                            mape_store = None
                            aic_val = None
                            plot_actual=metrics.get('plot_actual') if not is_deployment else None,
                            plot_preds=metrics.get('plot_preds')  if not is_deployment else None, 
                            success_msg = f"✅ Model '{model_name}' trained (deployment mode) with order ({p},{d},{q})"
                        else:
                            # Store TEST metrics (not validation)
                            mae_val = metrics.get('mae', 0)
                            rmse_val = metrics.get('rmse', 0)
                            mape_store = metrics.get('mape', 0)
                            aic_val = metrics.get('aic', 0)
                            
                            # Also get validation metrics for display
                            val_mae = metrics.get('val_mae', 0)
                            val_rmse = metrics.get('val_rmse', 0)
                            val_mape = metrics.get('val_mape', 0)
                            test_accuracy = 100 - mape_store
                            success_msg = f"✅ Model '{model_name}' trained with order ({p},{d},{q})! Val MAPE: {val_mape:.2f}% | Test MAPE: {mape_store:.2f}%"

                        TrainedModel.objects.create(
                            name=model_name,
                            model_file_path=model_path,
                            is_active=True,
                            p=p, d=d, q=q,
                            mae=mae_val,
                            rmse=rmse_val,
                            mape=mape_store,
                            aic=aic_val,
                            plot_actual=metrics.get('plot_actual'),
                            plot_preds=metrics.get('plot_preds'), 
                        )
                        messages.success(request, success_msg)
                        
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    messages.error(request, f"❌ Error: {str(e)}")
            else:
                messages.error(request, " Insufficient data. Need at least 10 records.")

    # Prepare for rendering
    if p is None: p = 1
    if d is None: d = 1
    if q is None: q = 1

    models = TrainedModel.objects.all().order_by('-training_date')
    return render(request, 'forecast_copra/train_model.html', {
        'models': models,
        'data_count': TrainingData.objects.count(),
        'graph': graph_base64,
        'diagnostic_graph': diagnostic_graph,
        'raw_series_graph': raw_series_graph, 
        'metrics': metrics,
        'comparison_rows': comparison_rows,
        'p': p, 'd': d, 'q': q,
    })
@login_required
def trained_models_view(request):
    # Get all models (latest first) for table
    model_list = TrainedModel.objects.all().order_by('-training_date')

    # Get models with AIC for comparison chart (best to worst), exclude full‑data deployment models
    # Deployment models are named with "db_model" prefix in train_model()
    models_for_chart = TrainedModel.objects.filter(
        aic__isnull=False
    ).exclude(
        name__startswith="db_model"
    ).order_by('aic')

    # Pagination
    paginator = Paginator(model_list, 10)  # 10 per page
    page_number = request.GET.get('page')
    models = paginator.get_page(page_number)

    return render(request, "forecast_copra/trained_models.html", {
        "models": models,
        "models_for_chart": models_for_chart,
    })
    
@login_required
def activate_model(request, model_id):
    """Activate a trained model"""
    if not request.user.is_staff:
        messages.error(request, 'Only admin users can perform this action.')
        return redirect('train_model')
    
    try:
        model = TrainedModel.objects.get(id=model_id)
        # Deactivate all other models first
        TrainedModel.objects.filter(is_active=True).update(is_active=False)
        # Activate this model
        model.is_active = True
        model.save()
        messages.success(request, f'✅ Model "{model.name}" has been activated.')
    except TrainedModel.DoesNotExist:
        messages.error(request, '❌ Model not found.')
    except Exception as e:
        messages.error(request, f'❌ Error activating model: {str(e)}')
    
    return redirect('trained_models_view')

@login_required
def deactivate_model(request, model_id):
    """Deactivate a trained model"""
    if not request.user.is_staff:
        messages.error(request, 'Only admin users can perform this action.')
        return redirect('trained_models_view')
    
    try:
        model = TrainedModel.objects.get(id=model_id)
        model.is_active = False
        model.save()
        messages.success(request, f'⏸️ Model "{model.name}" has been deactivated.')
    except TrainedModel.DoesNotExist:
        messages.error(request, '❌ Model not found.')
    except Exception as e:
        messages.error(request, f'❌ Error deactivating model: {str(e)}')
    
    return redirect('trained_models_view')

@login_required
def delete_model(request, model_id):
    """Delete a trained model"""
    if not request.user.is_staff:
        messages.error(request, 'Only admin users can perform this action.')
        return redirect('trained_models_view')
    
    try:
        model = TrainedModel.objects.get(id=model_id)
        model_name = model.name
        model_path = model.model_file_path
        
        # Delete the model file if it exists
        if model_path and os.path.exists(model_path):
            try:
                os.remove(model_path)
            except Exception as e:
                print(f"Warning: Could not delete model file {model_path}: {e}")
        
        # Delete the model record
        model.delete()
        messages.success(request, f'🗑️ Model "{model_name}" has been deleted.')
    except TrainedModel.DoesNotExist:
        messages.error(request, '❌ Model not found.')
    except Exception as e:
        messages.error(request, f'❌ Error deleting model: {str(e)}')
    
    return redirect('trained_models_view')

# ====================
# HELPER FUNCTIONS
# ====================

def process_excel_file(file_path):
    """Process Excel file and extract data"""
    try:
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Convert column names to lowercase and strip spaces
        df.columns = df.columns.str.strip().str.lower()
        
        # Map possible column names to standard names
        column_mapping = {
            'date': ['date', 'dates', 'day', 'days'],
            'farmgate_price': ['farmgate_price', 'price', 'farmgate', 'farmgate price', 'farmgate_price', 'farmgateprice'],
            'oil_price_trend': ['oil_price_trend', 'oil price', 'oil', 'oil trend', 'oil_price', 'oilprice'],
            'peso_dollar_rate': ['peso_dollar_rate', 'exchange rate', 'peso dollar', 'exchange', 'peso_dollar', 'pesodollar'],
            'diesel_price': ['diesel_price', 'diesel price', 'diesel'],
            'labor_min_wage': ['labor_min_wage', 'labor wage', 'labor_minimum_wage', 'minimum_wage', 'labor wage']
        }
        
        # Try to map columns
        actual_columns = {}
        for standard_col, possible_names in column_mapping.items():
            for possible in possible_names:
                if possible in df.columns:
                    actual_columns[standard_col] = possible
                    break
        
        # If columns not found, use first few columns as default
        if not actual_columns:
            if len(df.columns) >= 4:
                actual_columns = {
                    'date': df.columns[0],
                    'farmgate_price': df.columns[1],
                    'oil_price_trend': df.columns[2],
                    'peso_dollar_rate': df.columns[3]
                }
            else:
                return [], "Excel file must have at least 4 columns"
        
        # Process each row
        processed_data = []
        error_rows = []
        
        for index, row in df.iterrows():
            try:
                # Extract raw date value
                raw_date = row[actual_columns['date']]
                date_obj = None

                # 1) If it's already a pandas/py datetime, just take the date
                if isinstance(raw_date, (datetime, pd.Timestamp)):
                    date_obj = raw_date.date()

                # 2) If it's a numeric Excel serial (e.g. 44293), try Excel origin
                if date_obj is None and isinstance(raw_date, (int, float)) and not pd.isna(raw_date):
                    try:
                        parsed = pd.to_datetime(raw_date, origin='1899-12-30', unit='D')
                        if not pd.isna(parsed):
                            date_obj = parsed.date()
                    except Exception:
                        pass

                # 3) Fallback: flexible string parsing (handles 1/4/2021 etc.)
                if date_obj is None:
                    date_str = str(raw_date).strip()
                    parsed = pd.to_datetime(date_str, errors='coerce', dayfirst=False)
                    if pd.isna(parsed):
                        # Try again assuming day-first format
                        parsed = pd.to_datetime(date_str, errors='coerce', dayfirst=True)
                    if not pd.isna(parsed):
                        date_obj = parsed.date()

                if not date_obj:
                    error_rows.append(f"Row {index+2}: Could not parse date '{raw_date}'")
                    continue
                
                # Extract numeric values - ensure they are proper floats
                try:
                    farmgate_price_val = row[actual_columns['farmgate_price']]
                    # Handle if it's already a number or string
                    if pd.isna(farmgate_price_val):
                        continue  # Skip rows with NaN
                    farmgate_price = float(farmgate_price_val)
                    if pd.isna(farmgate_price) or not np.isfinite(farmgate_price):
                        continue  # Skip invalid values
                except (ValueError, TypeError):
                    continue  # Skip rows that can't be converted
                
                try:
                    oil_price_trend_val = row[actual_columns['oil_price_trend']]
                    if pd.isna(oil_price_trend_val):
                        oil_price_trend = 0.0
                    else:
                        oil_price_trend = float(oil_price_trend_val)
                        if pd.isna(oil_price_trend) or not np.isfinite(oil_price_trend):
                            oil_price_trend = 0.0
                except (ValueError, TypeError):
                    oil_price_trend = 0.0
                
                try:
                    peso_dollar_rate_val = row[actual_columns['peso_dollar_rate']]
                    if pd.isna(peso_dollar_rate_val):
                        peso_dollar_rate = 0.0
                    else:
                        peso_dollar_rate = float(peso_dollar_rate_val)
                        if pd.isna(peso_dollar_rate) or not np.isfinite(peso_dollar_rate):
                            peso_dollar_rate = 0.0
                except (ValueError, TypeError):
                    peso_dollar_rate = 0.0
                
                diesel_price = None
                labor_min_wage = None
                if 'diesel_price' in actual_columns:
                    try:
                        val = row[actual_columns['diesel_price']]
                        diesel_price = float(val) if not pd.isna(val) else None
                    except (ValueError, TypeError):
                        diesel_price = None
                if 'labor_min_wage' in actual_columns:
                    try:
                        val = row[actual_columns['labor_min_wage']]
                        labor_min_wage = float(val) if not pd.isna(val) else None
                    except (ValueError, TypeError):
                        labor_min_wage = None

                processed_data.append({
                    'date': date_obj,
                    'farmgate_price': farmgate_price,
                    'oil_price_trend': oil_price_trend,
                    'peso_dollar_rate': peso_dollar_rate,
                    'diesel_price': diesel_price,
                    'labor_min_wage': labor_min_wage,
                })
                
            except Exception as e:
                error_rows.append(f"Row {index+2}: {str(e)}")
                continue
        
        if error_rows:
            print(f"Excel processing warnings: {error_rows}")
        
        return processed_data, "Success"
        
    except Exception as e:
        return [], f"Error processing Excel file: {str(e)}"

def historical_trend(request):

    # ── 1. All actual price data (for date alignment only) ────────────────────
    data_qs = TrainingData.objects.all().order_by('date')
    if not data_qs.exists():
        return render(request, 'forecast_copra/historical_trend.html', {'no_data': True})

    df = pd.DataFrame(list(data_qs.values('date', 'farmgate_price')))
    df['date'] = pd.to_datetime(df['date'])
    df['farmgate_price'] = pd.to_numeric(df['farmgate_price'], errors='coerce').astype(float)
    df = df.sort_values('date').reset_index(drop=True)

    # ── 2. Active model — test-set actual vs predicted ────────────────────────
    eval_rows           = []
    eval_dates          = pd.DatetimeIndex([])
    eval_actual          = []
    eval_preds           = []
    active_model         = None
    training_cutoff_date = None

    try:
        active_model = TrainedModel.objects.filter(
            is_active=True,
            plot_actual__isnull=False,
            plot_preds__isnull=False,
        ).order_by('-training_date').first()

        if active_model:
            # 1. Get plot lists, if available (evaluation mode path)
            if active_model.plot_actual is not None and active_model.plot_preds is not None:
                raw_actual = list(active_model.plot_actual)
                raw_preds  = list(active_model.plot_preds)

                # 2. Determine the correct count (n) for eval group
                n = min(len(raw_actual), len(df))

                # 3. Slice to match dataset end and preserve alignment
                tail_slice = df.iloc[-n:].reset_index(drop=True)
                eval_actual = raw_actual[:n]
                eval_preds  = raw_preds[:n]
                eval_dates  = pd.to_datetime(tail_slice['date'].values)

                if len(eval_dates) > 0:
                    eval_rows = [
                        {
                            'date':      pd.Timestamp(d).strftime('%b %d, %Y'),
                            'actual':    round(float(a), 2),
                            'predicted': round(float(p), 2),
                            'error':     round(abs(float(a) - float(p)), 2),
                            'error_pct': round(
                                abs(float(a) - float(p)) / (abs(float(a)) + 1e-10) * 100, 2
                            ),
                        }
                        for d, a, p in zip(eval_dates, eval_actual, eval_preds)
                    ]

                    # Use the last evaluated date as the admin data cutoff in evaluation mode
                    training_cutoff_date = eval_dates[-1]

            # Deployment or no plot data: use last available historical date as cutoff
            if training_cutoff_date is None:
                training_cutoff_date = df['date'].max()
    except Exception as e:
        print(f"[historical_trend] eval error: {e}")

    # If no cutoff from evaluation model, check for deployment model (or old/legacy backup)
    if training_cutoff_date is None:
        active_model = TrainedModel.objects.filter(is_active=True).order_by('-training_date').first()
        if active_model:
            # For deployment mode we consider all data used by model as up to the model training time.
            # So new admin records should be strictly after the last training timestamp.
            training_cutoff_date = active_model.training_date.date()
        else:
            # Fallback if no active model exists
            training_cutoff_date = df['date'].max()

    # ── 3. Admin new data — TrainingData rows in a recent window around training cutoff ──
    admin_rows = []
    admin_df   = pd.DataFrame()

    admin_window_days = 5
    admin_window_start = None
    if training_cutoff_date is not None:
        admin_window_start = training_cutoff_date - timedelta(days=admin_window_days)

    if admin_window_start is not None:
        after_qs = TrainingData.objects.filter(
            date__gte=admin_window_start
        ).order_by('date')
    else:
        after_qs = TrainingData.objects.all().order_by('date')

    if after_qs.exists():
        admin_df = pd.DataFrame(list(after_qs.values('date', 'farmgate_price')))
        admin_df['date'] = pd.to_datetime(admin_df['date'])
        admin_df['farmgate_price'] = pd.to_numeric(
            admin_df['farmgate_price'], errors='coerce'
        ).astype(float)
        admin_df = admin_df.sort_values('date').reset_index(drop=True)

        admin_rows = [
            {
                'date':  row['date'].strftime('%b %d, %Y'),
                'price': round(float(row['farmgate_price']), 2),
            }
            for _, row in admin_df.iterrows()
        ]

    # ── 4. User forecast log ──────────────────────────────────────────────────
    log_rows = []
    log_df   = pd.DataFrame()

    logs = ForecastLog.objects.all().order_by('created_at')
    if logs.exists():
        log_df = pd.DataFrame(
            list(logs.values('created_at', 'price_predicted', 'forecast_horizon'))
        )
        log_df['created_at'] = pd.to_datetime(log_df['created_at']).dt.tz_localize(None)
        log_df['target_date'] = log_df.apply(
            lambda x: x['created_at'] + timedelta(days=int(x['forecast_horizon'] or 0)),
            axis=1,
        )
        log_df = log_df.sort_values('target_date').reset_index(drop=True)

        log_rows = [
            {
                'date':      row['target_date'].strftime('%b %d, %Y'),
                'predicted': round(float(row['price_predicted']), 2),
            }
            for _, row in log_df.iterrows()
        ]

    # Compare admin actual to user forecast for exact dates
    admin_forecast_rows = []
    if not admin_df.empty and not log_df.empty:
        merged = pd.merge(
            admin_df,
            log_df[['target_date', 'price_predicted']],
            left_on='date',
            right_on='target_date',
            how='inner'
        )

        if not merged.empty:
            admin_forecast_rows = [
                {
                    'date':      row['date'].strftime('%b %d, %Y'),
                    'actual':    round(float(row['farmgate_price']), 2),
                    'forecast':  round(float(row['price_predicted']), 2),
                    'diff':      round(float(row['farmgate_price']) - float(row['price_predicted']), 2),
                    'diff_pct':  round(abs(float(row['farmgate_price']) - float(row['price_predicted'])) / (abs(float(row['farmgate_price'])) + 1e-10) * 100, 2),
                }
                for _, row in merged.iterrows()
            ]

    # ── 5. Build chart (TEST-SET WINDOW ONLY — no 2021-2024 history) ──────────
    plt.close('all')

    # Collect ALL dates that should appear on the chart
    all_chart_dates  = list(eval_dates) if len(eval_dates) > 0 else []
    all_chart_values = []

    if not admin_df.empty:
        all_chart_dates += list(pd.to_datetime(admin_df['date'].values))
    if not log_df.empty:
        all_chart_dates += list(pd.to_datetime(log_df['target_date'].values))

    # Dynamic figure width — wider when more dates exist
    n_dates   = max(len(all_chart_dates), 10)
    fig_width = max(14, n_dates * 0.55)   # ~0.55 inch per date label

    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=140)
    bg = '#0f172a'
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # 5a. Test-set actual — solid green
    if len(eval_dates) > 0:
    # Ensure data is 1D to prevent 'y1 is not 1-dimensional' error
        import numpy as np
        d_pts = np.ravel(eval_dates)
        a_pts = np.ravel(eval_actual)

        ax.plot(d_pts, a_pts,
                color='#10b981', lw=2,
                marker='o', markersize=4,
                markerfacecolor='#065f46', markeredgecolor='#10b981',
                label='Actual Price (Test Set)', zorder=4)
                
        # Use the flattened variables here specifically
        ax.fill_between(d_pts, a_pts, color='#10b981', alpha=0.07)
    # 5b. Test-set predicted — dashed orange
    if len(eval_dates) > 0:
        mape_label = (
            f"Model Predicted (Test) — MAPE {active_model.mape:.1f}%"
            if (active_model and active_model.mape)
            else "Model Predicted (Test)"
        )
        ax.plot(eval_dates, eval_preds,
                color='#fb923c', lw=1.8, linestyle='--',
                marker='x', markersize=5, markeredgewidth=1.5,
                label=mape_label, zorder=5, alpha=0.95)

    # 5c. Vertical separator — where eval ends
    if training_cutoff_date is not None:
        ax.axvline(x=training_cutoff_date,
                   color='#334155', linestyle=':', lw=1.5, alpha=0.8, zorder=2)
        ax.text(
            training_cutoff_date, 1.01, ' ← eval end',
            transform=ax.get_xaxis_transform(),
            color='#64748b', fontsize=7, va='bottom',
        )

    # 5d. Admin new data — cyan solid line + dots
    if not admin_df.empty:
        # Bridge from last eval point
        if len(eval_dates) > 0:
            bx = [eval_dates[-1],                  admin_df['date'].iloc[0]]
            by = [eval_actual[-1],                 admin_df['farmgate_price'].iloc[0]]
            ax.plot(bx, by, color='#22d3ee', lw=1, linestyle=':', alpha=0.45, zorder=3)

        ax.plot(admin_df['date'], admin_df['farmgate_price'],
                color='#22d3ee', lw=2, linestyle='-',
                marker='o', markersize=6,
                markerfacecolor='#0e7490', markeredgecolor='#22d3ee',
                markeredgewidth=1.3,
                label='Admin New Data', zorder=6)

    # 5e. User forecast log — dashed rose
    if not log_df.empty:
        if not admin_df.empty:
            last_x = admin_df['date'].iloc[-1]
            last_y = admin_df['farmgate_price'].iloc[-1]
        elif len(eval_dates) > 0:
            last_x = eval_dates[-1]
            last_y = eval_preds[-1]
        else:
            last_x = last_y = None

        if last_x is not None:
            ax.plot([last_x, log_df['target_date'].iloc[0]],
                    [last_y, log_df['price_predicted'].iloc[0]],
                    color='#f43f5e', lw=1, linestyle=':', alpha=0.4, zorder=3)

        ax.plot(log_df['target_date'], log_df['price_predicted'],
                color='#f43f5e', lw=1.8, linestyle='--',
                marker='o', markersize=4,
                label='User Forecast Log', zorder=7, alpha=0.9)

    # ── 6. X-axis: ONE tick per specific date ─────────────────────────────────
    all_unique_dates = sorted(set(all_chart_dates))

    if all_unique_dates:
        ax.set_xticks(all_unique_dates)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                 fontsize=6.5, color='#94a3b8')
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                 fontsize=8, color='#94a3b8')

    # Tight x-limits — no 2021-2024 padding
    if all_unique_dates:
        pad = timedelta(days=3)
        ax.set_xlim(all_unique_dates[0] - pad, all_unique_dates[-1] + pad)

    # ── 7. Styling ────────────────────────────────────────────────────────────
    ax.tick_params(axis='y', colors='#94a3b8', labelsize=9)
    ax.set_ylabel('Farmgate Price (₱)', color='#94a3b8', fontsize=10)
    ax.yaxis.grid(True, color='#1e293b', linewidth=0.8, alpha=0.7)
    ax.xaxis.grid(True, color='#1e293b', linewidth=0.5, alpha=0.25)
    for s in ax.spines.values():
        s.set_visible(False)

    legend = ax.legend(facecolor='#1e293b', edgecolor='#334155',
                       loc='upper right', fontsize=8)
    plt.setp(legend.get_texts(), color='#cbd5e1')
    plt.tight_layout(pad=1.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=bg)
    graph = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # ── 8. Summary cards ──────────────────────────────────────────────────────
    eval_summary = None
    if active_model and active_model.mape is not None:
        eval_summary = {
            'name':     active_model.name,
            'order':    f"({active_model.p},{active_model.d},{active_model.q})",
            'mape':     round(active_model.mape,  2),
            'mae':      round(active_model.mae,   2) if active_model.mae   else '—',
            'rmse':     round(active_model.rmse,  2) if active_model.rmse  else '—',
            'aic':      round(active_model.aic,   2) if active_model.aic   else '—',
            'accuracy': round(100 - active_model.mape, 2),
        }

    return render(request, 'forecast_copra/historical_trend.html', {
        'graph':        graph,
        'eval_summary': eval_summary,
        'eval_rows':    eval_rows,
        'admin_rows':   admin_rows,
        'log_rows':     log_rows,
    })

def get_live_data_api(request):
    """API endpoint for mobile to get live market data.

    GET /api/live-data/
    Returns JSON with live_oil and live_peso.
    """
    if request.method != 'GET':
        return JsonResponse({'success': False, 'error': 'GET method required'})

    try:
        live_oil, live_peso = get_all_live_data()
        return JsonResponse({
            'success': True,
            'live_oil': live_oil,
            'live_peso': live_peso,
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
