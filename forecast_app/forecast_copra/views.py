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
import json
from io import BytesIO

from .forms import ExcelUploadForm, LoginForm, TrainingDataForm, ForecastForm
from .models import TrainingData, TrainedModel, ForecastLog, ExcelUpload
from .utils.arimax_model import ARIMAXModel

# ====================
# PUBLIC VIEWS
# ====================

def home(request):
    """Home page with forecast form"""

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
                forecast_horizon = int(form.cleaned_data['forecast_horizon'])

                # ── Run Forecast ─────────────────────────────────────────
                forecast_result = arimax.forecast(
                    steps=forecast_horizon,
                    use_latest_values=True,
                    latest_oil=oil_price,
                    latest_peso=peso_dollar,
                )

                # ── Extract Final Predicted Price ────────────────────────
                if hasattr(forecast_result, 'iloc'):
                    predicted_price = float(forecast_result.iloc[-1])
                else:
                    predicted_price = float(forecast_result[-1])

                # ── Log Forecast ─────────────────────────────────────────
                ForecastLog.objects.create(
                    forecast_horizon=forecast_horizon,
                    farmer_input_oil_price_trend=oil_price,
                    farmer_input_peso_dollar_rate=peso_dollar,
                    price_predicted=predicted_price,
                )

                # ── Forecast Dates ───────────────────────────────────────
                forecast_dates = pd.date_range(
                    start=datetime.now().date(),
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

                    # ────────────────────────────────────────────────────
                    # RECOMMENDATIONS (Objective 2)
                    # ────────────────────────────────────────────────────

                    # 1. OPTIMAL SELLING TIME
                    recommendations.append(
                        f"📅 OPTIMAL SELLING TIME: Based on the forecast, the best time to sell "
                        f"your copra is on <strong>{best_day_date}</strong> with an estimated price of "
                        f"<strong>₱{best_day_price:.2f}/kg</strong>. "
                        f"This is the highest projected price within your {forecast_horizon}-day forecast window."
                    )

                    # 2. SELL NOW OR WAIT?
                    if trend == 'increasing':
                        recommendations.append(
                            f"⏳ SELL OR WAIT: Prices are trending <strong>upward</strong>. "
                            f"Waiting until <strong>{best_day_date}</strong> could give you "
                            f"₱{best_day_price - start_price:.2f}/kg more than selling today. "
                            f"Only wait if your copra is properly dried and stored."
                        )
                    elif trend == 'decreasing':
                        recommendations.append(
                            f"🚨 SELL OR WAIT: Prices are trending <strong>downward</strong>. "
                            f"It is advised to <strong>sell as soon as possible</strong> to protect your income. "
                            f"Delaying may result in ₱{start_price - end_price:.2f}/kg loss."
                        )
                    else:
                        recommendations.append(
                            f"📊 SELL OR WAIT: Prices are <strong>stable</strong> with minimal change expected. "
                            f"You can sell at your convenience. Focus on reducing transport and "
                            f"handling costs to maximize your net income."
                        )

                    # 3. RISK ADVISORY
                    if volatility > 15:
                        recommendations.append(
                            f"⚠️ RISK ADVISORY: Price volatility is <strong>HIGH ({volatility:.1f}%)</strong>. "
                            f"Avoid selling all your copra on a single day. "
                            f"Split your sales across multiple days to reduce the risk of selling at a low point."
                        )
                    elif volatility > 7:
                        recommendations.append(
                            f"⚠️ RISK ADVISORY: Moderate price fluctuations detected ({volatility:.1f}% volatility). "
                            f"Monitor daily oil price and peso-dollar rate changes before finalizing "
                            f"your selling schedule."
                        )
                    else:
                        recommendations.append(
                            f"✅ RISK ADVISORY: Price forecast is <strong>stable "
                            f"(low volatility: {volatility:.1f}%)</strong>. "
                            f"You can confidently plan your transport and logistics ahead of time."
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
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                messages.error(request, f'Error making forecast: {str(e)}')
                return redirect('home')

    else:
        form = ForecastForm()

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
        'model_available': model_available,
        'active_model':    active_model,
        'model_info':      model_info,
    })

def recent_forecasts(request):
    """View all recent forecasts"""
    forecasts = ForecastLog.objects.all().order_by('-created_at')[:100]
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
                                    peso_dollar_rate=item['peso_dollar_rate']
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
            
        if processed_data and len(processed_data) > 0 and ('diagnose' in request.POST or 'excel_train' in request.POST):
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
        if processed_data and len(processed_data) > 0 and 'excel_train' in request.POST:
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
                        
                        if actual and preds and not is_deployment:
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
                            aic=aic_val
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
            'peso_dollar_rate': ['peso_dollar_rate', 'exchange rate', 'peso dollar', 'exchange', 'peso_dollar', 'pesodollar']
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
                
                processed_data.append({
                    'date': date_obj,
                    'farmgate_price': farmgate_price,
                    'oil_price_trend': oil_price_trend,
                    'peso_dollar_rate': peso_dollar_rate
                })
                
            except Exception as e:
                error_rows.append(f"Row {index+2}: {str(e)}")
                continue
        
        if error_rows:
            print(f"Excel processing warnings: {error_rows}")
        
        return processed_data, "Success"
        
    except Exception as e:
        return [], f"Error processing Excel file: {str(e)}"


def get_forecast_api(request):
    """API endpoint for forecast"""
    if request.method == 'POST':
        try:
            oil_price = float(request.POST.get('oil_price_trend'))
            peso_dollar = float(request.POST.get('peso_dollar_rate'))
            forecast_horizon = int(request.POST.get('forecast_horizon'))
            
            # Get active model
            active_model = TrainedModel.objects.get(is_active=True)
            
            # Load model and make prediction
            arimax = ARIMAXModel()
            arimax.load_model(active_model.model_file_path)
            
            # Create exogenous array
            exog_future = np.array([[oil_price, peso_dollar]] * forecast_horizon)
            
            # Make forecast
            forecast_result = arimax.forecast(steps=forecast_horizon, exog_future=exog_future)
            
            # Get the predicted price
            if hasattr(forecast_result, 'iloc'):
                predicted_price = float(forecast_result.iloc[-1])
            else:
                predicted_price = float(forecast_result[-1])
            
            # Save to logs
            ForecastLog.objects.create(
                forecast_horizon=forecast_horizon,
                farmer_input_oil_price_trend=oil_price,
                farmer_input_peso_dollar_rate=peso_dollar,
                price_predicted=predicted_price
            )
            
            return JsonResponse({
                'success': True,
                'predicted_price': predicted_price,
                'oil_price': oil_price,
                'peso_dollar_rate': peso_dollar,
                'forecast_horizon': forecast_horizon,
                'model_name': active_model.name,
                'accuracy': active_model.accuracy
            })
            
        except TrainedModel.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': 'No trained model available'
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({
        'success': False,
        'error': 'POST method required'
    })