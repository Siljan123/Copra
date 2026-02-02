from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # Required for background rendering
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
    if request.method == 'POST':
        form = ForecastForm(request.POST)
        if form.is_valid():
            # Get active model
            try:
                active_model = TrainedModel.objects.get(is_active=True)
            except TrainedModel.DoesNotExist:
                messages.error(request, 'No trained model available. Please check back later.')
                return redirect('home')
            
            try:
                # Load model and make prediction
                arimax = ARIMAXModel()
                arimax.load_model(active_model.model_file_path)
                
                # Prepare exogenous data for forecast
                oil_price = float(form.cleaned_data['oil_price_trend'])
                peso_dollar = float(form.cleaned_data['peso_dollar_rate'])
                forecast_horizon = int(form.cleaned_data['forecast_horizon'])
                
                # Create exogenous array for ARIMAX
                exog_future = np.array([[oil_price, peso_dollar]] * forecast_horizon)
                
                # Make forecast
                forecast_result = arimax.forecast(steps=forecast_horizon, exog_future=exog_future)
                
                # Get the predicted price (last value)
                if hasattr(forecast_result, 'iloc'):  # If it's a pandas Series
                    predicted_price = float(forecast_result.iloc[-1])
                else:  # If it's a numpy array
                    predicted_price = float(forecast_result[-1])
                
                # Save to logs (without user authentication)
                ForecastLog.objects.create(
                    forecast_horizon=forecast_horizon,
                    farmer_input_oil_price_trend=oil_price,
                    farmer_input_peso_dollar_rate=peso_dollar,
                    price_predicted=predicted_price
                )
                
                # Prepare forecast data for display
                forecast_dates = pd.date_range(
                    start=datetime.now().date(), 
                    periods=forecast_horizon, 
                    freq='D'
                ).strftime('%Y-%m-%d').tolist()
                
                if hasattr(forecast_result, 'tolist'):
                    forecast_values = forecast_result.tolist()
                elif hasattr(forecast_result, 'values'):
                    forecast_values = forecast_result.values.tolist()
                else:
                    forecast_values = list(forecast_result)
                
                forecast_data = list(zip(forecast_dates, forecast_values))
                
                return render(request, 'forecast_copra/forecast_result.html', {
                    'predicted_price': predicted_price,
                    'oil_price': oil_price,
                    'peso_dollar_rate': peso_dollar,
                    'forecast_horizon': forecast_horizon,
                    'model_name': active_model.name,
                    'forecast_data': forecast_data,
                    'accuracy': active_model.accuracy
                })
                
            except Exception as e:
                messages.error(request, f'Error making forecast: {str(e)}')
                return redirect('home')
    else:
        form = ForecastForm()
    
    # Check if there's an active model
    try:
        active_model = TrainedModel.objects.get(is_active=True)
        model_available = True
        model_info = f"Active Model: {active_model.name} (Accuracy: {active_model.accuracy}%)"
    except TrainedModel.DoesNotExist:
        model_available = False
        model_info = "No trained model available. Forecasts cannot be made."
    
    # Get recent forecasts for display
    recent_forecasts = ForecastLog.objects.all().order_by('-created_at')[:5]
    
    return render(request, 'forecast_copra/home.html', {
        'form': form,
        'recent_forecasts': recent_forecasts,
        'model_available': model_available,
        'active_model': active_model,
        'model_info': model_info
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
    """Train ARIMAX model - with Excel upload and Visualization"""
    if not request.user.is_staff:
        messages.error(request, 'Only admin users can access this page.')
        return redirect('home')
    
    graph_base64 = None # Initialize empty
    
    if request.method == 'POST':
        processed_data = None
        source_name = ""

        # --- DATA GATHERING ---
        if 'excel_train' in request.POST:
            excel_file = request.FILES.get('excel_file')
            if not excel_file:
                messages.error(request, 'Please select an Excel file.')
                return redirect('train_model')
            
            fs = FileSystemStorage()
            filename = fs.save(f'temp_training/{excel_file.name}', excel_file)
            file_path = fs.path(filename)
            processed_data, message = process_excel_file(file_path) # Assumes this exists
            source_name = "Excel file"
            if os.path.exists(file_path): os.remove(file_path) # Cleanup

        elif 'db_train' in request.POST:
            processed_data = list(TrainingData.objects.all().values())
            source_name = "Database"

        # --- TRAINING LOGIC ---
        if processed_data and len(processed_data) >= 10:
            try:
                # Get parameters
                p = int(request.POST.get('p', 1) or 1)
                d = int(request.POST.get('d', 1) or 1)
                q = int(request.POST.get('q', 1) or 1)

                arimax = ARIMAXModel(order=(p, d, q))
                metrics = arimax.train(processed_data)

                if 'error' in metrics:
                    messages.error(request, f'Training failed: {metrics["error"]}')
                    return redirect('train_model')

                # --- PLOTTING LOGIC ---
                actual = metrics.get('plot_actual', [])
                preds = metrics.get('plot_preds', [])
                
                if actual and preds:
                    plt.figure(figsize=(10, 4))
                    sns.set_style("whitegrid")
                    plt.plot(actual, label='Actual Price', color='#2ecc71', linewidth=2)
                    plt.plot(preds, label='Predicted Price', color='#e74c3c', linestyle='--', linewidth=2)
                    plt.title(f'Model Accuracy: {metrics.get("accuracy", 0):.2f}%')
                    plt.legend()
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    plt.close()
                    graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                # --- SAVE MODEL ---
                model_prefix = "model" if 'excel_train' in request.POST else "db_model"
                model_name = f"{model_prefix}_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
                model_path = arimax.save_model(model_name)

                TrainedModel.objects.update(is_active=False)
                TrainedModel.objects.create(
                    name=model_name,
                    model_file_path=model_path,
                    accuracy=round(metrics.get('accuracy', 0), 2),
                    is_active=True, p=p, d=d, q=q
                )

                messages.success(request, f"✅ Model '{model_name}' trained successfully from {source_name}!")
                
                # CRITICAL: Render instead of redirect to show the graph immediately
                models = TrainedModel.objects.all().order_by('-training_date')
                return render(request, 'forecast_copra/train_model.html', {
                    'models': models,
                    'data_count': TrainingData.objects.count(),
                    'graph': graph_base64,
                    'metrics': metrics
                })

            except Exception as e:
                messages.error(request, f'❌ Error: {str(e)}')
                return redirect('train_model')

    # Default GET state
    models = TrainedModel.objects.all().order_by('-training_date')
    return render(request, 'forecast_copra/train_model.html', {
        'models': models,
        'data_count': TrainingData.objects.count(),
        'graph': None
    })
    
@login_required
def trained_models_view(request):
    # Get all models (latest first)
    model_list = TrainedModel.objects.all().order_by('-training_date')

    # Pagination
    paginator = Paginator(model_list, 10)  # 10 per page
    page_number = request.GET.get('page')
    models = paginator.get_page(page_number)

    return render(request, "forecast_copra/trained_models.html", {
        "models": models
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
                # Extract data with error handling
                date_str = str(row[actual_columns['date']])
                
                # Try to convert date in various formats
                date_obj = None
                date_formats = [
                    '%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y', 
                    '%d-%m-%Y', '%Y.%m.%d', '%d.%m.%Y', '%m.%d.%Y',
                    '%Y%m%d', '%d%m%Y', '%m%d%Y'
                ]
                
                for fmt in date_formats:
                    try:
                        date_obj = datetime.strptime(str(date_str).strip(), fmt).date()
                        break
                    except ValueError:
                        continue
                
                if not date_obj:
                    error_rows.append(f"Row {index+2}: Could not parse date '{date_str}'")
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