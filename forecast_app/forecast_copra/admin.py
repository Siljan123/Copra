import pandas as pd
from django.contrib import admin, messages
from django.contrib.auth.admin import UserAdmin
from django.urls import path
from django.shortcuts import render, redirect
from django.template.response import TemplateResponse
from django.http import HttpResponseRedirect
from django.utils.safestring import mark_safe
from .models import User, TrainingData, TrainedModel, ForecastLog, ExcelUpload, DOEDieselUpload, DOEDieselPrice
from .doe_parser import parse_doe_pdf  # Import your parser


@admin.register(TrainingData)
class TrainingDataAdmin(admin.ModelAdmin):
    list_display = ('date', 'farmgate_price', 'oil_price_trend', 'peso_dollar_rate', 'diesel_price', 'labor_min_wage')
    list_filter = ('date',)
    search_fields = ('date',)
    date_hierarchy = 'date'
    ordering = ('-date',)
    list_editable = ('farmgate_price', 'oil_price_trend', 'peso_dollar_rate', 'diesel_price', 'labor_min_wage')
    list_per_page = 25
    show_full_result_count = True


@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'mae', 'mape', 'rmse', 'aic', 'training_date', 'is_active', 'p', 'd', 'q','plot_actual', 'plot_preds')
    list_filter = ('is_active', 'training_date')
    readonly_fields = ('training_date',)
    list_per_page = 20
    fieldsets = (
        ('Model Info', {
            'fields': ('name', 'is_active')
        }),
        ('ARIMA Parameters', {
            'fields': ('p', 'd', 'q'),
            'classes': ('collapse',),
        }),
        ('Performance Metrics', {
            'fields': ('mae', 'mape', 'rmse', 'aic'),
            'classes': ('collapse',),
        }),
        ('Metadata', {
            'fields': ('training_date',),
            'classes': ('collapse',),
        }),
    )
    actions = ['activate_selected_models']

    def has_add_permission(self, request):
        return False

    def activate_selected_models(self, request, queryset):
        TrainedModel.objects.update(is_active=False)
        queryset.update(is_active=True)
        self.message_user(request, "✅ Selected model has been activated.")
    activate_selected_models.short_description = "✅ Activate selected model"


@admin.register(ForecastLog)
class ForecastLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'get_model_name', 'price_predicted', 'forecast_horizon', 'created_at')
    list_filter = ('created_at', 'model_used')
    search_fields = ('model_used__name',)
    readonly_fields = (
        'model_used',
        'created_at',
        'forecast_horizon',
        'farmer_input_oil_price_trend',
        'farmer_input_peso_dollar_rate',
        'farmer_input_diesel_price',
        'farmer_input_labor_min_wage',
        'price_predicted',
    )
    list_per_page = 30
    date_hierarchy = 'created_at'
    fieldsets = (
        ('Forecast Result', {
            'fields': ('model_used', 'forecast_horizon', 'price_predicted')
        }),
        ('Farmer Inputs', {
            'fields': ('farmer_input_oil_price_trend', 'farmer_input_peso_dollar_rate', 'farmer_input_diesel_price', 'farmer_input_labor_min_wage'),
            'classes': ('collapse',),
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',),
        }),
    )

    @admin.display(description='Model Used', ordering='model_used__name')
    def get_model_name(self, obj):
        return obj.model_used.name if obj.model_used else '—'

    def has_add_permission(self, request):
        return False


@admin.register(ExcelUpload)
class ExcelUploadAdmin(admin.ModelAdmin):
    list_display = ('id', 'file', 'uploaded_at', 'processed', 'rows_imported')
    list_filter = ('processed', 'uploaded_at')
    readonly_fields = ('uploaded_at', 'processed', 'rows_imported')
    list_per_page = 20
    fieldsets = (
        ('Upload File', {
            'fields': ('file',)
        }),
        ('Processing Info', {
            'fields': ('uploaded_at', 'processed', 'rows_imported'),
            'classes': ('collapse',),
        }),
    )

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        try:
            df = pd.read_excel(obj.file.path)
            df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
            count = 0
            for _, row in df.iterrows():
                # Build defaults dict with available columns
                defaults = {
                    'farmgate_price': row.get('farmgate_price'),
                    'oil_price_trend': row.get('oil_price_trend'),
                    'peso_dollar_rate': row.get('peso_dollar_rate'),
                }
                # Add new factors if present in Excel
                if 'diesel_price' in row and pd.notna(row['diesel_price']):
                    defaults['diesel_price'] = row['diesel_price']
                if 'labor_min_wage' in row and pd.notna(row['labor_min_wage']):
                    defaults['labor_min_wage'] = row['labor_min_wage']
                
                TrainingData.objects.update_or_create(
                    date=row['date'],
                    defaults=defaults
                )
                count += 1
            obj.processed = True
            obj.rows_imported = count
            obj.save()
            messages.success(request, f"✅ Successfully imported {count} rows into Training Data.")
        except Exception as e:
            messages.error(request, f"❌ Failed to process file: {e}")


# ═══════════════════════════════════════════════════════════════════
# DOE Diesel PDF Upload Admin (with auto-parsing)
# ═══════════════════════════════════════════════════════════════════

@admin.register(DOEDieselUpload)
class DOEDieselUploadAdmin(admin.ModelAdmin):
    list_display = ['id', 'start_date', 'end_date', 'uploaded_at', 'processed', 'display_provincial_avg', 'price_count']
    list_filter = ['processed', 'uploaded_at']
    fields = ['pdf_file', 'start_date', 'end_date']
    
    def save_model(self, request, obj, form, change):
        """Auto-process PDF when saved"""
        super().save_model(request, obj, form, change)
        if not obj.processed:
            self.process_pdf(obj, request)
    
    def process_pdf(self, upload, request):
        """Process the PDF and extract prices"""
        try:
            # Read PDF file
            pdf_bytes = upload.pdf_file.read()
            
            # Parse PDF using your parser
            prices_data = parse_doe_pdf(pdf_bytes, upload.period_label)
            
            if not prices_data:
                messages.warning(request, f'⚠ No price data found in {upload.pdf_file.name}')
                upload.notes = 'No price data found in PDF'
                upload.save()
                return
            
            # Delete existing prices for this upload
            upload.prices.all().delete()
            
            # Create new price entries (only average now)
            for price_data in prices_data:
                DOEDieselPrice.objects.create(
                    upload=upload,
                    municipality=price_data['municipality'],
                    average=price_data['average'],
                    period_label=upload.period_label,
                    start_date=upload.start_date,
                    end_date=upload.end_date
                )
            
            # Calculate provincial average
            averages = [p['average'] for p in prices_data if p['average']]
            provincial_avg = sum(averages) / len(averages) if averages else 0
            
            # Mark as processed
            upload.processed = True
            upload.notes = f'✅ Success! Extracted {len(prices_data)} municipalities. Provincial Avg: ₱{provincial_avg:.2f}/L'
            upload.save()
            
            # Success message
            messages.success(
                request, 
                mark_safe(
                    f'✅ <strong>Success!</strong> Extracted {len(prices_data)} municipalities.<br>'
                    f'📊 <strong>Provincial Average: ₱{provincial_avg:.2f}/L</strong><br>'
                    f'📅 Period: {upload.period_label or "Not specified"}'
                )
            )
            
        except Exception as e:
            messages.error(request, f'❌ Error processing PDF: {str(e)}')
            upload.notes = f'Error: {str(e)}'
            upload.save()
    
    def display_provincial_avg(self, obj):
        """Show provincial average in list view"""
        if obj.processed:
            prices = obj.prices.all()
            if prices:
                averages = [p.average for p in prices if p.average]
                if averages:
                    provincial_avg = sum(averages) / len(averages)
                    return mark_safe(f'<strong style="color: #28a745;">₱{provincial_avg:.2f}/L</strong>')
        return '—'
    display_provincial_avg.short_description = 'Provincial Average'
    
    def price_count(self, obj):
        return obj.prices.count()
    price_count.short_description = '# Municipalities'


@admin.register(DOEDieselPrice)
class DOEDieselPriceAdmin(admin.ModelAdmin):
    list_display = ['municipality', 'average', 'start_date', 'end_date', 'upload', 'created_at']  # Removed price_low and price_high
    list_filter = ['municipality', 'upload', 'created_at']
    search_fields = ['municipality', 'start_date', 'end_date']
    readonly_fields = ['created_at']
    list_per_page = 50
    
    fieldsets = (
        ('Location', {
            'fields': ('municipality', 'province')
        }),
        ('Price Information', {
            'fields': ('average',)
        }),
        ('Metadata', {
            'fields': ('upload', 'period_label', 'created_at'),
            'classes': ('collapse',),
        }),
    )