import pandas as pd
from django.contrib import admin, messages
from django.contrib.auth.admin import UserAdmin
from .models import User, TrainingData, TrainedModel, ForecastLog, ExcelUpload


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