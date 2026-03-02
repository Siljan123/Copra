import pandas as pd
import os
from django.contrib import admin, messages
from django.contrib.auth.admin import UserAdmin
from django.utils import timezone
from .models import User, TrainingData, TrainedModel, ForecastLog, ExcelUpload

@admin.register(TrainingData)
class TrainingDataAdmin(admin.ModelAdmin):
    list_display = ('date', 'farmgate_price', 'oil_price_trend', 'peso_dollar_rate')
    list_filter = ('date',)
    search_fields = ('date',)
    date_hierarchy = 'date'
    ordering = ('-date',)
    list_editable = ('farmgate_price', 'oil_price_trend', 'peso_dollar_rate')
    list_per_page = 25                      # ✅ Pagination for cleaner tables
    show_full_result_count = True

@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'mae', 'mape', 'rmse', 'aic', 'training_date', 'is_active', 'p', 'd', 'q')
    list_filter = ('is_active', 'training_date')
    readonly_fields = ('training_date',)
    list_per_page = 20
    # ✅ Organize the form into sections
    fieldsets = (
        ('Model Info', {
            'fields': ('name', 'is_active')
        }),
        ('ARIMA Parameters', {
            'fields': ('p', 'd', 'q'),
            'classes': ('collapse',),       # Collapsible section
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
        self.message_user(request, "Selected model has been activated.")
    activate_selected_models.short_description = "✅ Activate selected model"

@admin.register(ForecastLog)
class ForecastLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'price_predicted', 'forecast_horizon', 'created_at')
    list_filter = ('created_at',)
    readonly_fields = ('created_at',)
    list_per_page = 30
    date_hierarchy = 'created_at'   
    
    def has_add_permission(self, request):
        return False    # ✅ Date drill-down in list view

@admin.register(ExcelUpload)
class ExcelUploadAdmin(admin.ModelAdmin):
    list_display = ('id', 'file', 'uploaded_at', 'processed', 'rows_imported')
    list_filter = ('processed', 'uploaded_at')
    readonly_fields = ('uploaded_at', 'processed', 'rows_imported')
    list_per_page = 20
    # ✅ Separate editable fields from read-only info
    fieldsets = (
        ('Upload File', {
            'fields': ('file',)
        }),
        ('Processing Info', {
            'fields': ('uploaded_at', 'processed', 'rows_imported'),
            'classes': ('collapse',),
        }),
    )