from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, TrainingData, TrainedModel, ForecastLog
from .models import ExcelUpload

@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'user_type', 'is_staff')
    list_filter = ('user_type', 'is_staff', 'is_superuser')
    fieldsets = UserAdmin.fieldsets + (
        ('User Type', {'fields': ('user_type',)}),
    )

@admin.register(TrainingData)
class TrainingDataAdmin(admin.ModelAdmin):
    list_display = ('date', 'farmgate_price', 'oil_price_trend', 'peso_dollar_rate')
    list_filter = ('date',)
    search_fields = ('date',)
    date_hierarchy = 'date'

@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'model_file_path', 'accuracy', 'training_date', 'is_active')
    list_filter = ('is_active', 'training_date')
    readonly_fields = ('training_date',)
    actions = ['activate_model']

    def activate_model(self, request, queryset):
        queryset.update(is_active=True)
        self.message_user(request, "Selected models activated")
    activate_model.short_description = "Activate selected models"

@admin.register(ForecastLog)
class ForecastLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'farmer', 'price_predicted', 'forecast_horizon', 'created_at')
    list_filter = ('created_at', 'farmer')
    readonly_fields = ('created_at',)
    search_fields = ('farmer__username',)
    
@admin.register(ExcelUpload)
class ExcelUploadAdmin(admin.ModelAdmin):
    list_display = ('id', 'file', 'uploaded_at', 'processed', 'rows_imported')
    list_filter = ('processed', 'uploaded_at')
    readonly_fields = ('uploaded_at', 'processed', 'rows_imported')