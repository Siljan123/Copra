from django.db import models
from django.contrib.auth.models import AbstractUser
import os

class User(AbstractUser):
    USER_TYPE_CHOICES = (
        ('admin', 'Admin'),
        ('farmer', 'Farmer'),
    )
    user_type = models.CharField(max_length=10, choices=USER_TYPE_CHOICES, default='admin')

class TrainingData(models.Model):
    date = models.DateField()
    farmgate_price = models.DecimalField(max_digits=10, decimal_places=2)
    oil_price_trend = models.DecimalField(max_digits=10, decimal_places=2)
    peso_dollar_rate = models.DecimalField(max_digits=10, decimal_places=2)
    diesel_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True, help_text="Diesel price per liter")
    labor_min_wage = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True, help_text="Minimum labor wage")
    
    class Meta:
        verbose_name_plural = "Training Data"
        ordering = ['-date']
    
    def __str__(self):
        return f"Data for {self.date}"

class TrainedModel(models.Model):
    name = models.CharField(max_length=100)
    model_file_path = models.CharField(max_length=255)
    training_date = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)
    
    # ARIMAX parameters
    p = models.IntegerField(default=1, help_text="AR order (autoregressive)")
    d = models.IntegerField(default=1, help_text="I order (differencing)")
    q = models.IntegerField(default=1, help_text="MA order (moving average)")
    
    # Error Metrics
    mae = models.DecimalField(
        max_digits=10, 
        decimal_places=4, 
        null=True, 
        blank=True, 
        help_text="Mean Absolute Error"
    )
    rmse = models.DecimalField(
        max_digits=10, 
        decimal_places=4, 
        null=True, 
        blank=True, 
        help_text="Root Mean Squared Error"
    )
    mape = models.DecimalField(
        max_digits=10, 
        decimal_places=4, 
        null=True, 
        blank=True, 
        help_text="Mean Absolute Percentage Error (%)"
    )
    
    # Information Criterion
    aic = models.DecimalField(
        max_digits=15, 
        decimal_places=4, 
        null=True, 
        blank=True, 
        help_text="Akaike Information Criterion"
    )
    #plots
    plot_actual = models.JSONField(null=True, blank=True)   # list of floats
    plot_preds  = models.JSONField(null=True, blank=True)   # list of floats
    class Meta:
        verbose_name_plural = "Trained Models"
        ordering = ['-training_date']
    
    def __str__(self):
        return f"{self.name} (p={self.p}, d={self.d}, q={self.q})"
    
    def save(self, *args, **kwargs):
        # Logic to ensure only one active model is used for forecasting
        if self.is_active:
            TrainedModel.objects.filter(is_active=True).exclude(pk=self.pk).update(is_active=False)
        super().save(*args, **kwargs)

class ForecastLog(models.Model):
     # ✅ NEW — FK to TrainedModel for auditability
    model_used = models.ForeignKey(
        TrainedModel,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='forecast_logs',
        help_text="The ARIMAX model that produced this forecast"
    )
    forecast_horizon = models.IntegerField(help_text="Number of days to forecast")
    farmer_input_oil_price_trend = models.DecimalField(max_digits=10, decimal_places=2)
    farmer_input_peso_dollar_rate = models.DecimalField(max_digits=10, decimal_places=4)
    farmer_input_diesel_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    farmer_input_labor_min_wage = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    price_predicted = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name_plural = "Forecast Logs"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Forecast #{self.id} - {self.created_at.date()}"

class ExcelUpload(models.Model):
    file = models.FileField(upload_to='excel_uploads/%Y/%m/%d/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    rows_imported = models.IntegerField(default=0)
    
    def __str__(self):
        return f"Excel upload - {self.uploaded_at}"