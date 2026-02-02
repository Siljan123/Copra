from django.db import models
from django.contrib.auth.models import AbstractUser
import os

class User(AbstractUser):
    USER_TYPE_CHOICES = (
        ('admin', 'Admin'),
        ('farmer', 'Farmer'),
    )
    user_type = models.CharField(max_length=10, choices=USER_TYPE_CHOICES, default='farmer')

class TrainingData(models.Model):
    date = models.DateField()
    farmgate_price = models.DecimalField(max_digits=10, decimal_places=2)
    oil_price_trend = models.DecimalField(max_digits=10, decimal_places=2)
    peso_dollar_rate = models.DecimalField(max_digits=10, decimal_places=2)
    
    class Meta:
        verbose_name_plural = "Training Data"
        ordering = ['-date']
    
    def __str__(self):
        return f"Data for {self.date}"

class TrainedModel(models.Model):
    name = models.CharField(max_length=100)
    model_file_path = models.CharField(max_length=255)
    accuracy = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    training_date = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)
    # ARIMAX parameters
    p = models.IntegerField(default=1, help_text="AR order (autoregressive)")
    d = models.IntegerField(default=1, help_text="I order (differencing)")
    q = models.IntegerField(default=1, help_text="MA order (moving average)")
    
    class Meta:
        verbose_name_plural = "Trained Models"
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        # Ensure only one active model exists
        if self.is_active:
            TrainedModel.objects.filter(is_active=True).update(is_active=False)
        super().save(*args, **kwargs)

class ForecastLog(models.Model):
    farmer = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)  # Fixed indentation
    forecast_horizon = models.IntegerField(help_text="Number of days to forecast")
    farmer_input_oil_price_trend = models.DecimalField(max_digits=10, decimal_places=2)
    farmer_input_peso_dollar_rate = models.DecimalField(max_digits=10, decimal_places=2)
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