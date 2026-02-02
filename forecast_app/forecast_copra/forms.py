from django import forms
from django.contrib.auth.forms import AuthenticationForm
from .models import TrainingData

class LoginForm(AuthenticationForm):
    username = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter username',
            'required': 'required'
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter password',
            'required': 'required'
        })
    )
class TrainingDataForm(forms.ModelForm):
    class Meta:
        model = TrainingData
        fields = ['date', 'farmgate_price', 'oil_price_trend', 'peso_dollar_rate']
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
            'farmgate_price': forms.NumberInput(attrs={'class': 'form-control'}),
            'oil_price_trend': forms.NumberInput(attrs={'class': 'form-control'}),
            'peso_dollar_rate': forms.NumberInput(attrs={'class': 'form-control'}),
        }

class ForecastForm(forms.Form):
    oil_price_trend = forms.DecimalField(
        label='Oil Price Trend',
        max_digits=10,
        decimal_places=2,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    peso_dollar_rate = forms.DecimalField(
        label='Peso-Dollar Rate',
        max_digits=10,
        decimal_places=2,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    forecast_horizon = forms.IntegerField(
        label='Forecast Horizon (days)',
        min_value=1,
        max_value=365,
        initial=30,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
class ExcelUploadForm(forms.Form):
    excel_file = forms.FileField(
        label='Select Excel File',
        help_text='Upload Excel file with columns: date, farmgate_price, oil_price_trend, peso_dollar_rate'
    )
    
    def clean_excel_file(self):
        excel_file = self.cleaned_data['excel_file']
        # Check file extension
        if not excel_file.name.endswith(('.xlsx', '.xls')):
            raise forms.ValidationError('Only Excel files are allowed (.xlsx, .xls)')
        return excel_file