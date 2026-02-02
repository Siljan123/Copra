from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('recent-forecasts/', views.recent_forecasts, name='recent_forecasts'),
    path('api/forecast/', views.get_forecast_api, name='forecast_api'),
]