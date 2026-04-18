from django.urls import path
from . import views
from . import api_views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('', views.home, name='home'),
    path('recent-forecasts/', views.recent_forecasts, name='recent_forecasts'),
    path("historical/trend/", views.historical_trend, name="historical_trend"),
    path('api/live-data/', views.get_live_data_api, name='get_live_data_api'),
    path('api/historical-trend/', api_views.historical_trend_api, name='historical_trend_api'),
    path('api/forecast/', api_views.forecast_api, name='forecast_api'),
    path('api/forecasts/recent/', api_views.recent_forecasts_api, name='recent_forecasts_api'),
]   
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)