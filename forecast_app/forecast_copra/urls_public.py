from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('', views.home, name='home'),
    path('recent-forecasts/', views.recent_forecasts, name='recent_forecasts'),
    path("historical/trend/", views.historical_trend, name="historical_trend"),
    # path('api/forecast/', views.get_forecast_api, name='forecast_api'),
]
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)