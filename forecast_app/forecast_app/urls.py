from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin-panel/', include('forecast_copra.urls_admin')),
    path('admin/', admin.site.urls),
    path('', include('forecast_copra.urls_public')),
]