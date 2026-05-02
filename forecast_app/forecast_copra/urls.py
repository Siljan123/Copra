from django.urls import path
from . import views
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    # Public routes
    path('', views.home, name='home'),
    path('recent-forecasts/', views.recent_forecasts, name='recent_forecasts'),
    # path('api/forecast/', views.get_forecast_api, name='forecast_api'),
    path("__reload__/", include("django_browser_reload.urls")),
    path("historical/trend/", views.historical_trend, name="historical_trend"),
    # Admin routes
    path('admin-panel/login/', views.admin_login, name='admin_login'),
    path('admin-panel/logout/', views.admin_logout, name='admin_logout'),
    path('admin-panel/dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin-panel/manage-data/', views.manage_data, name='manage_data'),
    path('admin-panel/train-model/', views.train_model, name='train_model'),
    path('admin-panel/trained-models/', views.trained_models_view, name='trained_models_view'),
    path('admin-panel/model/<int:model_id>/activate/', views.activate_model, name='activate_model'),
    path('admin-panel/model/<int:model_id>/deactivate/', views.deactivate_model, name='deactivate_model'),
    path('admin-panel/model/<int:model_id>/delete/', views.delete_model, name='delete_model'),
    path('admin-panel/models/bulk-delete/', views.bulk_delete_models, name='bulk_delete_models'),
]
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    


    # ── Forecast Dates ──Back test
                # forecast_start = (
                #     latest_farmgate_date + timedelta(days=1)
                #     if latest_farmgate_date
                #     else datetime.now().date()
                # )