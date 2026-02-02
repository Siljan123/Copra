from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.admin_login, name='admin_login'),
    path('logout/', views.admin_logout, name='admin_logout'),
    path('dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('manage-data/', views.manage_data, name='manage_data'),
    path('train-model/', views.train_model, name='train_model'),
    path("trained-models/", views.trained_models_view, name="trained_models_view"),
    path('model/<int:model_id>/activate/', views.activate_model, name='activate_model'),
    path('model/<int:model_id>/deactivate/', views.deactivate_model, name='deactivate_model'),
    path('model/<int:model_id>/delete/', views.delete_model, name='delete_model'),
]