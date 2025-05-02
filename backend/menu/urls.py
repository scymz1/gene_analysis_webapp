# RestaurantCore/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('upload-fasta/', views.upload_fasta, name='upload_fasta'),
    path('clear-cache/', views.clear_cache, name='clear-cache'),
    path('download-result/', views.download_result, name='download_result'),
    path('generate-visualization/', views.generate_visualization, name='generate_visualization'),
]
