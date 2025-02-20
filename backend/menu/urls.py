# RestaurantCore/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('upload-csv/', views.upload_csv, name='upload-csv'),
    path('clear-cache/', views.clear_cache, name='clear-cache'),
    path('finetune-model/', views.finetune_model, name='finetune-model'),
    path('download-model/', views.download_model, name='download-model'),
    path('train-fixed-embeddings/', views.train_fixed_embeddings, name='train-fixed-embeddings'),
    path('files/', views.list_files, name='list-files'),
    path('download/', views.download_file, name='download-file'),
    path('bulk-download/', views.bulk_download, name='bulk-download'),
]