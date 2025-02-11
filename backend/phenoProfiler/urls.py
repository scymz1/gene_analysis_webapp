from django.contrib import admin
from django.urls import path, include
from .views import MorphologyProfileView

app_name = 'phenoProfiler'

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('phenoProfiler/', include('phenoProfiler.urls')),
    path('analyze/', MorphologyProfileView.as_view(), name='analyze'),
]