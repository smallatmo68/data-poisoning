from django.urls import path
from .views import (
    ApplyDefenseView,
    BatchSampleActionView,
    DownloadCleanDatasetView,
    SampleActionView,
    SampleDetailView,
)

urlpatterns = [
    path('tasks/<uuid:task_id>/apply/', ApplyDefenseView.as_view(), name='defense-apply'),
    path('samples/<uuid:result_id>/action/', SampleActionView.as_view(), name='sample-action'),
    path('samples/<uuid:result_id>/detail/', SampleDetailView.as_view(), name='sample-detail'),
    path('samples/batch-action/', BatchSampleActionView.as_view(), name='batch-sample-action'),
    path('<uuid:clean_result_id>/download/', DownloadCleanDatasetView.as_view(), name='defense-download'),
]
