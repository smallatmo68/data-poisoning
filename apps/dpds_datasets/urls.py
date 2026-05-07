from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import DatasetPreviewView, DatasetUploadView, DatasetViewSet
from apps.preprocessing.views import CreatePreprocessView

router = DefaultRouter()
router.register('', DatasetViewSet, basename='dataset')

urlpatterns = [
    path('upload/', DatasetUploadView.as_view(), name='dataset-upload'),
    path('<uuid:dataset_id>/preview/', DatasetPreviewView.as_view(), name='dataset-preview'),
    path('<uuid:dataset_id>/preprocess/', CreatePreprocessView.as_view(), name='dataset-preprocess'),
    path('', include(router.urls)),
]
