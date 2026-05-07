from django.urls import path
from .views import CreatePreprocessView, PreprocessDetailView

urlpatterns = [
    path('<uuid:preprocess_id>/', PreprocessDetailView.as_view(), name='preprocess-detail'),
]
