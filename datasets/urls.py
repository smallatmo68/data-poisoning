from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    DatasetFileViewSet, DataColumnViewSet, DatasetViewSet,
    UploadRecordViewSet, DataLineageViewSet, DetectionMethodViewSet,
    DetectionTaskViewSet, PoisonedRecordViewSet,
    DatasetUploadView, BuiltinDatasetImportView, DetectionReportView
)

router = DefaultRouter()
router.register(r'files', DatasetFileViewSet, basename='datasetfile')
router.register(r'columns', DataColumnViewSet, basename='datacolumn')
router.register(r'datasets', DatasetViewSet, basename='dataset')
router.register(r'uploads', UploadRecordViewSet, basename='uploadrecord')
router.register(r'lineages', DataLineageViewSet, basename='datalineage')
router.register(r'methods', DetectionMethodViewSet, basename='detectionmethod')
router.register(r'tasks', DetectionTaskViewSet, basename='detectiontask')
router.register(r'poisoned', PoisonedRecordViewSet, basename='poisonedrecord')

urlpatterns = [
    path('', include(router.urls)),
    path('upload/', DatasetUploadView.as_view(), name='dataset-upload'),
    path('import-builtin/', BuiltinDatasetImportView.as_view(), name='import-builtin'),
    path('report/<int:task_id>/', DetectionReportView.as_view(), name='detection-report'),
]
