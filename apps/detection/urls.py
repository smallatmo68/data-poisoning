from django.urls import path
from .views import (
    CreateDetectionTaskView,
    DetectionTaskDetailView,
    DetectionTaskProgressView,
    DetectionResultsView,
    SampleDetailView,
    AlgorithmConfigView,
    DetectorsView,
    TaskAnalysisView,
)

urlpatterns = [
    path('tasks/', CreateDetectionTaskView.as_view(), name='detection-task-create'),
    path('tasks/<uuid:task_id>/', DetectionTaskDetailView.as_view(), name='detection-task-detail'),
    path('tasks/<uuid:task_id>/progress/', DetectionTaskProgressView.as_view(), name='detection-task-progress'),
    path('tasks/<uuid:task_id>/results/', DetectionResultsView.as_view(), name='detection-task-results'),
    path('tasks/<uuid:task_id>/analysis/', TaskAnalysisView.as_view(), name='detection-task-analysis'),
    path('samples/<uuid:result_id>/', SampleDetailView.as_view(), name='detection-sample-detail'),
    path('detectors/', DetectorsView.as_view(), name='detectors-metadata'),
    path('algorithm-params/', AlgorithmConfigView.as_view(), name='algorithm-config'),
]
