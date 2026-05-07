from django.contrib import admin
from .models import AlgorithmConfig, DetectionTask, DetectionResult


@admin.register(DetectionTask)
class DetectionTaskAdmin(admin.ModelAdmin):
    list_display = ['task_no', 'dataset', 'status', 'progress', 'risk_score', 'created_at']
    list_filter = ['status']
    readonly_fields = ['task_no', 'started_at', 'finished_at', 'risk_score']


@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = ['sample_id', 'risk_type', 'confidence', 'suggestion', 'detector_name']
    list_filter = ['risk_type', 'suggestion']


@admin.register(AlgorithmConfig)
class AlgorithmConfigAdmin(admin.ModelAdmin):
    list_display = ['detector_name', 'detector_type', 'enabled', 'weight']
