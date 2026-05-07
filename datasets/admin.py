from django.contrib import admin
from .models import DatasetFile, DataColumn, Dataset, UploadRecord, DataLineage, DetectionMethod, DetectionTask, PoisonedRecord


@admin.register(DatasetFile)
class DatasetFileAdmin(admin.ModelAdmin):
    list_display = ['filename', 'file_size', 'checksum', 'data_format', 'row_count', 'column_count', 'created_at']
    search_fields = ['filename', 'checksum']
    list_filter = ['data_format', 'encoding', 'has_header', 'created_at']
    readonly_fields = ['checksum', 'created_at']


@admin.register(DataColumn)
class DataColumnAdmin(admin.ModelAdmin):
    list_display = ['name', 'dataset', 'ordinal', 'column_type', 'is_target', 'is_text', 'null_rate', 'quality_status']
    search_fields = ['name', 'dataset__name']
    list_filter = ['column_type', 'is_target', 'is_text', 'quality_status']
    raw_id_fields = ['dataset']


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'version', 'source_type', 'status', 'poisoning_type', 'total_rows', 'contamination_rate', 'created_at']
    search_fields = ['name', 'description']
    list_filter = ['source_type', 'status', 'poisoning_type', 'domain', 'task_type', 'created_at']
    readonly_fields = ['metadata', 'created_at', 'updated_at']
    raw_id_fields = ['file', 'uploaded_by']


@admin.register(UploadRecord)
class UploadRecordAdmin(admin.ModelAdmin):
    list_display = ['dataset', 'source', 'uploader', 'original_filename', 'file_size', 'ip_address', 'created_at']
    search_fields = ['dataset__name', 'original_filename', 'ip_address']
    list_filter = ['source', 'created_at']
    readonly_fields = ['created_at']
    raw_id_fields = ['dataset', 'uploader']


@admin.register(DataLineage)
class DataLineageAdmin(admin.ModelAdmin):
    list_display = ['dataset', 'operation_type', 'operation_name', 'rows_affected', 'operator', 'created_at']
    search_fields = ['dataset__name', 'operation_name', 'description']
    list_filter = ['operation_type', 'created_at']
    readonly_fields = ['created_at']
    raw_id_fields = ['dataset', 'parent_lineage', 'root_lineage', 'operator']


@admin.register(DetectionMethod)
class DetectionMethodAdmin(admin.ModelAdmin):
    list_display = ['name', 'code', 'method_type', 'version', 'is_active', 'created_at']
    search_fields = ['name', 'code', 'description']
    list_filter = ['method_type', 'is_active', 'created_at']
    readonly_fields = ['created_at']


@admin.register(DetectionTask)
class DetectionTaskAdmin(admin.ModelAdmin):
    list_display = ['name', 'dataset', 'method', 'status', 'total_samples', 'detected_poisoned', 'detection_rate', 'created_at']
    search_fields = ['name', 'dataset__name']
    list_filter = ['method', 'status', 'created_at']
    readonly_fields = ['result_summary', 'detailed_results', 'error_message', 'traceback', 'created_at', 'started_at', 'completed_at']
    raw_id_fields = ['dataset', 'method', 'created_by']


@admin.register(PoisonedRecord)
class PoisonedRecordAdmin(admin.ModelAdmin):
    list_display = ['dataset', 'task', 'row_index', 'original_label', 'predicted_label', 'confidence', 'severity', 'verification_status', 'created_at']
    search_fields = ['dataset__name', 'original_label', 'predicted_label']
    list_filter = ['severity', 'verification_status', 'poisoning_type', 'created_at']
    readonly_fields = ['created_at', 'verified_at']
    raw_id_fields = ['task', 'dataset', 'verified_by']
