from rest_framework import serializers
from django.contrib.auth.models import User
from .models import (
    DatasetFile, DataColumn, Dataset, UploadRecord, DataLineage,
    DetectionMethod, DetectionTask, PoisonedRecord
)


class DatasetFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatasetFile
        fields = [
            'id', 'filename', 'file_path', 'file_size', 'checksum',
            'data_format', 'row_count', 'column_count', 'encoding',
            'delimiter', 'has_header', 'created_at'
        ]


class DataColumnSerializer(serializers.ModelSerializer):
    column_type_display = serializers.CharField(source='get_column_type_display', read_only=True)
    quality_status_display = serializers.CharField(source='get_quality_status_display', read_only=True)

    class Meta:
        model = DataColumn
        fields = [
            'id', 'name', 'ordinal', 'column_type', 'column_type_display',
            'is_target', 'is_text', 'null_count', 'null_rate',
            'unique_count', 'unique_rate', 'avg_length',
            'min_value', 'max_value', 'label_distribution',
            'quality_status', 'quality_status_display', 'description'
        ]


class UploadRecordSerializer(serializers.ModelSerializer):
    source_display = serializers.CharField(source='get_source_display', read_only=True)
    uploader_username = serializers.CharField(source='uploader.username', read_only=True)

    class Meta:
        model = UploadRecord
        fields = [
            'id', 'dataset', 'source', 'source_display',
            'uploader', 'uploader_username', 'original_filename',
            'file_size', 'ip_address', 'user_agent', 'notes', 'created_at'
        ]


class DataLineageSerializer(serializers.ModelSerializer):
    operation_type_display = serializers.CharField(source='get_operation_type_display', read_only=True)
    operator_username = serializers.CharField(source='operator.username', read_only=True)

    class Meta:
        model = DataLineage
        fields = [
            'id', 'dataset', 'operation_type', 'operation_type_display',
            'operation_name', 'description', 'input_datasets', 'output_datasets',
            'parent_lineage', 'root_lineage', 'operator', 'operator_username',
            'parameters', 'result_summary', 'rows_affected', 'created_at'
        ]


class DatasetSerializer(serializers.ModelSerializer):
    source_type_display = serializers.CharField(source='get_source_type_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    poisoning_type_display = serializers.CharField(source='get_poisoning_type_display', read_only=True)
    uploaded_by_username = serializers.CharField(source='uploaded_by.username', read_only=True)
    file_info = DatasetFileSerializer(source='file', read_only=True)
    column_count_detail = serializers.SerializerMethodField()
    lineage_count = serializers.SerializerMethodField()
    detection_count = serializers.SerializerMethodField()

    class Meta:
        model = Dataset
        fields = [
            'id', 'name', 'version', 'source_type', 'source_type_display',
            'status', 'status_display', 'poisoning_type', 'poisoning_type_display',
            'file', 'file_info', 'description', 'domain', 'task_type',
            'total_rows', 'clean_rows', 'poisoned_rows', 'contamination_rate',
            'tags', 'metadata', 'uploaded_by', 'uploaded_by_username',
            'created_at', 'updated_at', 'column_count_detail', 'lineage_count', 'detection_count'
        ]
        read_only_fields = ['created_at', 'updated_at']

    def get_column_count_detail(self, obj):
        return obj.columns.count()

    def get_lineage_count(self, obj):
        return obj.lineage_records.count()

    def get_detection_count(self, obj):
        return obj.detection_tasks.count()


class DatasetDetailSerializer(DatasetSerializer):
    columns = DataColumnSerializer(many=True, read_only=True)
    upload_records = UploadRecordSerializer(many=True, read_only=True)
    lineages = DataLineageSerializer(many=True, read_only=True)

    class Meta(DatasetSerializer.Meta):
        fields = DatasetSerializer.Meta.fields + ['columns', 'upload_records', 'lineages']


class DetectionMethodSerializer(serializers.ModelSerializer):
    method_type_display = serializers.CharField(source='get_method_type_display', read_only=True)
    task_count = serializers.SerializerMethodField()

    class Meta:
        model = DetectionMethod
        fields = [
            'id', 'name', 'code', 'method_type', 'method_type_display',
            'description', 'version', 'parameters_schema',
            'is_active', 'task_count', 'created_at'
        ]

    def get_task_count(self, obj):
        return obj.tasks.count()


class DetectionTaskSerializer(serializers.ModelSerializer):
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    method_name = serializers.CharField(source='method.name', read_only=True)
    method_code = serializers.CharField(source='method.code', read_only=True)
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    poisoned_record_count = serializers.SerializerMethodField()

    class Meta:
        model = DetectionTask
        fields = [
            'id', 'dataset', 'dataset_name', 'method', 'method_name', 'method_code',
            'name', 'status', 'status_display', 'parameters',
            'total_samples', 'detected_poisoned', 'detection_rate', 'false_positive_rate',
            'result_summary', 'detailed_results', 'error_message', 'traceback',
            'started_at', 'completed_at', 'duration_seconds',
            'created_by', 'created_by_username', 'created_at', 'poisoned_record_count'
        ]
        read_only_fields = ['created_at']

    def get_poisoned_record_count(self, obj):
        return obj.poisoned_records.count()


class PoisonedRecordSerializer(serializers.ModelSerializer):
    severity_display = serializers.CharField(source='get_severity_display', read_only=True)
    verification_status_display = serializers.CharField(source='get_verification_status_display', read_only=True)
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    task_name = serializers.CharField(source='task.name', read_only=True)
    verified_by_username = serializers.CharField(source='verified_by.username', read_only=True)

    class Meta:
        model = PoisonedRecord
        fields = [
            'id', 'task', 'task_name', 'dataset', 'dataset_name',
            'row_index', 'row_hash', 'original_label', 'predicted_label',
            'confidence', 'severity', 'severity_display',
            'trigger_pattern', 'attack_vector', 'poisoning_type',
            'verification_status', 'verification_status_display',
            'verified_by', 'verified_by_username', 'verified_at', 'verification_notes',
            'data_snapshot', 'feature_importance', 'created_at'
        ]


class DatasetUploadSerializer(serializers.Serializer):
    file = serializers.FileField(help_text='上传的数据文件')
    name = serializers.CharField(max_length=200, help_text='数据集名称')
    description = serializers.CharField(required=False, allow_blank=True, help_text='数据集描述')
    domain = serializers.CharField(max_length=100, required=False, allow_blank=True, help_text='应用领域')
    task_type = serializers.CharField(max_length=100, required=False, allow_blank=True, help_text='任务类型')
    tags = serializers.ListField(child=serializers.CharField(), required=False, default=list, help_text='标签')
    notes = serializers.CharField(required=False, allow_blank=True, help_text='上传备注')


class DatasetImportBuiltinSerializer(serializers.Serializer):
    source_path = serializers.CharField(max_length=500, required=True, help_text='内置数据集路径')
    auto_detect_columns = serializers.BooleanField(default=True, help_text='自动检测列类型')
    register_lineage = serializers.BooleanField(default=True, help_text='注册数据谱系')
