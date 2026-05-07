from rest_framework import serializers
from .models import Dataset


class DatasetSerializer(serializers.ModelSerializer):
    owner_name = serializers.CharField(source='owner.username', read_only=True, default='')

    class Meta:
        model = Dataset
        fields = [
            'id', 'dataset_name', 'dataset_type', 'file_size', 'sample_count',
            'label_field', 'text_field', 'column_meta', 'status', 'error_message',
            'owner_name', 'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'file_size', 'sample_count', 'column_meta', 'status', 'created_at', 'updated_at']


class DatasetUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    dataset_name = serializers.CharField(max_length=255, required=False, default='')
    dataset_type = serializers.ChoiceField(choices=['csv', 'image', 'text'], default='csv')
    label_field = serializers.CharField(max_length=100, required=False, default='')
    text_field = serializers.CharField(max_length=100, required=False, default='')
