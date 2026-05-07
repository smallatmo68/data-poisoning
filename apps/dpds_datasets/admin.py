from django.contrib import admin
from .models import Dataset

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['dataset_name', 'dataset_type', 'status', 'sample_count', 'owner', 'created_at']
    list_filter = ['dataset_type', 'status']
    search_fields = ['dataset_name']
    readonly_fields = ['file_md5', 'file_size', 'sample_count', 'column_meta', 'created_at', 'updated_at']
