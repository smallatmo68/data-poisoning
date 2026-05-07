from django.contrib import admin
from .models import PreprocessResult

@admin.register(PreprocessResult)
class PreprocessResultAdmin(admin.ModelAdmin):
    list_display = ['dataset', 'status', 'created_at']
    list_filter = ['status']
    readonly_fields = ['summary_json', 'output_path', 'created_at']
