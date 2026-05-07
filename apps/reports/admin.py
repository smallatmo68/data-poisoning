from django.contrib import admin
from .models import Report

@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    list_display = ['task', 'report_type', 'created_by', 'created_at']
    readonly_fields = ['file_path']
