from django.contrib import admin
from .models import CleanResult

@admin.register(CleanResult)
class CleanResultAdmin(admin.ModelAdmin):
    list_display = ['task', 'removed_count', 'relabel_count', 'ignored_count', 'created_at']
    readonly_fields = ['clean_dataset_path', 'strategy_json']
