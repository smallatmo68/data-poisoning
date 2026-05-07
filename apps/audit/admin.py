from django.contrib import admin
from .models import AuditLog

@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'action', 'target_type', 'ip_address', 'created_at']
    list_filter = ['action']
    search_fields = ['user__username', 'detail']
    readonly_fields = ['user', 'action', 'target_type', 'target_id', 'detail', 'ip_address', 'created_at']
