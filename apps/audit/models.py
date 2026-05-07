from django.db import models


class AuditLog(models.Model):
    """记录用户关键操作，用于安全审计。"""

    ACTION_CHOICES = [
        ('login', '登录'),
        ('logout', '退出'),
        ('upload', '上传数据集'),
        ('delete', '删除'),
        ('detect', '创建检测任务'),
        ('defense', '执行无害化'),
        ('export', '导出/下载'),
        ('config', '修改配置'),
        ('other', '其他'),
    ]

    user = models.ForeignKey(
        'auth.User', null=True, blank=True,
        on_delete=models.SET_NULL, related_name='audit_logs'
    )
    action = models.CharField('操作类型', max_length=30, choices=ACTION_CHOICES)
    target_type = models.CharField('目标类型', max_length=50, blank=True)
    target_id = models.CharField('目标 ID', max_length=50, blank=True)
    detail = models.TextField('操作详情', blank=True)
    ip_address = models.GenericIPAddressField('IP 地址', null=True, blank=True)
    created_at = models.DateTimeField('操作时间', auto_now_add=True)

    class Meta:
        db_table = 'dpds_audit_log'
        verbose_name = '审计日志'
        verbose_name_plural = '审计日志'
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user} | {self.action} | {self.created_at:%Y-%m-%d %H:%M:%S}'
