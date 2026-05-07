import uuid
from django.contrib.auth.models import User
from django.db import models
from apps.detection.models import DetectionTask


class Report(models.Model):
    REPORT_TYPES = [
        ('html', 'HTML 报告'),
        ('markdown', 'Markdown 报告'),
        ('json', 'JSON 报告'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task = models.ForeignKey(DetectionTask, on_delete=models.CASCADE, related_name='reports')
    title = models.CharField('报告标题', max_length=200, blank=True)
    report_type = models.CharField('报告类型', max_length=20, choices=REPORT_TYPES, default='html')
    file_path = models.CharField('报告文件路径', max_length=500, blank=True)
    summary = models.TextField('报告摘要', blank=True,
        help_text='检测结果的简要总结')
    llm_content = models.TextField('LLM 生成内容', blank=True,
        help_text='DeepSeek 生成的专业分析报告')
    analysis_json = models.JSONField('分析数据', default=dict, blank=True,
        help_text='报告使用的统计数据（风险分布、检测器统计等）')
    created_by = models.ForeignKey(
        User, null=True, blank=True, on_delete=models.SET_NULL, related_name='reports'
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'dpds_report'
        verbose_name = '检测报告'
        ordering = ['-created_at']

    def __str__(self):
        return f'报告 [{self.task}] ({self.report_type})'
