import uuid

from django.db import models

from apps.dpds_datasets.models import Dataset


class PreprocessResult(models.Model):
    STATUS_PENDING = 'pending'
    STATUS_RUNNING = 'running'
    STATUS_SUCCESS = 'success'
    STATUS_FAILED = 'failed'

    STATUS_CHOICES = [
        (STATUS_PENDING, '待执行'),
        (STATUS_RUNNING, '执行中'),
        (STATUS_SUCCESS, '成功'),
        (STATUS_FAILED, '失败'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='preprocess_results')
    params_json = models.JSONField('预处理参数', default=dict)
    summary_json = models.JSONField('预处理摘要', default=dict)
    output_path = models.CharField('输出文件路径', max_length=500, blank=True)
    status = models.CharField('状态', max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    error_message = models.TextField('错误信息', blank=True)
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    updated_at = models.DateTimeField('更新时间', auto_now=True)

    class Meta:
        db_table = 'dpds_preprocess_result'
        verbose_name = '预处理结果'
        ordering = ['-created_at']

    def __str__(self):
        return f'预处理 [{self.dataset.dataset_name}] - {self.status}'
