import uuid

from django.contrib.auth.models import User
from django.db import models

from apps.dpds_datasets.models import Dataset


class AlgorithmConfig(models.Model):
    """检测器的可配置参数。"""

    DATASET_TYPE_CHOICES = [
        ('csv', 'CSV 表格'),
        ('image', '图像'),
        ('text', '文本'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    detector_name = models.CharField('检测器名称', max_length=100, unique=True)
    display_name = models.CharField('显示名称', max_length=100, blank=True)
    detector_type = models.CharField('检测器类型', max_length=50)
    enabled = models.BooleanField('是否启用', default=True)
    default_params = models.JSONField('默认参数', default=dict)
    weight = models.FloatField('融合权重', default=0.25)
    description = models.TextField('描述', blank=True)
    supported_dataset_types = models.JSONField('支持的数据集类型', default=list, blank=True)
    requires_baseline = models.BooleanField('需要基准数据集', default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'dpds_algorithm_config'
        verbose_name = '算法配置'
        verbose_name_plural = '算法配置'

    def __str__(self):
        return f'{self.detector_name} (weight={self.weight})'


class DetectionTask(models.Model):
    STATUS_QUEUED = 'queued'
    STATUS_RUNNING = 'running'
    STATUS_SUCCESS = 'success'
    STATUS_FAILED = 'failed'
    STATUS_CANCELLED = 'cancelled'

    STATUS_CHOICES = [
        (STATUS_QUEUED, '排队中'),
        (STATUS_RUNNING, '执行中'),
        (STATUS_SUCCESS, '成功'),
        (STATUS_FAILED, '失败'),
        (STATUS_CANCELLED, '已取消'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task_no = models.CharField('任务编号', max_length=32, unique=True, blank=True)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='detection_tasks')
    baseline_dataset = models.ForeignKey(
        Dataset, null=True, blank=True,
        on_delete=models.SET_NULL, related_name='baseline_for_tasks'
    )
    detector_config = models.JSONField('检测器配置', default=dict)
    status = models.CharField('状态', max_length=20, choices=STATUS_CHOICES, default=STATUS_QUEUED)
    progress = models.IntegerField('进度 0-100', default=0)
    risk_score = models.FloatField('综合风险分数', null=True, blank=True)
    error_message = models.TextField('错误信息', blank=True)
    created_by = models.ForeignKey(
        User, null=True, blank=True, on_delete=models.SET_NULL, related_name='dpds_detection_tasks'
    )
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    started_at = models.DateTimeField('开始时间', null=True, blank=True)
    finished_at = models.DateTimeField('结束时间', null=True, blank=True)

    class Meta:
        db_table = 'dpds_detection_task'
        verbose_name = '检测任务'
        verbose_name_plural = '检测任务'
        ordering = ['-created_at']

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if not self.task_no:
            from django.utils import timezone
            now = timezone.now()
            self.task_no = f'T{now.strftime("%Y%m%d")}{str(self.id)[:8].upper()}'
            super().save(update_fields=['task_no'])

    def __str__(self):
        return f'检测任务 {self.task_no} [{self.status}]'


class DetectionResult(models.Model):
    """每个可疑样本的检测结果索引（详情在 MongoDB）。"""

    RISK_TYPES = [
        ('label_poison', '标签投毒'),
        ('backdoor', '后门攻击'),
        ('distribution_shift', '分布偏移'),
        ('anomaly', '异常样本'),
        ('unknown', '未知'),
    ]

    SUGGESTION_CHOICES = [
        ('remove', '建议删除'),
        ('relabel', '建议重新标注'),
        ('ignore', '忽略'),
        ('review', '人工复查'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task = models.ForeignKey(DetectionTask, on_delete=models.CASCADE, related_name='results')
    sample_id = models.CharField('样本标识', max_length=50)
    risk_type = models.CharField('风险类型', max_length=30, choices=RISK_TYPES)
    confidence = models.FloatField('置信度 0-1')
    suggestion = models.CharField('处理建议', max_length=20, choices=SUGGESTION_CHOICES, default='review')
    detector_name = models.CharField('检测器名称', max_length=100)
    detail_doc_id = models.CharField('MongoDB 文档 ID', max_length=50, blank=True)
    triggered_features = models.JSONField('触发特征', default=list, blank=True,
        help_text='触发检测的特征列表，如异常词、异常值等')
    metric_detail = models.JSONField('指标详情', default=dict, blank=True,
        help_text='检测算法输出的中间指标')
    reason = models.TextField('判定原因', blank=True,
        help_text='检测器给出的自然语言解释')
    raw_data_snapshot = models.JSONField('原始数据快照', default=dict, blank=True,
        help_text='该样本在原始数据集中的字段值')
    processed_data_snapshot = models.JSONField('处理后数据快照', default=dict, blank=True,
        help_text='该样本经过预处理后的字段值')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'dpds_detection_result'
        verbose_name = '检测结果'
        ordering = ['-confidence']

    def __str__(self):
        return f'样本 {self.sample_id} | {self.risk_type} | {self.confidence:.2f}'
