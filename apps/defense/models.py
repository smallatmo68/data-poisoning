import uuid
from django.contrib.auth.models import User
from django.db import models
from apps.detection.models import DetectionTask, DetectionResult
from apps.dpds_datasets.models import Dataset


class CleanResult(models.Model):
    """无害化处理结果，导出净化数据集。"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task = models.ForeignKey(DetectionTask, on_delete=models.CASCADE, related_name='clean_results')
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='clean_results')
    clean_dataset_path = models.CharField('净化数据集路径', max_length=500, blank=True)
    strategy_json = models.JSONField('处理策略', default=dict)
    removed_count = models.IntegerField('删除样本数', default=0)
    relabel_count = models.IntegerField('重新标注样本数', default=0)
    ignored_count = models.IntegerField('忽略样本数', default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'dpds_clean_result'
        verbose_name = '净化结果'
        ordering = ['-created_at']

    def __str__(self):
        return f'净化结果 [{self.task}] 删除={self.removed_count}'


class DefenseSampleAction(models.Model):
    """对单个可疑样本的人工复查操作记录。"""

    ACTION_CHOICES = [
        ('remove', '删除'),
        ('relabel', '重新标注'),
        ('ignore', '忽略'),
        ('confirm_poison', '确认投毒'),
        ('mark_clean', '标记为正常'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    clean_result = models.ForeignKey(
        CleanResult, on_delete=models.CASCADE,
        related_name='sample_actions', null=True, blank=True
    )
    detection_result = models.ForeignKey(
        DetectionResult, on_delete=models.CASCADE,
        related_name='defense_actions'
    )
    action = models.CharField('操作', max_length=20, choices=ACTION_CHOICES)
    original_label = models.CharField('原始标签', max_length=200, blank=True)
    corrected_label = models.CharField('修正标签', max_length=200, blank=True)
    operator = models.ForeignKey(
        User, null=True, blank=True, on_delete=models.SET_NULL,
        related_name='defense_actions'
    )
    reason = models.TextField('操作理由', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'dpds_defense_sample_action'
        verbose_name = '样本复查操作'
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.get_action_display()} - 样本 {self.detection_result.sample_id}'
