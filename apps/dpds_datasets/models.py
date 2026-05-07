import os
import uuid

from django.contrib.auth.models import User
from django.db import models


def dataset_upload_path(instance, filename):
    ext = os.path.splitext(filename)[1]
    return f'uploads/{uuid.uuid4().hex}{ext}'


class Dataset(models.Model):
    """数据集主体记录（支持 CSV、图像 zip、文本 CSV）。"""

    TYPE_CSV = 'csv'
    TYPE_IMAGE = 'image'
    TYPE_TEXT = 'text'
    DATASET_TYPES = [
        (TYPE_CSV, 'CSV 表格'),
        (TYPE_IMAGE, '图像数据集'),
        (TYPE_TEXT, '文本数据集'),
    ]

    STATUS_PENDING = 'pending'
    STATUS_PARSING = 'parsing'
    STATUS_READY = 'ready'
    STATUS_ERROR = 'error'
    STATUS_ARCHIVED = 'archived'
    STATUS_CHOICES = [
        (STATUS_PENDING, '待解析'),
        (STATUS_PARSING, '解析中'),
        (STATUS_READY, '就绪'),
        (STATUS_ERROR, '解析错误'),
        (STATUS_ARCHIVED, '已归档'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset_name = models.CharField('数据集名称', max_length=255)
    dataset_type = models.CharField('数据类型', max_length=20, choices=DATASET_TYPES, default=TYPE_CSV)
    file = models.FileField('文件', upload_to=dataset_upload_path)
    file_md5 = models.CharField('文件 MD5', max_length=32, blank=True)
    file_size = models.BigIntegerField('文件大小 (bytes)', default=0)
    sample_count = models.IntegerField('样本数量', default=0)
    label_field = models.CharField('标签字段', max_length=100, blank=True)
    text_field = models.CharField('文本字段', max_length=100, blank=True)
    storage_path = models.CharField('原始存储路径', max_length=500, blank=True)
    processed_path = models.CharField('预处理后路径', max_length=500, blank=True)
    column_meta = models.JSONField('列元数据', default=dict, blank=True)
    status = models.CharField('状态', max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    error_message = models.TextField('错误信息', blank=True)
    owner = models.ForeignKey(
        User, null=True, blank=True, on_delete=models.SET_NULL, related_name='datasets'
    )
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    updated_at = models.DateTimeField('更新时间', auto_now=True)

    class Meta:
        db_table = 'dpds_dataset'
        verbose_name = '数据集'
        verbose_name_plural = '数据集'
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.dataset_name} ({self.dataset_type})'
