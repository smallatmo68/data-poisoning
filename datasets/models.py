from django.db import models
from django.contrib.auth.models import User
import os


class DatasetFile(models.Model):
    """数据集物理文件管理"""
    filename = models.CharField(max_length=255, verbose_name='文件名')
    file_path = models.CharField(max_length=500, verbose_name='文件路径')
    file_size = models.BigIntegerField(verbose_name='文件大小(字节)')
    checksum = models.CharField(max_length=64, verbose_name='MD5校验码')
    data_format = models.CharField(max_length=50, default='csv', verbose_name='数据格式')
    row_count = models.IntegerField(default=0, verbose_name='数据行数')
    column_count = models.IntegerField(default=0, verbose_name='数据列数')
    encoding = models.CharField(max_length=20, default='utf-8', verbose_name='文件编码')
    delimiter = models.CharField(max_length=10, default=',', verbose_name='分隔符')
    has_header = models.BooleanField(default=True, verbose_name='是否有表头')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'dataset_file'
        verbose_name = '数据集文件'
        verbose_name_plural = '数据集文件'
        unique_together = ['checksum']

    def __str__(self):
        return self.filename


class DataColumn(models.Model):
    """数据列定义"""
    COLUMN_TYPE_CHOICES = [
        ('text', '文本'),
        ('label', '标签/类别'),
        ('numeric', '数值'),
        ('datetime', '日期时间'),
        ('boolean', '布尔值'),
        ('id', 'ID/索引'),
    ]

    QUALITY_STATUS_CHOICES = [
        ('normal', '正常'),
        ('suspicious', '可疑'),
        ('poisoned', '疑似投毒'),
        ('cleaned', '已清洗'),
    ]

    dataset = models.ForeignKey('Dataset', on_delete=models.CASCADE, related_name='columns', verbose_name='所属数据集')
    name = models.CharField(max_length=100, verbose_name='列名')
    ordinal = models.IntegerField(verbose_name='列顺序')
    column_type = models.CharField(max_length=20, choices=COLUMN_TYPE_CHOICES, default='text', verbose_name='列类型')
    is_target = models.BooleanField(default=False, verbose_name='是否为目标列')
    is_text = models.BooleanField(default=False, verbose_name='是否为文本列')
    null_count = models.IntegerField(default=0, verbose_name='空值数量')
    null_rate = models.FloatField(default=0.0, verbose_name='空值比例')
    unique_count = models.IntegerField(default=0, verbose_name='唯一值数量')
    unique_rate = models.FloatField(default=0.0, verbose_name='唯一值比例')
    avg_length = models.FloatField(default=0.0, verbose_name='平均长度')
    min_value = models.CharField(max_length=255, blank=True, verbose_name='最小值')
    max_value = models.CharField(max_length=255, blank=True, verbose_name='最大值')
    label_distribution = models.JSONField(default=dict, verbose_name='标签分布')
    quality_status = models.CharField(max_length=20, choices=QUALITY_STATUS_CHOICES, default='normal', verbose_name='质量状态')
    description = models.CharField(max_length=500, blank=True, verbose_name='列描述')

    class Meta:
        db_table = 'data_column'
        verbose_name = '数据列'
        verbose_name_plural = '数据列'
        unique_together = ['dataset', 'name']
        ordering = ['ordinal']

    def __str__(self):
        return f"{self.dataset.name}.{self.name}"


class Dataset(models.Model):
    """数据集主表"""
    SOURCE_TYPE_CHOICES = [
        ('builtin', '内置'),
        ('upload', '用户上传'),
        ('derived', '衍生数据'),
    ]

    STATUS_CHOICES = [
        ('uploaded', '已上传'),
        ('parsing', '解析中'),
        ('analyzing', '分析中'),
        ('ready', '就绪'),
        ('archived', '已归档'),
        ('error', '错误'),
    ]

    POISONING_TYPE_CHOICES = [
        ('none', '无'),
        ('backdoor', '后门投毒'),
        ('mislabelled', '错误标签'),
        ('anomaly', '异常值'),
        ('mixed', '混合类型'),
    ]

    name = models.CharField(max_length=200, verbose_name='数据集名称', db_index=True)
    version = models.CharField(max_length=50, default='1.0.0', verbose_name='版本号')
    source_type = models.CharField(max_length=20, choices=SOURCE_TYPE_CHOICES, default='builtin', verbose_name='来源类型')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploaded', verbose_name='状态')
    poisoning_type = models.CharField(max_length=20, choices=POISONING_TYPE_CHOICES, default='none', verbose_name='投毒类型')

    file = models.OneToOneField(DatasetFile, on_delete=models.CASCADE, related_name='dataset', verbose_name='数据文件', null=True, blank=True)

    description = models.TextField(blank=True, verbose_name='描述')
    domain = models.CharField(max_length=100, blank=True, verbose_name='应用领域')
    task_type = models.CharField(max_length=100, blank=True, verbose_name='任务类型')

    total_rows = models.IntegerField(default=0, verbose_name='总行数')
    clean_rows = models.IntegerField(default=0, verbose_name='正常行数')
    poisoned_rows = models.IntegerField(default=0, verbose_name='投毒行数')
    contamination_rate = models.FloatField(default=0.0, verbose_name='污染率')

    tags = models.JSONField(default=list, verbose_name='标签列表')
    metadata = models.JSONField(default=dict, verbose_name='元数据')

    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='uploaded_datasets', verbose_name='上传者')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        db_table = 'dataset'
        verbose_name = '数据集'
        verbose_name_plural = '数据集'
        ordering = ['-created_at']
        unique_together = ['name', 'version']

    def __str__(self):
        return f"{self.name} v{self.version}"

    def save(self, *args, **kwargs):
        if self.pk is None:
            existing = Dataset.objects.filter(name=self.name).first()
            if existing:
                existing.version = self.version
                existing.file = self.file
                existing.status = self.status
                existing.updated_at = models.functions.Now()
                for field in ['description', 'domain', 'task_type', 'total_rows', 'clean_rows', 'poisoned_rows', 'contamination_rate', 'tags', 'metadata']:
                    setattr(existing, field, getattr(self, field))
                existing.save()
                self.pk = existing.pk
                return
        super().save(*args, **kwargs)


class UploadRecord(models.Model):
    """用户上传记录"""
    SOURCE_CHOICES = [
        ('web', '网页上传'),
        ('api', 'API上传'),
        ('cli', '命令行上传'),
        ('sync', '同步导入'),
    ]

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='upload_records', verbose_name='数据集')
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default='web', verbose_name='上传来源')
    uploader = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='upload_records', verbose_name='上传者')
    original_filename = models.CharField(max_length=255, verbose_name='原始文件名')
    file_size = models.BigIntegerField(verbose_name='文件大小')
    ip_address = models.GenericIPAddressField(null=True, blank=True, verbose_name='IP地址')
    user_agent = models.CharField(max_length=500, blank=True, verbose_name='浏览器信息')
    notes = models.TextField(blank=True, verbose_name='备注')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='上传时间')

    class Meta:
        db_table = 'upload_record'
        verbose_name = '上传记录'
        verbose_name_plural = '上传记录'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.dataset.name} - {self.get_source_display()}"


class DataLineage(models.Model):
    """数据谱系追踪表"""
    OPERATION_TYPE_CHOICES = [
        ('import', '数据导入'),
        ('filter', '数据过滤'),
        ('transform', '数据转换'),
        ('merge', '数据合并'),
        ('split', '数据分割'),
        ('clean', '数据清洗'),
        ('augment', '数据增强'),
        ('detection', '投毒检测'),
        ('repair', '数据修复'),
    ]

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='lineage_records', verbose_name='数据集')
    operation_type = models.CharField(max_length=20, choices=OPERATION_TYPE_CHOICES, verbose_name='操作类型')
    operation_name = models.CharField(max_length=200, verbose_name='操作名称')
    description = models.TextField(blank=True, verbose_name='操作描述')

    input_datasets = models.JSONField(default=list, verbose_name='输入数据集')
    output_datasets = models.JSONField(default=list, verbose_name='输出数据集')

    parent_lineage = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='child_lineages', verbose_name='父谱系')
    root_lineage = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='descendant_lineages', verbose_name='根谱系')

    operator = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='lineage_operations', verbose_name='操作者')
    parameters = models.JSONField(default=dict, verbose_name='操作参数')
    result_summary = models.JSONField(default=dict, verbose_name='结果摘要')

    rows_affected = models.IntegerField(default=0, verbose_name='影响行数')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='操作时间')

    class Meta:
        db_table = 'data_lineage'
        verbose_name = '数据谱系'
        verbose_name_plural = '数据谱系'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.dataset.name} - {self.operation_name}"


class DetectionMethod(models.Model):
    """检测方法注册"""
    METHOD_TYPE_CHOICES = [
        ('statistical', '统计检测'),
        ('ml_based', '机器学习检测'),
        ('rule_based', '规则检测'),
        ('hybrid', '混合检测'),
    ]

    name = models.CharField(max_length=100, unique=True, verbose_name='方法名称')
    code = models.CharField(max_length=50, unique=True, verbose_name='方法代码')
    method_type = models.CharField(max_length=20, choices=METHOD_TYPE_CHOICES, verbose_name='方法类型')
    description = models.TextField(verbose_name='方法描述')
    version = models.CharField(max_length=20, default='1.0.0', verbose_name='版本')
    parameters_schema = models.JSONField(default=dict, verbose_name='参数模式')
    is_active = models.BooleanField(default=True, verbose_name='是否启用')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        db_table = 'detection_method'
        verbose_name = '检测方法'
        verbose_name_plural = '检测方法'

    def __str__(self):
        return f"{self.name} ({self.code})"


class DetectionTask(models.Model):
    """检测任务"""
    STATUS_CHOICES = [
        ('queued', '排队中'),
        ('running', '运行中'),
        ('completed', '已完成'),
        ('failed', '失败'),
        ('cancelled', '已取消'),
    ]

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='detection_tasks', verbose_name='检测数据集')
    method = models.ForeignKey(DetectionMethod, on_delete=models.CASCADE, related_name='tasks', verbose_name='检测方法')
    name = models.CharField(max_length=200, verbose_name='任务名称')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued', verbose_name='状态')

    parameters = models.JSONField(default=dict, verbose_name='检测参数')

    total_samples = models.IntegerField(default=0, verbose_name='检测样本数')
    detected_poisoned = models.IntegerField(default=0, verbose_name='检测出投毒数')
    detection_rate = models.FloatField(default=0.0, verbose_name='检测率')
    false_positive_rate = models.FloatField(default=0.0, verbose_name='误报率')

    result_summary = models.JSONField(default=dict, verbose_name='结果摘要')
    detailed_results = models.JSONField(default=list, verbose_name='详细结果')

    error_message = models.TextField(blank=True, verbose_name='错误信息')
    traceback = models.TextField(blank=True, verbose_name='错误追踪')

    started_at = models.DateTimeField(null=True, blank=True, verbose_name='开始时间')
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name='完成时间')
    duration_seconds = models.IntegerField(default=0, verbose_name='持续时间(秒)')

    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='detection_tasks', verbose_name='创建者')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        db_table = 'detection_task'
        verbose_name = '检测任务'
        verbose_name_plural = '检测任务'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} [{self.get_status_display()}]"


class PoisonedRecord(models.Model):
    """投毒记录详情"""
    SEVERITY_CHOICES = [
        ('low', '低'),
        ('medium', '中'),
        ('high', '高'),
        ('critical', '严重'),
    ]

    VERIFICATION_STATUS_CHOICES = [
        ('detected', '已检测'),
        ('verified', '已验证'),
        ('false_positive', '误报'),
        ('confirmed', '已确认'),
        ('repaired', '已修复'),
        ('ignored', '已忽略'),
    ]

    task = models.ForeignKey(DetectionTask, on_delete=models.CASCADE, related_name='poisoned_records', verbose_name='所属任务')
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='poisoned_records', verbose_name='数据集')
    row_index = models.IntegerField(verbose_name='行索引')
    row_hash = models.CharField(max_length=64, verbose_name='行哈希')

    original_label = models.CharField(max_length=100, verbose_name='原始标签')
    predicted_label = models.CharField(max_length=100, blank=True, verbose_name='预测标签')
    confidence = models.FloatField(default=0.0, verbose_name='置信度')
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES, default='medium', verbose_name='严重程度')

    trigger_pattern = models.CharField(max_length=500, blank=True, verbose_name='触发模式')
    attack_vector = models.CharField(max_length=200, blank=True, verbose_name='攻击向量')
    poisoning_type = models.CharField(max_length=50, blank=True, verbose_name='投毒类型')

    verification_status = models.CharField(max_length=20, choices=VERIFICATION_STATUS_CHOICES, default='detected', verbose_name='验证状态')
    verified_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='verified_records', verbose_name='验证者')
    verified_at = models.DateTimeField(null=True, blank=True, verbose_name='验证时间')
    verification_notes = models.TextField(blank=True, verbose_name='验证备注')

    data_snapshot = models.JSONField(default=dict, verbose_name='数据快照')
    feature_importance = models.JSONField(default=dict, verbose_name='特征重要性')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='检测时间')

    class Meta:
        db_table = 'poisoned_record'
        verbose_name = '投毒记录'
        verbose_name_plural = '投毒记录'
        unique_together = ['task', 'row_index']
        ordering = ['-confidence', '-severity']

    def __str__(self):
        return f"{self.dataset.name}[{self.row_index}]: {self.original_label}->{self.predicted_label}"
