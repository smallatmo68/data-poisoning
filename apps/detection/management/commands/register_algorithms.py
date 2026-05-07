"""
将 8 个检测器的 AlgorithmConfig 初始记录写入数据库。
已存在的记录按 detector_name 做 update_or_create（幂等操作）。
"""

from django.core.management.base import BaseCommand

from apps.detection.models import AlgorithmConfig

ALGORITHMS = [
    {
        'detector_name': 'cleanlab',
        'display_name': 'Cleanlab 标签投毒检测',
        'detector_type': 'label_poison',
        'enabled': True,
        'weight': 0.30,
        'description': '基于 Confident Learning 的标签投毒检测，训练逻辑回归分类器识别低置信度标签样本。无 cleanlab 时自动降级为 self-confidence 方法。',
        'supported_dataset_types': ['csv'],
        'requires_baseline': False,
        'default_params': {
            'label_field': '',
            'max_samples': 5000,
        },
    },
    {
        'detector_name': 'isolation_forest',
        'display_name': '孤立森林异常检测',
        'detector_type': 'anomaly',
        'enabled': True,
        'weight': 0.20,
        'description': '孤立森林算法，对特征空间中离群的样本给出高置信度风险评分，适合高维数据异常检测。',
        'supported_dataset_types': ['csv'],
        'requires_baseline': False,
        'default_params': {
            'label_field': '',
            'max_samples': 5000,
            'contamination': 0.05,
            'n_estimators': 100,
        },
    },
    {
        'detector_name': 'lof',
        'display_name': '局部离群因子检测',
        'detector_type': 'anomaly',
        'enabled': True,
        'weight': 0.15,
        'description': '局部离群因子（LOF）检测，基于邻域密度识别异常样本，对簇结构数据效果好。',
        'supported_dataset_types': ['csv'],
        'requires_baseline': False,
        'default_params': {
            'label_field': '',
            'max_samples': 5000,
            'n_neighbors': 20,
            'contamination': 0.05,
        },
    },
    {
        'detector_name': 'ks_drift',
        'display_name': 'KS 检验分布漂移检测',
        'detector_type': 'distribution',
        'enabled': True,
        'weight': 0.15,
        'description': 'KS 检验分布漂移检测，逐列比较待检测数据集与基准数据集的数值分布差异。需提供基准数据集。',
        'supported_dataset_types': ['csv'],
        'requires_baseline': True,
        'default_params': {
            'alpha': 0.05,
            'label_field': '',
        },
    },
    {
        'detector_name': 'mmd_drift',
        'display_name': 'MMD 分布漂移检测',
        'detector_type': 'distribution',
        'enabled': True,
        'weight': 0.10,
        'description': '最大均值差异（MMD）分布漂移检测，优先使用 alibi-detect，不可用时自动降级为 RBF 核 MMD。需提供基准数据集。',
        'supported_dataset_types': ['csv'],
        'requires_baseline': True,
        'default_params': {
            'alpha': 0.05,
            'n_permutations': 200,
            'label_field': '',
        },
    },
    {
        'detector_name': 'spectral_signature',
        'display_name': '谱签名后门检测',
        'detector_type': 'backdoor',
        'enabled': True,
        'weight': 0.10,
        'description': '谱签名后门检测，对特征矩阵做 SVD 分解，识别在主成分方向上异常显著的样本。',
        'supported_dataset_types': ['csv'],
        'requires_baseline': False,
        'default_params': {
            'label_field': '',
            'max_samples': 3000,
            'n_components': 10,
            'threshold_percentile': 95,
        },
    },
    {
        'detector_name': 'activation_clustering',
        'display_name': '激活聚类后门检测',
        'detector_type': 'backdoor',
        'enabled': False,
        'weight': 0.00,
        'description': '激活聚类后门检测（需深度学习模型），当前为占位实现，功能待完善。',
        'supported_dataset_types': ['csv'],
        'requires_baseline': False,
        'default_params': {
            'label_field': '',
            'max_samples': 3000,
        },
    },
    {
        'detector_name': 'influence',
        'display_name': '影响函数投毒检测',
        'detector_type': 'influence',
        'enabled': False,
        'weight': 0.00,
        'description': '影响函数投毒检测（需深度学习模型），当前为占位实现，功能待完善。',
        'supported_dataset_types': ['csv'],
        'requires_baseline': False,
        'default_params': {
            'label_field': '',
            'max_samples': 3000,
        },
    },
    {
        'detector_name': 'onion',
        'display_name': 'ONION 文本异常检测',
        'detector_type': 'backdoor',
        'enabled': True,
        'weight': 0.10,
        'description': 'ONION 文本异常检测，基于罕见词频率和文本长度异常检测后门触发器。适合含文本列的数据集。',
        'supported_dataset_types': ['csv', 'text'],
        'requires_baseline': False,
        'default_params': {
            'text_field': '',
            'max_samples': 5000,
        },
    },
    {
        'detector_name': 'strip',
        'display_name': 'STRIP 熵值异常检测',
        'detector_type': 'anomaly',
        'enabled': True,
        'weight': 0.10,
        'description': 'STRIP 熵值异常检测，基于特征分布熵值和扰动敏感度识别异常样本。',
        'supported_dataset_types': ['csv'],
        'requires_baseline': False,
        'default_params': {
            'label_field': '',
            'max_samples': 5000,
        },
    },
    {
        'detector_name': 'statistical',
        'display_name': '统计异常检测',
        'detector_type': 'anomaly',
        'enabled': True,
        'weight': 0.10,
        'description': '统计方法异常检测，使用 Z-score 和 IQR 检测数值离群点、罕见标签和文本长度异常。',
        'supported_dataset_types': ['csv'],
        'requires_baseline': False,
        'default_params': {
            'label_field': '',
            'text_field': '',
            'max_samples': 5000,
            'z_threshold': 3.0,
        },
    },
    {
        'detector_name': 'comprehensive',
        'display_name': '综合投票检测',
        'detector_type': 'anomaly',
        'enabled': True,
        'weight': 0.15,
        'description': '综合投票检测器，聚合孤立森林、LOF 和统计检测器的结果，通过多数投票给出最终判定。',
        'supported_dataset_types': ['csv'],
        'requires_baseline': False,
        'default_params': {
            'label_field': '',
            'max_samples': 3000,
        },
    },
]


class Command(BaseCommand):
    help = '注册/更新检测算法配置到数据库（幂等操作）'

    def add_arguments(self, parser):
        parser.add_argument(
            '--reset',
            action='store_true',
            help='删除所有现有配置后重新写入（慎用）',
        )

    def handle(self, *args, **options):
        if options['reset']:
            count, _ = AlgorithmConfig.objects.all().delete()
            self.stdout.write(self.style.WARNING(f'已删除 {count} 条现有配置'))

        created_count = 0
        updated_count = 0

        for algo in ALGORITHMS:
            defaults = {k: v for k, v in algo.items() if k != 'detector_name'}
            _, created = AlgorithmConfig.objects.update_or_create(
                detector_name=algo['detector_name'],
                defaults=defaults,
            )
            if created:
                created_count += 1
                self.stdout.write(f'  [新增] {algo["detector_name"]} ({algo["detector_type"]})')
            else:
                updated_count += 1
                self.stdout.write(f'  [更新] {algo["detector_name"]} ({algo["detector_type"]})')

        self.stdout.write(self.style.SUCCESS(
            f'\n完成：新增 {created_count} 条，更新 {updated_count} 条，共 {len(ALGORITHMS)} 个检测器'
        ))
