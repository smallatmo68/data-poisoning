from django.core.management.base import BaseCommand
from datasets.models import DetectionMethod


class Command(BaseCommand):
    help = '注册系统内置的检测方法'

    def handle(self, *args, **options):
        methods = [
            {
                'name': 'ONION后门检测',
                'code': 'onion',
                'method_type': 'statistical',
                'description': '基于困惑度(PPL)的异常检测方法，源自BackdoorDetection项目ONION算法。通过检测文本中异常的触发器片段、罕见字符/词汇、以及标签内文本长度异常来识别后门投毒样本',
                'version': '1.1.0',
                'parameters_schema': {
                    'ppl_threshold': {'type': 'float', 'default': 2.0, 'min': 0.5, 'max': 10.0, 'description': 'PPL异常分数阈值'},
                    'rare_word_threshold': {'type': 'float', 'default': 0.0005, 'min': 0.00001, 'max': 0.01, 'description': '罕见词频率阈值'},
                    'rare_char_threshold': {'type': 'float', 'default': 0.0001, 'min': 0.00001, 'max': 0.01, 'description': '罕见字符频率阈值'},
                    'sample_size': {'type': 'int', 'default': 5000, 'min': 100, 'max': 100000, 'description': '最大检测样本数'}
                }
            },
            {
                'name': 'STRIP熵检测',
                'code': 'strip',
                'method_type': 'statistical',
                'description': '基于信息熵的异常检测方法，源自BackdoorDetection项目STRIP算法。通过文本词频熵的异常偏离和扰动一致性检测投毒样本，投毒样本通常具有异常低熵或对扰动不敏感的特征',
                'version': '1.1.0',
                'parameters_schema': {
                    'frr': {'type': 'float', 'default': 0.01, 'min': 0.001, 'max': 0.1, 'description': '误拒率(False Rejection Rate)'},
                    'swap_ratio': {'type': 'float', 'default': 0.5, 'min': 0.1, 'max': 0.9, 'description': '扰动替换比例'},
                    'repeat': {'type': 'int', 'default': 3, 'min': 1, 'max': 10, 'description': '扰动重复次数'},
                    'sample_size': {'type': 'int', 'default': 5000, 'min': 100, 'max': 100000, 'description': '最大检测样本数'}
                }
            },
            {
                'name': 'BackdoorDetection后门检测',
                'code': 'backdoor_detect',
                'method_type': 'hybrid',
                'description': '源自BackdoorDetection项目的综合后门检测算法。结合文本扰动敏感度分析和跨标签触发器检测，识别含有后门触发词的投毒样本。核心逻辑：对文本进行mask替换扰动，若扰动后对数似然变化极小则说明文本对扰动不敏感（投毒特征），同时检测跨标签的触发词模式',
                'version': '1.0.0',
                'parameters_schema': {
                    'pct_words_masked': {'type': 'float', 'default': 0.7, 'min': 0.1, 'max': 1.0, 'description': '扰动时mask的词比例'},
                    'span_length': {'type': 'int', 'default': 2, 'min': 1, 'max': 10, 'description': '扰动span长度'},
                    'n_perturbation': {'type': 'int', 'default': 5, 'min': 1, 'max': 20, 'description': '扰动次数'},
                    'sample_size': {'type': 'int', 'default': 3000, 'min': 100, 'max': 100000, 'description': '最大检测样本数'}
                }
            },
            {
                'name': 'Influence-Based错误标签检测',
                'code': 'influence_mislabelled',
                'method_type': 'ml_based',
                'description': '源自Influence-Based-Glitch-Detection项目的影响函数检测算法。基于NIL(Negative Influence Label)信号检测错误标签样本：计算每个样本在各标签词频分布下的对数似然，若样本在非自身标签下的似然更高，则判定为错误标签。模拟CNCI(Conditional Negative Counterfactual Influence)信号的核心逻辑',
                'version': '1.0.0',
                'parameters_schema': {
                    'contamination': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 0.5, 'description': '预期污染率'},
                    'sample_size': {'type': 'int', 'default': 5000, 'min': 100, 'max': 100000, 'description': '最大检测样本数'}
                }
            },
            {
                'name': 'Influence-Based异常检测',
                'code': 'influence_anomaly',
                'method_type': 'ml_based',
                'description': '源自Influence-Based-Glitch-Detection项目的影响函数异常检测算法。基于PCID(Positive Counterfactual Influence Deviation)信号，使用Isolation Forest对多维特征（文本长度、词数、特殊字符比、词汇多样性、信息熵等）进行异常检测，识别与正常数据分布显著偏离的异常样本',
                'version': '1.0.0',
                'parameters_schema': {
                    'contamination': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 0.5, 'description': '预期污染率'},
                    'sample_size': {'type': 'int', 'default': 5000, 'min': 100, 'max': 100000, 'description': '最大检测样本数'}
                }
            },
            {
                'name': 'Defend认证防御检测',
                'code': 'defend_certified',
                'method_type': 'ml_based',
                'description': '源自defend_framework项目的认证防御(Certified Defense)算法。基于Bagging集成投票机制，训练多个决策树子模型，通过投票边际(Vote Margin)分析识别被投毒的样本。若样本在多数子模型中被错误分类或投票边际极低，则判定为投毒样本。模拟DPA(Data Poisoning Attack)认证防御的核心逻辑',
                'version': '1.0.0',
                'parameters_schema': {
                    'confidence': {'type': 'float', 'default': 0.999, 'min': 0.9, 'max': 0.9999, 'description': '认证置信度'},
                    'n_bags': {'type': 'int', 'default': 100, 'min': 10, 'max': 500, 'description': 'Bagging子模型数量'},
                    'sample_size': {'type': 'int', 'default': 5000, 'min': 100, 'max': 100000, 'description': '最大检测样本数'}
                }
            },
            {
                'name': '综合统计检测',
                'code': 'statistical',
                'method_type': 'statistical',
                'description': '基于多种统计指标（标签分布、文本长度Z-score、重复词比例等）的综合投毒检测方法，适用于各类文本分类数据集',
                'version': '1.0.0',
                'parameters_schema': {
                    'threshold': {'type': 'float', 'default': 0.7, 'min': 0.1, 'max': 1.0, 'description': '检测阈值'},
                    'sample_size': {'type': 'int', 'default': 5000, 'min': 100, 'max': 100000, 'description': '最大检测样本数'}
                }
            },
            {
                'name': '综合多方法联合检测',
                'code': 'comprehensive',
                'method_type': 'hybrid',
                'description': '联合使用ONION、STRIP、BackdoorDetection和统计检测四种方法进行投票式检测，被多种方法同时标记的样本将获得更高的严重程度评级。综合检测能有效降低单一方法的误报率，提高检测准确性',
                'version': '1.0.0',
                'parameters_schema': {
                    'sample_size': {'type': 'int', 'default': 3000, 'min': 100, 'max': 100000, 'description': '最大检测样本数'}
                }
            },
        ]

        created_count = 0
        updated_count = 0

        for method_data in methods:
            obj, created = DetectionMethod.objects.update_or_create(
                code=method_data['code'],
                defaults={
                    'name': method_data['name'],
                    'method_type': method_data['method_type'],
                    'description': method_data['description'],
                    'version': method_data['version'],
                    'parameters_schema': method_data['parameters_schema'],
                    'is_active': True,
                }
            )
            if created:
                created_count += 1
                self.stdout.write(self.style.SUCCESS(f'  创建检测方法: {obj.name} ({obj.code})'))
            else:
                updated_count += 1
                self.stdout.write(self.style.WARNING(f'  更新检测方法: {obj.name} ({obj.code})'))

        self.stdout.write(self.style.SUCCESS(
            f'\n检测方法注册完成: 新建 {created_count} 个, 更新 {updated_count} 个'
        ))
