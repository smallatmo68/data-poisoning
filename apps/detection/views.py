import logging
import math
import threading
from collections import defaultdict

from django.db import close_old_connections
from rest_framework import status
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.dpds_datasets.models import Dataset
from algorithm_engine.registry import list_detectors
from .models import AlgorithmConfig, DetectionTask, DetectionResult
from .services import get_task_progress_from_redis, run_detection

logger = logging.getLogger('dpds.detection')


def _run_detection_thread(task_id: str):
    """在独立线程中运行检测，确保 Django DB 连接独立管理。"""
    close_old_connections()
    logger.info('[thread] 开始执行检测任务 %s', task_id)
    try:
        run_detection(task_id)
        logger.info('[thread] 检测任务 %s 完成', task_id)
    except Exception as e:
        logger.exception('[thread] 检测任务 %s 异常: %s', task_id, e)
    finally:
        close_old_connections()


def ok(data=None, msg='OK'):
    return Response({'code': 0, 'msg': msg, 'data': data})


def err(msg, code=4001, http_status=status.HTTP_400_BAD_REQUEST):
    return Response({'code': code, 'msg': msg, 'data': None}, status=http_status)


class CreateDetectionTaskView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        dataset_id = request.data.get('dataset_id')
        detector_names = request.data.get('detectors', ['cleanlab', 'isolation_forest'])
        extra_config = request.data.get('config', {})
        baseline_id = request.data.get('baseline_dataset_id')

        try:
            dataset = Dataset.objects.get(id=dataset_id)
        except Dataset.DoesNotExist:
            return err('数据集不存在')

        if dataset.status != Dataset.STATUS_READY:
            return err(f'数据集状态为 {dataset.status}，尚未就绪')

        baseline = None
        if baseline_id:
            try:
                baseline = Dataset.objects.get(id=baseline_id)
            except Dataset.DoesNotExist:
                return err('基准数据集不存在')

        config = {'detectors': detector_names, **extra_config}
        task = DetectionTask.objects.create(
            dataset=dataset,
            baseline_dataset=baseline,
            detector_config=config,
            created_by=request.user,
        )

        # 在独立线程中执行检测（确保不受请求生命周期影响）
        t = threading.Thread(target=_run_detection_thread, args=(str(task.id),), daemon=True)
        t.start()
        logger.info('已启动检测线程，任务 %s', task.id)

        return ok({'task_id': str(task.id), 'task_no': task.task_no}, msg='检测任务已创建')

    def get(self, request):
        tasks = DetectionTask.objects.filter(created_by=request.user).order_by('-created_at')[:50]
        data = [{
            'task_id': str(t.id),
            'task_no': t.task_no,
            'dataset': str(t.dataset_id),
            'status': t.status,
            'progress': t.progress,
            'risk_score': t.risk_score,
            'created_at': t.created_at.isoformat(),
        } for t in tasks]
        return ok(data)


class DetectionTaskDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, task_id):
        try:
            task = DetectionTask.objects.select_related('dataset').get(id=task_id)
        except DetectionTask.DoesNotExist:
            return err('任务不存在', http_status=status.HTTP_404_NOT_FOUND)

        # 检查失败检测器
        failed_detectors = []
        if task.status == DetectionTask.STATUS_SUCCESS:
            config = task.detector_config or {}
            requested = config.get('detectors', [])
            succeeded = set(
                DetectionResult.objects.filter(task=task)
                .values_list('detector_name', flat=True)
                .distinct()
            )
            for det_name in requested:
                if det_name not in succeeded:
                    from apps.detection.models import AlgorithmConfig as AC
                    ac = AC.objects.filter(detector_name=det_name).first()
                    failed_detectors.append({
                        'detector_name': det_name,
                        'display_name': ac.display_name if ac else det_name,
                        'status': 'failed',
                        'error_message': '检测器执行失败或无结果返回',
                    })

        return ok({
            'task_id': str(task.id),
            'task_no': task.task_no,
            'dataset': {'id': str(task.dataset_id), 'name': task.dataset.dataset_name},
            'status': task.status,
            'progress': task.progress,
            'risk_score': task.risk_score,
            'detector_config': task.detector_config,
            'error_message': task.error_message,
            'failed_detectors': failed_detectors,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'finished_at': task.finished_at.isoformat() if task.finished_at else None,
        })


class DetectionTaskProgressView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, task_id):
        try:
            task = DetectionTask.objects.get(id=task_id)
        except DetectionTask.DoesNotExist:
            return err('任务不存在', http_status=status.HTTP_404_NOT_FOUND)

        redis_info = get_task_progress_from_redis(str(task_id))
        return ok({
            'task_id': str(task.id),
            'status': task.status,
            'progress': redis_info.get('progress', task.progress),
            'message': redis_info.get('message', ''),
        })


class DetectionResultsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, task_id):
        try:
            task = DetectionTask.objects.get(id=task_id)
        except DetectionTask.DoesNotExist:
            return err('任务不存在', http_status=status.HTTP_404_NOT_FOUND)

        if task.status != DetectionTask.STATUS_SUCCESS:
            return ok({'status': task.status, 'results': [], 'msg': '任务尚未完成'})

        results = DetectionResult.objects.filter(task=task).order_by('-confidence')[:500]
        data = [{
            'id': str(r.id),
            'sample_id': r.sample_id,
            'risk_type': r.risk_type,
            'confidence': r.confidence,
            'suggestion': r.suggestion,
            'detector_name': r.detector_name,
            'reason': r.reason,
            'triggered_features': r.triggered_features,
        } for r in results]

        return ok({
            'task_id': str(task.id),
            'risk_score': task.risk_score,
            'total_suspicious': len(data),
            'results': data,
        })


class SampleDetailView(APIView):
    """获取单个检测结果的完整详情（含数据快照）。"""
    permission_classes = [IsAuthenticated]

    def get(self, request, result_id):
        try:
            r = DetectionResult.objects.select_related('task__dataset').get(id=result_id)
        except DetectionResult.DoesNotExist:
            return err('检测结果不存在', http_status=status.HTTP_404_NOT_FOUND)

        return ok({
            'id': str(r.id),
            'sample_id': r.sample_id,
            'risk_type': r.risk_type,
            'confidence': r.confidence,
            'suggestion': r.suggestion,
            'detector_name': r.detector_name,
            'reason': r.reason,
            'triggered_features': r.triggered_features,
            'metric_detail': r.metric_detail,
            'raw_data_snapshot': r.raw_data_snapshot,
            'processed_data_snapshot': r.processed_data_snapshot,
            'task_no': r.task.task_no if r.task else '',
            'dataset_name': r.task.dataset.dataset_name if r.task and r.task.dataset else '',
            'created_at': r.created_at.isoformat() if r.created_at else None,
        })


class AlgorithmConfigView(APIView):
    def get_permissions(self):
        if self.request.method == 'GET':
            return [IsAuthenticated()]
        return [IsAdminUser()]

    def get(self, request):
        configs = AlgorithmConfig.objects.all()
        data = [{
            'detector_name': c.detector_name,
            'display_name': c.display_name or c.detector_name,
            'detector_type': c.detector_type,
            'enabled': c.enabled,
            'weight': c.weight,
            'description': c.description or '',
            'requires_baseline': c.requires_baseline,
            'default_params': c.default_params,
        } for c in configs]
        return ok(data)

    def put(self, request):
        updates = request.data if isinstance(request.data, list) else [request.data]
        for item in updates:
            name = item.get('detector_name')
            if not name:
                continue
            AlgorithmConfig.objects.update_or_create(
                detector_name=name,
                defaults={
                    'detector_type': item.get('detector_type', 'unknown'),
                    'enabled': item.get('enabled', True),
                    'weight': item.get('weight', 0.25),
                    'default_params': item.get('default_params', {}),
                },
            )
        return ok(msg='算法配置已更新')


class DetectorsView(APIView):
    """返回所有可用检测器的元数据，供前端算法选择使用。"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # 从数据库读取配置
        configs = {
            c.detector_name: c
            for c in AlgorithmConfig.objects.all()
        }
        # 从注册中心获取所有已知检测器名称
        all_names = list_detectors()

        data = []
        for name in all_names:
            cfg = configs.get(name)
            if cfg:
                data.append({
                    'detector_name': cfg.detector_name,
                    'display_name': cfg.display_name or cfg.detector_name,
                    'detector_type': cfg.detector_type,
                    'enabled': cfg.enabled,
                    'weight': cfg.weight,
                    'description': cfg.description,
                    'supported_dataset_types': cfg.supported_dataset_types or ['csv'],
                    'requires_baseline': cfg.requires_baseline,
                    'default_params': cfg.default_params,
                })
            else:
                data.append({
                    'detector_name': name,
                    'display_name': name,
                    'detector_type': 'unknown',
                    'enabled': True,
                    'weight': 0.25,
                    'description': '',
                    'supported_dataset_types': ['csv'],
                    'requires_baseline': False,
                    'default_params': {},
                })

        return ok(data)


# ── 检测器方法说明模板 ──────────────────────────────────────────────────────
METHOD_EXPLANATIONS = {
    'cleanlab': '通过训练逻辑回归分类器，对比每个样本的真实标签与模型预测标签的置信度差异。置信度越低，说明模型对该样本的标签判断越不确定，可能被错误标注。',
    'isolation_forest': '孤立森林算法通过随机分割特征空间来"孤立"每个样本。需要分割次数越少的样本越容易被孤立，说明它在特征空间中越偏离主流分布，属于异常点。',
    'lof': '局部离群因子（LOF）通过比较每个样本与其邻域样本的密度来判断异常。如果一个样本的邻域密度远低于其邻居，说明它可能是异常样本。',
    'ks_drift': 'KS 检验逐列比较待检测数据集与基准数据集的数值分布。KS 统计量越大、p 值越小，说明该列的分布差异越显著，存在数据漂移。',
    'mmd_drift': '最大均值差异（MMD）在高维特征空间中比较两个数据集的整体分布差异。通过核函数映射后计算均值差异，能够捕捉非线性的分布变化。',
    'spectral_signature': '对特征矩阵做 SVD 分解，分析各样本在主成分方向上的投影。在主成分上投影值异常大的样本，可能是被注入的后门触发样本。',
    'activation_clustering': '激活聚类方法对模型中间层的激活值进行聚类分析。后门样本在激活空间中会形成独立的小簇，与正常样本的激活模式不同。',
    'influence': '影响函数方法计算每个训练样本对模型损失的影响。投毒样本通常对模型在测试集上的损失有异常大的负面影响。',
    'onion': 'ONION 方法通过检测文本中的罕见词和异常长度来识别可能被注入的后门触发器。',
    'strip': 'STRIP 方法通过对输入进行扰动并观察模型输出的变化来判断样本是否为后门样本。后门样本对扰动不敏感。',
    'statistical': '统计方法通过检测罕见标签、异常长度、重复词等统计特征来识别可疑样本。',
    'comprehensive': '综合投票方法聚合多个检测器的结果，通过多数投票机制给出最终判定。',
}

RISK_TYPE_LABELS = {
    'label_poison': '标签投毒',
    'backdoor': '后门攻击',
    'distribution_shift': '分布偏移',
    'anomaly': '异常样本',
    'unknown': '未知风险',
}

SUGGESTION_LABELS = {
    'remove': '建议删除',
    'relabel': '建议重标',
    'review': '人工复查',
    'ignore': '忽略',
}


class TaskAnalysisView(APIView):
    """返回检测任务的结构化分析数据，用于前端可解释性报告展示。"""
    permission_classes = [IsAuthenticated]

    def get(self, request, task_id):
        try:
            task = DetectionTask.objects.select_related('dataset').get(id=task_id)
        except DetectionTask.DoesNotExist:
            return err('任务不存在', http_status=status.HTTP_404_NOT_FOUND)

        if task.status != DetectionTask.STATUS_SUCCESS:
            return err('任务尚未完成，无法生成分析')

        results = list(DetectionResult.objects.filter(task=task))
        total_suspicious = len(results)

        if total_suspicious == 0:
            return ok({
                'risk_decomposition': [],
                'per_detector_analysis': [],
                'label_analysis': {},
                'confidence_stats': {},
                'suggestion_breakdown': {},
                'cross_detector_overlap': [],
                'conclusion': {
                    'risk_level': 'low',
                    'risk_score': task.risk_score or 0,
                    'summary': '未检测到可疑样本，数据集安全性良好。',
                    'recommendations': ['数据集可安全用于模型训练'],
                },
            })

        total_samples = task.dataset.sample_count or 1

        # ── 1. 读取算法配置（权重 + 显示名）────────────────────────
        config_map = {}
        for ac in AlgorithmConfig.objects.all():
            config_map[ac.detector_name] = ac

        # ── 2. 按检测器分组统计 ─────────────────────────────────────
        det_groups = defaultdict(list)
        for r in results:
            det_groups[r.detector_name].append(r)

        # ── 3. 风险分解（按 detector_type 聚合）────────────────────
        type_contribution = defaultdict(lambda: {'count': 0, 'weight': 0, 'max_ratio': 0})
        for det_name, items in det_groups.items():
            cfg = config_map.get(det_name)
            det_type = cfg.detector_type if cfg else 'unknown'
            weight = cfg.weight if cfg else 0.25
            ratio = len(items) / total_samples
            type_contribution[det_type]['count'] += len(items)
            type_contribution[det_type]['weight'] = max(type_contribution[det_type]['weight'], weight)
            type_contribution[det_type]['max_ratio'] = max(type_contribution[det_type]['max_ratio'], ratio)

        risk_score = task.risk_score or 0
        risk_decomposition = []
        for det_type, info in type_contribution.items():
            contribution = info['weight'] * info['max_ratio']
            risk_decomposition.append({
                'detector_type': det_type,
                'detector_type_label': RISK_TYPE_LABELS.get(det_type, det_type),
                'weight': round(info['weight'], 2),
                'ratio': round(info['max_ratio'], 4),
                'contribution': round(contribution, 4),
                'suspicious_count': info['count'],
            })
        risk_decomposition.sort(key=lambda x: x['contribution'], reverse=True)

        # ── 4. 各检测器详细分析 ─────────────────────────────────────
        per_detector_analysis = []
        for det_name, items in det_groups.items():
            cfg = config_map.get(det_name)
            confidences = [r.confidence for r in items]
            det_type = cfg.detector_type if cfg else 'unknown'
            display_name = (cfg.display_name or det_name) if cfg else det_name

            # 按 risk_type 分组
            rt_groups = defaultdict(int)
            for r in items:
                rt_groups[r.risk_type] += 1

            # 按 suggestion 分组
            sug_groups = defaultdict(int)
            for r in items:
                sug_groups[r.suggestion] += 1

            # 生成发现描述
            top_type = max(rt_groups.items(), key=lambda x: x[1])
            findings = (
                f"在 {total_samples} 个样本中发现 {len(items)} 个可疑样本"
                f"（占比 {len(items)/total_samples*100:.1f}%），"
                f"平均置信度 {sum(confidences)/len(confidences)*100:.1f}%，"
                f"最高 {max(confidences)*100:.1f}%。"
                f"主要风险类型为「{RISK_TYPE_LABELS.get(top_type[0], top_type[0])}」"
                f"（{top_type[1]} 条）。"
            )

            # 检测器特有统计
            detail_stats = {}
            if det_type == 'label_poison':
                detail_stats['risk_type_distribution'] = dict(rt_groups)
            elif det_type == 'distribution':
                drifted_cols = [r.sample_id.replace('COL-', '') for r in items]
                detail_stats['drifted_columns'] = drifted_cols
                detail_stats['drifted_count'] = len(drifted_cols)
            elif det_type == 'anomaly':
                detail_stats['avg_confidence'] = round(sum(confidences) / len(confidences), 4)
                detail_stats['max_confidence'] = round(max(confidences), 4)
            elif det_type == 'backdoor':
                detail_stats['suspicious_count'] = len(items)

            per_detector_analysis.append({
                'detector_name': det_name,
                'display_name': display_name,
                'detector_type': det_type,
                'detector_type_label': RISK_TYPE_LABELS.get(det_type, det_type),
                'status': 'success',
                'suspicious_count': len(items),
                'total_samples': total_samples,
                'ratio': round(len(items) / total_samples, 4),
                'avg_confidence': round(sum(confidences) / len(confidences), 4),
                'max_confidence': round(max(confidences), 4),
                'min_confidence': round(min(confidences), 4),
                'method_explanation': METHOD_EXPLANATIONS.get(det_name, '该检测器通过特定算法识别可疑样本。'),
                'findings': findings,
                'risk_type_distribution': {RISK_TYPE_LABELS.get(k, k): v for k, v in rt_groups.items()},
                'suggestion_distribution': {SUGGESTION_LABELS.get(k, k): v for k, v in sug_groups.items()},
                'detail_stats': detail_stats,
            })
        per_detector_analysis.sort(key=lambda x: x['suspicious_count'], reverse=True)

        # ── 5. 标签分布分析 ─────────────────────────────────────────
        label_field = task.dataset.label_field
        label_analysis = {}
        if label_field:
            rt_all = defaultdict(int)
            for r in results:
                rt_all[r.risk_type] += 1
            label_analysis = {
                'label_field': label_field,
                'risk_type_distribution': {RISK_TYPE_LABELS.get(k, k): v for k, v in rt_all.items()},
            }

        # ── 6. 置信度统计 ───────────────────────────────────────────
        all_conf = sorted([r.confidence for r in results])
        n = len(all_conf)
        conf_mean = sum(all_conf) / n
        conf_median = all_conf[n // 2] if n % 2 else (all_conf[n//2 - 1] + all_conf[n//2]) / 2
        conf_std = math.sqrt(sum((c - conf_mean) ** 2 for c in all_conf) / n) if n > 1 else 0

        def percentile(data, p):
            k = (len(data) - 1) * p / 100
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data[int(k)]
            return data[f] * (c - k) + data[c] * (k - f)

        confidence_stats = {
            'count': n,
            'mean': round(conf_mean, 4),
            'median': round(conf_median, 4),
            'std': round(conf_std, 4),
            'min': round(all_conf[0], 4),
            'max': round(all_conf[-1], 4),
            'p25': round(percentile(all_conf, 25), 4),
            'p75': round(percentile(all_conf, 75), 4),
            'p90': round(percentile(all_conf, 90), 4),
            'p99': round(percentile(all_conf, 99), 4),
            'high_conf_count': sum(1 for c in all_conf if c >= 0.8),
            'medium_conf_count': sum(1 for c in all_conf if 0.5 <= c < 0.8),
            'low_conf_count': sum(1 for c in all_conf if c < 0.5),
        }

        # ── 7. 处理建议分布 ─────────────────────────────────────────
        sug_counts = defaultdict(int)
        for r in results:
            sug_counts[r.suggestion] += 1
        suggestion_breakdown = {}
        for sug, count in sug_counts.items():
            suggestion_breakdown[SUGGESTION_LABELS.get(sug, sug)] = {
                'suggestion': sug,
                'count': count,
                'pct': round(count / total_suspicious * 100, 1),
            }

        # ── 8. 交叉检测（被多个检测器标记的样本）──────────────────
        sample_detectors = defaultdict(set)
        for r in results:
            sample_detectors[r.sample_id].add(r.detector_name)
        cross_detector_overlap = []
        for sid, dets in sample_detectors.items():
            if len(dets) > 1:
                cross_detector_overlap.append({
                    'sample_id': sid,
                    'detectors': sorted(dets),
                    'count': len(dets),
                })
        cross_detector_overlap.sort(key=lambda x: x['count'], reverse=True)

        # ── 9. 综合结论 ─────────────────────────────────────────────
        risk_level = 'high' if risk_score >= 0.15 else 'medium' if risk_score >= 0.05 else 'low'
        risk_level_text = {'high': '高风险', 'medium': '中风险', 'low': '低风险'}[risk_level]
        pct = round(total_suspicious / total_samples * 100, 1)

        recommendations = []
        if confidence_stats['high_conf_count'] > 0:
            recommendations.append(f"建议对 {confidence_stats['high_conf_count']} 个高置信度（≥80%）可疑样本执行删除操作")
        if sug_counts.get('relabel', 0) > 0:
            recommendations.append(f"建议对 {sug_counts['relabel']} 个标签投毒样本进行人工复核并重新标注")
        if sug_counts.get('review', 0) > 0:
            recommendations.append(f"建议对 {sug_counts['review']} 个待复查样本进行人工审查确认")
        if len(cross_detector_overlap) > 0:
            recommendations.append(f"有 {len(cross_detector_overlap)} 个样本被多个检测器同时标记，应优先处理")
        if not recommendations:
            recommendations.append('数据集整体风险较低，可安全用于模型训练')

        if risk_level == 'high':
            summary = f'该数据集存在高度数据投毒风险（综合风险分数 {risk_score*100:.1f}%），共发现 {total_suspicious} 个可疑样本（约占 {pct}%）。强烈建议在训练前对数据集进行全面清洗。'
        elif risk_level == 'medium':
            summary = f'该数据集存在中等数据投毒风险（综合风险分数 {risk_score*100:.1f}%），共发现 {total_suspicious} 个可疑样本（约占 {pct}%）。建议针对高置信度样本进行人工复核。'
        else:
            summary = f'该数据集安全风险较低（综合风险分数 {risk_score*100:.1f}%），发现 {total_suspicious} 个潜在可疑样本（约占 {pct}%）。整体数据质量可信。'

        # ── 10. 图表数据 ─────────────────────────────────────────────
        # 各检测器对比数据
        detector_comparison = []
        for d in per_detector_analysis:
            detector_comparison.append({
                'name': d['display_name'],
                'detector_name': d['detector_name'],
                'risk_score': round(d['ratio'], 4),
                'suspicious_count': d['suspicious_count'],
                'avg_confidence': d['avg_confidence'],
            })

        # 风险类型分布
        risk_type_dist = []
        rt_all = defaultdict(int)
        for r in results:
            rt_all[r.risk_type] += 1
        for rt, count in rt_all.items():
            risk_type_dist.append({
                'name': RISK_TYPE_LABELS.get(rt, rt),
                'value': count,
            })

        # 置信度分布直方图
        conf_histogram = []
        bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for lo, hi in bins:
            count = sum(1 for c in all_conf if lo <= c < hi)
            conf_histogram.append({
                'range': f'{lo:.1f}-{hi:.1f}',
                'count': count,
            })
        # 补充 1.0 的情况
        conf_histogram[-1]['count'] += sum(1 for c in all_conf if c >= 1.0)

        # 处理建议分布
        suggestion_dist = []
        for sug_key in ['remove', 'relabel', 'review', 'ignore']:
            count = sug_counts.get(sug_key, 0)
            if count > 0:
                suggestion_dist.append({
                    'name': SUGGESTION_LABELS.get(sug_key, sug_key),
                    'value': count,
                })

        # 读取失败检测器信息
        failed_detectors = []
        meta = {}
        try:
            from algorithm_engine.mongo_client import get_detection_detail
            if task.detail_doc_id:
                meta = get_detection_detail(task.detail_doc_id) or {}
        except Exception:
            pass
        # 从 task.detector_config 中获取请求的检测器列表
        requested_detectors = (task.detector_config or {}).get('detectors', [])
        succeeded_detectors = [d['detector_name'] for d in per_detector_analysis]
        for det_name in requested_detectors:
            if det_name not in succeeded_detectors:
                from algorithm_engine.registry import get_detector as get_det
                from apps.detection.models import AlgorithmConfig as AC
                ac = AC.objects.filter(detector_name=det_name).first()
                failed_detectors.append({
                    'detector_name': det_name,
                    'display_name': ac.display_name if ac else det_name,
                    'detector_type': ac.detector_type if ac else 'unknown',
                    'status': 'failed',
                    'error_message': '检测器执行失败或未返回结果',
                    'suggestion': '请检查数据集是否包含该检测器所需的特征类型',
                })

        chart_data = {
            'detector_comparison': detector_comparison,
            'risk_type_distribution': risk_type_dist,
            'confidence_histogram': conf_histogram,
            'suggestion_distribution': suggestion_dist,
        }

        return ok({
            'risk_decomposition': risk_decomposition,
            'per_detector_analysis': per_detector_analysis,
            'failed_detectors': failed_detectors,
            'label_analysis': label_analysis,
            'confidence_stats': confidence_stats,
            'suggestion_breakdown': suggestion_breakdown,
            'cross_detector_overlap': cross_detector_overlap[:50],
            'chart_data': chart_data,
            'conclusion': {
                'risk_level': risk_level,
                'risk_level_text': risk_level_text,
                'risk_score': round(risk_score, 4),
                'total_suspicious': total_suspicious,
                'total_samples': total_samples,
                'pct': pct,
                'summary': summary,
                'recommendations': recommendations,
            },
        })
