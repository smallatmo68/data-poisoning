import logging
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from django.conf import settings

from apps.detection.models import AlgorithmConfig, DetectionResult, DetectionTask
from algorithm_engine.engine import DetectionEngine
from algorithm_engine.mongo_client import save_detection_detail

logger = logging.getLogger('dpds.detection')


def run_detection(task_id: str):
    """执行检测任务主流程。被 Celery task 或同步线程调用。"""
    try:
        task = DetectionTask.objects.select_related('dataset').get(id=task_id)
    except DetectionTask.DoesNotExist:
        logger.error('DetectionTask %s 不存在', task_id)
        return

    task.status = DetectionTask.STATUS_RUNNING
    task.started_at = datetime.now(timezone.utc)
    task.save(update_fields=['status', 'started_at'])

    try:
        failed_count = _do_detection(task)
        task.progress = 100
        if failed_count > 0:
            # 部分检测器失败，但任务本身完成
            task.status = DetectionTask.STATUS_SUCCESS
            config = task.detector_config or {}
            requested = config.get('detectors', [])
            if failed_count >= len(requested):
                task.status = DetectionTask.STATUS_FAILED
                task.error_message = '所有检测器均执行失败'
            else:
                logger.warning('任务 %s 部分检测器失败: %d/%d', task_id, failed_count, len(requested))
        else:
            task.status = DetectionTask.STATUS_SUCCESS
    except Exception as e:
        logger.exception('检测任务 %s 失败: %s', task_id, e)
        task.status = DetectionTask.STATUS_FAILED
        task.error_message = str(e)
    finally:
        task.finished_at = datetime.now(timezone.utc)
        task.save(update_fields=['status', 'progress', 'error_message', 'finished_at'])


def _do_detection(task: DetectionTask) -> int:
    """执行检测，返回失败的检测器数量。"""
    dataset = task.dataset
    config = task.detector_config or {}
    detector_names = config.get('detectors', ['cleanlab', 'isolation_forest'])

    # 加载数据集
    storage_path = dataset.storage_path or str(settings.MEDIA_ROOT / str(dataset.file))
    df = pd.read_csv(storage_path)

    # 加载基准数据集（如有）
    baseline_df = None
    if task.baseline_dataset:
        bpath = task.baseline_dataset.storage_path or str(settings.MEDIA_ROOT / str(task.baseline_dataset.file))
        baseline_df = pd.read_csv(bpath)

    # 全局 config
    run_config = {
        'label_field': dataset.label_field or config.get('label_field', ''),
        'text_field': dataset.text_field or config.get('text_field', ''),
    }
    run_config.update(config)

    # 从 AlgorithmConfig 读取权重
    weights = {}
    for ac in AlgorithmConfig.objects.filter(enabled=True):
        weights[ac.detector_type] = ac.weight

    def progress_cb(pct, msg):
        task.progress = pct
        task.save(update_fields=['progress'])
        _update_redis_progress(str(task.id), pct, msg)

    engine = DetectionEngine()
    merged = engine.run(
        dataset=df,
        detector_names=detector_names,
        config=run_config,
        baseline_dataset=baseline_df,
        weights=weights or None,
        progress_callback=progress_cb,
    )

    task.risk_score = merged.get('risk_score')

    # 统计失败检测器
    meta = merged.get('meta', {})
    failed_detectors = meta.get('detectors_failed', [])
    failed_count = len(failed_detectors)

    # 保存详情到 MongoDB
    doc_id = save_detection_detail('detection_results', {
        'task_id': str(task.id),
        'dataset_id': str(dataset.id),
        'risk_score': merged.get('risk_score'),
        'per_detector': merged.get('per_detector'),
        'samples': merged.get('samples', [])[:500],
        'meta': meta,
    })

    # 保存索引到 MySQL（每个可疑样本一条记录，含快照）
    results_to_create = []
    raw_df = df
    proc_df = None
    if dataset.processed_path and dataset.processed_path != storage_path:
        try:
            proc_df = pd.read_csv(dataset.processed_path)
        except Exception:
            pass

    for s in merged.get('samples', [])[:2000]:
        row_idx = s.get('row_index')
        raw_snapshot = {}
        proc_snapshot = {}

        if row_idx is not None:
            if row_idx < len(raw_df):
                row = raw_df.iloc[row_idx]
                raw_snapshot = {col: _safe_val(val) for col, val in row.items()}
            if proc_df is not None and row_idx < len(proc_df):
                row_p = proc_df.iloc[row_idx]
                proc_snapshot = {col: _safe_val(val) for col, val in row_p.items()}

        detail = s.get('detail', {})
        triggered = _extract_triggered_features(detail, s)
        reason = _build_reason(s, detail)

        results_to_create.append(DetectionResult(
            task=task,
            sample_id=s.get('sample_id', ''),
            risk_type=s.get('risk_type', 'unknown'),
            confidence=s.get('confidence', 0.0),
            suggestion=s.get('suggestion', 'review'),
            detector_name=s.get('detector_name', merged.get('per_detector', [{}])[0].get('detector_name', '')),
            detail_doc_id=doc_id or '',
            triggered_features=triggered,
            metric_detail=detail,
            reason=reason,
            raw_data_snapshot=raw_snapshot,
            processed_data_snapshot=proc_snapshot,
        ))

    if results_to_create:
        DetectionResult.objects.bulk_create(results_to_create, batch_size=500)

    task.save(update_fields=['risk_score'])
    return failed_count


def _update_redis_progress(task_id: str, progress: int, message: str):
    try:
        import redis
        from django.conf import settings
        r = redis.from_url(settings.REDIS_URL)
        r.setex(f'dpds:task:{task_id}:progress', 3600, f'{progress}:{message}')
    except Exception:
        pass


def get_task_progress_from_redis(task_id: str) -> dict:
    try:
        import redis
        from django.conf import settings
        r = redis.from_url(settings.REDIS_URL)
        val = r.get(f'dpds:task:{task_id}:progress')
        if val:
            val = val.decode()
            pct, _, msg = val.partition(':')
            return {'progress': int(pct), 'message': msg}
    except Exception:
        pass
    return {}


def _safe_val(val):
    """确保值可 JSON 序列化。"""
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    if pd.isna(val):
        return None
    if isinstance(val, (int, float, str, bool)):
        return val
    return str(val)


def _extract_triggered_features(detail: dict, sample: dict) -> list:
    """从检测结果详情中提取触发特征列表。"""
    features = []
    if 'original_label' in detail and 'predicted_label' in detail:
        features.append(f'标签冲突: {detail["original_label"]} → {detail["predicted_label"]}')
    if 'label_quality' in detail:
        features.append(f'标签质量: {detail["label_quality"]:.4f}')
    if 'anomaly_score' in detail:
        features.append(f'异常分数: {detail["anomaly_score"]:.4f}')
    if 'trigger_words' in detail:
        features.extend(detail['trigger_words'])
    if 'drift_features' in detail:
        features.extend(detail['drift_features'])
    if 'entropy' in detail:
        features.append(f'熵值: {detail["entropy"]:.4f}')
    if 'perturbation_sensitivity' in detail:
        features.append(f'扰动敏感度: {detail["perturbation_sensitivity"]:.4f}')
    if not features and sample.get('recommended_label'):
        features.append(f'推荐标签: {sample["recommended_label"]}')
    return features


def _build_reason(sample: dict, detail: dict) -> str:
    """构建自然语言判定原因。"""
    risk_type = sample.get('risk_type', 'unknown')
    conf = sample.get('confidence', 0)
    suggestion = sample.get('suggestion', 'review')

    parts = []
    if risk_type == 'label_poison':
        if 'original_label' in detail and 'predicted_label' in detail:
            parts.append(
                f'该样本原始标签为"{detail["original_label"]}"，'
                f'但模型预测标签为"{detail["predicted_label"]}"'
            )
            if 'label_quality' in detail:
                parts.append(f'标签质量分数仅 {detail["label_quality"]:.4f}')
        else:
            parts.append('该样本标签质量较低')
    elif risk_type == 'backdoor':
        parts.append('该样本具有后门攻击特征')
        if 'trigger_words' in detail:
            parts.append(f'触发词: {", ".join(detail["trigger_words"][:5])}')
    elif risk_type == 'distribution_shift':
        parts.append('该样本与基准数据集分布存在显著偏移')
    elif risk_type == 'anomaly':
        parts.append('该样本在特征空间中表现为异常')
        if 'anomaly_score' in detail:
            parts.append(f'异常分数: {detail["anomaly_score"]:.4f}')
    else:
        parts.append('该样本被检测为可疑')

    parts.append(f'置信度 {conf*100:.1f}%')

    action_map = {'remove': '建议删除', 'relabel': '建议重新标注', 'review': '建议人工复查', 'ignore': '可忽略'}
    parts.append(action_map.get(suggestion, '建议人工复查'))

    return '，'.join(parts) + '。'
