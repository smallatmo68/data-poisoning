"""
algorithm_engine.engine
────────────────────────
DetectionEngine：统一编排多个检测器的执行流程。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .registry import get_detector

logger = logging.getLogger('algorithm_engine')

# 默认融合权重
DEFAULT_WEIGHTS = {
    'label_poison': 0.35,
    'distribution': 0.25,
    'backdoor': 0.30,
    'anomaly': 0.10,
    'influence': 0.05,
}


class DetectionEngine:
    """
    统一调度多个检测器，合并结果，计算综合风险分数。

    用法：
        engine = DetectionEngine()
        result = engine.run(
            dataset=df,
            detector_names=['cleanlab', 'isolation_forest', 'ks_drift'],
            config={
                'label_field': 'label',
                'cleanlab': {'max_samples': 3000},
            },
            baseline_dataset=baseline_df,
            progress_callback=lambda p, msg: ...,
        )
    """

    def run(
        self,
        dataset: pd.DataFrame,
        detector_names: list[str],
        config: dict | None = None,
        baseline_dataset: pd.DataFrame | None = None,
        weights: dict | None = None,
        progress_callback=None,
    ) -> dict:
        config = config or {}
        weights = weights or DEFAULT_WEIGHTS

        n = len(detector_names)
        all_results: list[dict] = []
        failed_detectors: list[str] = []

        for i, name in enumerate(detector_names):
            pct = int((i / n) * 90)
            _emit(progress_callback, pct, f'正在运行检测器: {name}')

            detector = get_detector(name)
            if detector is None:
                logger.warning('检测器 %s 未找到，跳过', name)
                failed_detectors.append(name)
                continue

            # 取该检测器专属 config（如果有）
            det_config = {**config, **(config.get(name) or {})}

            result = detector.safe_run(dataset, baseline_dataset, det_config)
            all_results.append(result)

            if result['detector_status'] not in ('success',):
                failed_detectors.append(name)

        _emit(progress_callback, 92, '合并检测结果')
        merged = self._merge_results(all_results, weights)

        _emit(progress_callback, 100, '检测完成')
        merged['meta'] = {
            'detectors_requested': detector_names,
            'detectors_failed': failed_detectors,
            'finished_at': datetime.now(timezone.utc).isoformat(),
        }
        return merged

    # ── 私有方法 ─────────────────────────────────────────────────────

    @staticmethod
    def _merge_results(all_results: list[dict], weights: dict) -> dict:
        total_samples = 0
        all_suspicious: list[dict] = []
        per_detector: list[dict] = []
        weighted_scores: list[tuple[float, float]] = []  # (score, weight)

        for res in all_results:
            det_type = res.get('detector_type', 'unknown')
            w = weights.get(det_type, 0.1)

            summary = res.get('summary', {})
            tot = summary.get('total_samples', 0)
            sus = summary.get('suspicious_count', len(res.get('samples', [])))
            det_score = sus / tot if tot > 0 else 0.0

            if tot > 0:
                total_samples = max(total_samples, tot)
            if res.get('detector_status') == 'success' and tot > 0:
                weighted_scores.append((det_score, w))

            per_detector.append({
                'detector_name': res.get('detector_name'),
                'detector_type': det_type,
                'status': res.get('detector_status'),
                'risk_level': summary.get('risk_level', 'unknown'),
                'suspicious_count': sus,
                'total_samples': tot,
                'error_message': res.get('error_message'),
            })

            all_suspicious.extend(res.get('samples', []))

        # 融合风险分数
        if weighted_scores:
            total_w = sum(w for _, w in weighted_scores)
            risk_score = sum(s * w for s, w in weighted_scores) / total_w if total_w else 0.0
        else:
            risk_score = 0.0

        risk_score = round(min(risk_score, 1.0), 4)
        risk_level = 'high' if risk_score >= 0.15 else 'medium' if risk_score >= 0.05 else 'low'

        # 去重（同一 sample_id 保留最高置信度）
        seen: dict[str, dict] = {}
        for s in all_suspicious:
            sid = s.get('sample_id', '')
            if sid not in seen or s.get('confidence', 0) > seen[sid].get('confidence', 0):
                seen[sid] = s
        unique_samples = sorted(seen.values(), key=lambda x: x.get('confidence', 0), reverse=True)

        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'total_samples': total_samples,
            'total_suspicious': len(unique_samples),
            'per_detector': per_detector,
            'samples': unique_samples,
        }


def _emit(callback, progress: int, message: str):
    if callback:
        try:
            callback(progress, message)
        except Exception:
            pass
