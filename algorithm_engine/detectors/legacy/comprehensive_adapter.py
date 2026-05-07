"""
综合投票检测器。
聚合多个检测器结果，通过多数投票给出最终判定。
"""

from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

from algorithm_engine.base import BaseDetector
from algorithm_engine.registry import register_detector, get_detector

logger = logging.getLogger('algorithm_engine')


@register_detector
class ComprehensiveDetector(BaseDetector):
    """
    综合投票检测器：运行多个子检测器，对同一样本的检测结果进行投票聚合。
    """

    detector_name = 'comprehensive'
    detector_type = 'anomaly'

    def validate_input(self, dataset: pd.DataFrame, config: dict) -> None:
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError('dataset 必须是 pandas.DataFrame')

    def run(self, dataset: pd.DataFrame, baseline_dataset=None, config: dict | None = None) -> dict:
        config = config or {}
        max_samples = config.get('max_samples', 3000)

        df = dataset.copy()
        if len(df) > max_samples:
            df = df.sample(max_samples, random_state=42)

        # 运行子检测器
        sub_detectors = ['isolation_forest', 'lof', 'statistical']
        all_samples = defaultdict(list)

        for det_name in sub_detectors:
            detector = get_detector(det_name)
            if detector is None:
                continue
            try:
                result = detector.safe_run(df, baseline_dataset, config)
                if result.get('detector_status') == 'success':
                    for s in result.get('samples', []):
                        all_samples[s['sample_id']].append({
                            'detector': det_name,
                            'confidence': s['confidence'],
                            'risk_type': s['risk_type'],
                            'suggestion': s['suggestion'],
                        })
            except Exception as e:
                logger.warning('综合检测器子检测器 %s 失败: %s', det_name, e)

        # 投票聚合
        samples = []
        for sample_id, detections in all_samples.items():
            vote_count = len(detections)
            if vote_count < 2:
                continue  # 至少 2 个检测器标记才算可疑

            avg_conf = sum(d['confidence'] for d in detections) / vote_count
            # 多数投票决定风险类型
            type_votes = defaultdict(int)
            for d in detections:
                type_votes[d['risk_type']] += 1
            majority_type = max(type_votes, key=type_votes.get)

            confidence = min(avg_conf * (vote_count / len(sub_detectors)), 1.0)

            samples.append({
                'sample_id': sample_id,
                'row_index': int(sample_id.split('-')[-1]) if '-' in sample_id else 0,
                'risk_type': majority_type,
                'confidence': round(confidence, 4),
                'suggestion': 'review' if vote_count < len(sub_detectors) else 'remove',
                'detail': {
                    'vote_count': vote_count,
                    'total_detectors': len(sub_detectors),
                    'detectors': [d['detector'] for d in detections],
                    'avg_confidence': round(avg_conf, 4),
                    'type_votes': dict(type_votes),
                },
            })

        samples.sort(key=lambda x: x['confidence'], reverse=True)
        risk_level = self._calc_risk_level(len(samples), len(df))

        return self.make_result(
            detector_name=self.detector_name,
            detector_type=self.detector_type,
            summary={
                'total_samples': len(df),
                'suspicious_count': len(samples),
                'risk_level': risk_level,
                'sub_detectors_used': sub_detectors,
            },
            samples=samples,
            metrics={
                'sub_detectors_run': len(sub_detectors),
                'min_votes_required': 2,
            },
        )

    @staticmethod
    def _calc_risk_level(suspicious: int, total: int) -> str:
        if total == 0:
            return 'unknown'
        ratio = suspicious / total
        if ratio >= 0.15:
            return 'high'
        if ratio >= 0.05:
            return 'medium'
        return 'low'
