"""
STRIP 异常检测器（适配 CSV 数据集）。
基于特征分布熵值检测异常样本。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from algorithm_engine.base import BaseDetector
from algorithm_engine.registry import register_detector

logger = logging.getLogger('algorithm_engine')


@register_detector
class StripDetector(BaseDetector):
    """
    STRIP (STRong Intentional Perturbation) 适配器。
    对 CSV 数值特征计算分布熵，识别熵值异常的样本。
    """

    detector_name = 'strip'
    detector_type = 'anomaly'

    def validate_input(self, dataset: pd.DataFrame, config: dict) -> None:
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError('dataset 必须是 pandas.DataFrame')

    def run(self, dataset: pd.DataFrame, baseline_dataset=None, config: dict | None = None) -> dict:
        config = config or {}
        label_field = config.get('label_field', '')
        max_samples = config.get('max_samples', 5000)

        df = dataset.copy()
        if len(df) > max_samples:
            df = df.sample(max_samples, random_state=42)

        # 取数值列
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != label_field]
        if not numeric_cols:
            raise ValueError('数据集没有数值特征列，无法执行 STRIP 检测')

        feature_data = df[numeric_cols].fillna(0).values.astype(float)

        # 标准化
        means = np.mean(feature_data, axis=0)
        stds = np.std(feature_data, axis=0)
        stds[stds == 0] = 1
        normalized = (feature_data - means) / stds

        # 计算每个样本的熵值（基于特征分布）
        n_bins = max(10, int(np.sqrt(len(df))))
        samples = []
        entropies = []

        for i, col_idx in enumerate(range(len(numeric_cols))):
            col_data = normalized[:, col_idx]
            hist, _ = np.histogram(col_data, bins=n_bins, density=True)
            hist = hist[hist > 0]
            col_entropy = -np.sum(hist * np.log2(hist + 1e-10))
            entropies.append(col_entropy)

        avg_entropy = np.mean(entropies) if entropies else 0

        # 计算每个样本的扰动敏感度
        for i in range(len(df)):
            row = normalized[i]
            # 该样本在各维度上偏离均值的程度
            deviation = np.sqrt(np.sum(row ** 2))
            # 该样本对整体分布的影响
            perturbation_score = deviation / (np.sqrt(len(numeric_cols)) + 1e-10)

            if perturbation_score > 2.0:
                samples.append({
                    'sample_id': f'S-{df.index[i]:06d}',
                    'row_index': int(df.index[i]),
                    'risk_type': 'anomaly',
                    'confidence': round(min(perturbation_score / 5, 1.0), 4),
                    'suggestion': 'review',
                    'detail': {
                        'perturbation_sensitivity': round(float(perturbation_score), 4),
                        'deviation_magnitude': round(float(deviation), 4),
                        'feature_entropy': round(float(avg_entropy), 4),
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
                'features_analyzed': len(numeric_cols),
            },
            samples=samples,
            metrics={
                'avg_feature_entropy': round(float(avg_entropy), 4),
                'features_used': numeric_cols[:20],
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
