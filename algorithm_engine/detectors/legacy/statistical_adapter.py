"""
统计异常检测器。
使用 Z-score 和 IQR 方法检测数值异常和标签分布异常。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from algorithm_engine.base import BaseDetector
from algorithm_engine.registry import register_detector

logger = logging.getLogger('algorithm_engine')


@register_detector
class StatisticalDetector(BaseDetector):
    """
    统计方法检测器：检测罕见标签、异常长度、数值离群点。
    """

    detector_name = 'statistical'
    detector_type = 'anomaly'

    def validate_input(self, dataset: pd.DataFrame, config: dict) -> None:
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError('dataset 必须是 pandas.DataFrame')

    def run(self, dataset: pd.DataFrame, baseline_dataset=None, config: dict | None = None) -> dict:
        config = config or {}
        label_field = config.get('label_field', '')
        text_field = config.get('text_field', '')
        max_samples = config.get('max_samples', 5000)
        z_threshold = config.get('z_threshold', 3.0)

        df = dataset.copy()
        if len(df) > max_samples:
            df = df.sample(max_samples, random_state=42)

        samples = []

        # 1. 数值列 Z-score 离群检测
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != label_field]
        if numeric_cols:
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) < 10:
                    continue
                mean = col_data.mean()
                std = col_data.std()
                if std == 0:
                    continue
                z_scores = ((col_data - mean) / std).abs()
                outliers = z_scores[z_scores > z_threshold]
                for idx, z in outliers.items():
                    samples.append({
                        'sample_id': f'S-{idx:06d}',
                        'row_index': int(idx),
                        'risk_type': 'anomaly',
                        'confidence': round(min(float(z) / (z_threshold * 2), 1.0), 4),
                        'suggestion': 'review',
                        'detail': {
                            'outlier_column': col,
                            'z_score': round(float(z), 4),
                            'value': round(float(col_data[idx]), 4),
                            'column_mean': round(float(mean), 4),
                            'column_std': round(float(std), 4),
                            'method': 'z_score',
                        },
                    })

        # 2. 标签分布异常检测
        if label_field and label_field in df.columns:
            label_counts = df[label_field].value_counts()
            total = len(df)
            for label, count in label_counts.items():
                ratio = count / total
                if ratio < 0.01 and count < 5:  # 罕见标签
                    rare_indices = df[df[label_field] == label].index.tolist()
                    for idx in rare_indices[:50]:
                        samples.append({
                            'sample_id': f'S-{idx:06d}',
                            'row_index': int(idx),
                            'risk_type': 'label_poison',
                            'confidence': round(max(0.5, 1 - ratio * 10), 4),
                            'suggestion': 'review',
                            'detail': {
                                'rare_label': str(label),
                                'label_count': int(count),
                                'label_ratio': round(ratio, 6),
                                'method': 'rare_label',
                            },
                        })

        # 3. 文本长度异常检测
        if text_field and text_field in df.columns:
            lengths = df[text_field].dropna().astype(str).str.len()
            if len(lengths) > 10:
                mean_len = lengths.mean()
                std_len = lengths.std()
                if std_len > 0:
                    for idx, length in lengths.items():
                        z = abs(length - mean_len) / std_len
                        if z > z_threshold:
                            samples.append({
                                'sample_id': f'S-{idx:06d}',
                                'row_index': int(idx),
                                'risk_type': 'anomaly',
                                'confidence': round(min(float(z) / (z_threshold * 2), 1.0), 4),
                                'suggestion': 'review',
                                'detail': {
                                    'text_length': int(length),
                                    'length_zscore': round(float(z), 4),
                                    'mean_length': round(float(mean_len), 4),
                                    'method': 'text_length',
                                },
                            })

        # 去重（同一样本保留最高置信度）
        seen = {}
        for s in samples:
            sid = s['sample_id']
            if sid not in seen or s['confidence'] > seen[sid]['confidence']:
                seen[sid] = s
        samples = sorted(seen.values(), key=lambda x: x['confidence'], reverse=True)

        risk_level = self._calc_risk_level(len(samples), len(df))

        return self.make_result(
            detector_name=self.detector_name,
            detector_type=self.detector_type,
            summary={
                'total_samples': len(df),
                'suspicious_count': len(samples),
                'risk_level': risk_level,
                'numeric_cols_checked': len(numeric_cols),
            },
            samples=samples,
            metrics={
                'z_threshold': z_threshold,
                'numeric_columns': numeric_cols[:20],
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
