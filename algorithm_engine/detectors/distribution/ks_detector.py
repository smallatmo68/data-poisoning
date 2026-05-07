"""KS 检验分布漂移检测器。比较待检测数据集与基准数据集。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from algorithm_engine.base import BaseDetector
from algorithm_engine.registry import register_detector

logger = logging.getLogger('algorithm_engine')


@register_detector
class KSDriftDetector(BaseDetector):
    detector_name = 'ks_drift'
    detector_type = 'distribution'

    def validate_input(self, dataset, config):
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError('dataset 必须是 pandas.DataFrame')

    def run(self, dataset: pd.DataFrame, baseline_dataset=None, config: dict | None = None) -> dict:
        config = config or {}
        alpha = config.get('alpha', 0.05)
        label_field = config.get('label_field', '')

        if baseline_dataset is None or not isinstance(baseline_dataset, pd.DataFrame):
            return self.format_error('KS 检测需要 baseline_dataset，请提供基准数据集')

        numeric_cols = [
            c for c in dataset.select_dtypes(include=[np.number]).columns
            if c != label_field and c in baseline_dataset.columns
        ]

        if not numeric_cols:
            return self.format_error('没有可比较的数值列')

        drift_results = []
        drifted_cols = []

        for col in numeric_cols:
            a = dataset[col].dropna().values
            b = baseline_dataset[col].dropna().values
            if len(a) < 5 or len(b) < 5:
                continue
            ks_stat, p_value = stats.ks_2samp(a, b)
            is_drift = p_value < alpha
            drift_results.append({
                'column': col,
                'ks_statistic': round(float(ks_stat), 6),
                'p_value': round(float(p_value), 6),
                'is_drift': is_drift,
            })
            if is_drift:
                drifted_cols.append(col)

        n_drifted = len(drifted_cols)
        n_total = len(drift_results)
        risk_level = 'high' if n_drifted / n_total >= 0.5 else 'medium' if n_drifted > 0 else 'low' if n_total else 'unknown'

        # 分布漂移没有"样本级"结果，用列级结果模拟
        samples = [{
            'sample_id': f'COL-{r["column"]}',
            'row_index': -1,
            'risk_type': 'distribution_shift',
            'confidence': round(r['ks_statistic'], 4),
            'suggestion': 'review',
            'detail': r,
        } for r in drift_results if r['is_drift']]

        return self.make_result(
            detector_name=self.detector_name,
            detector_type=self.detector_type,
            summary={
                'total_columns': n_total,
                'drifted_columns': n_drifted,
                'risk_level': risk_level,
                'alpha': alpha,
            },
            samples=samples,
            metrics={'column_results': drift_results},
        )
