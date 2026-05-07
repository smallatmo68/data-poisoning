"""MMD（Maximum Mean Discrepancy）分布漂移检测器。使用 RBF kernel 实现。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from algorithm_engine.base import BaseDetector
from algorithm_engine.registry import register_detector

logger = logging.getLogger('algorithm_engine')


@register_detector
class MMDDriftDetector(BaseDetector):
    detector_name = 'mmd_drift'
    detector_type = 'distribution'

    def validate_input(self, dataset, config):
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError('dataset 必须是 pandas.DataFrame')

    def run(self, dataset: pd.DataFrame, baseline_dataset=None, config: dict | None = None) -> dict:
        config = config or {}
        label_field = config.get('label_field', '')
        threshold = config.get('mmd_threshold', 0.02)
        max_samples = config.get('max_samples', 2000)

        if baseline_dataset is None:
            return self.format_error('MMD 检测需要 baseline_dataset')

        # 先尝试 alibi-detect
        try:
            return self._run_alibi(dataset, baseline_dataset, label_field, threshold, max_samples)
        except ImportError:
            logger.debug('alibi-detect 不可用，使用自实现 MMD')

        return self._run_self_mmd(dataset, baseline_dataset, label_field, threshold, max_samples)

    def _run_alibi(self, dataset, baseline_dataset, label_field, threshold, max_samples) -> dict:
        from alibi_detect.cd import MMDDrift
        import tensorflow as tf  # alibi_detect CD 需要 TF

        X_ref = _extract_numeric(baseline_dataset, label_field, max_samples)
        X_test = _extract_numeric(dataset, label_field, max_samples)

        cd = MMDDrift(X_ref, p_val=threshold, backend='tensorflow')
        result = cd.predict(X_test)

        is_drift = bool(result['data']['is_drift'])
        p_val = float(result['data']['p_val'])
        mmd_val = float(result['data']['distance'])

        risk_level = 'high' if is_drift and p_val < 0.01 else 'medium' if is_drift else 'low'

        return self.make_result(
            detector_name=self.detector_name,
            detector_type=self.detector_type,
            summary={
                'is_drift': is_drift,
                'risk_level': risk_level,
                'method': 'alibi-detect-MMDDrift',
            },
            samples=[{
                'sample_id': 'GLOBAL',
                'row_index': -1,
                'risk_type': 'distribution_shift',
                'confidence': round(1.0 - p_val, 4),
                'suggestion': 'review' if is_drift else 'ignore',
                'detail': {'p_value': p_val, 'mmd': mmd_val, 'is_drift': is_drift},
            }] if is_drift else [],
            metrics={'p_value': p_val, 'mmd_distance': mmd_val},
        )

    def _run_self_mmd(self, dataset, baseline_dataset, label_field, threshold, max_samples) -> dict:
        X_ref = _extract_numeric(baseline_dataset, label_field, max_samples)
        X_test = _extract_numeric(dataset, label_field, max_samples)

        mmd_val = _compute_mmd(X_ref, X_test)
        is_drift = mmd_val > threshold
        risk_level = 'high' if mmd_val > threshold * 3 else 'medium' if is_drift else 'low'

        return self.make_result(
            detector_name=self.detector_name,
            detector_type=self.detector_type,
            summary={'is_drift': is_drift, 'risk_level': risk_level, 'method': 'self-MMD-RBF'},
            samples=[{
                'sample_id': 'GLOBAL',
                'row_index': -1,
                'risk_type': 'distribution_shift',
                'confidence': round(min(mmd_val / threshold, 1.0), 4),
                'suggestion': 'review' if is_drift else 'ignore',
                'detail': {'mmd': round(float(mmd_val), 6), 'threshold': threshold, 'is_drift': is_drift},
            }] if is_drift else [],
            metrics={'mmd_distance': round(float(mmd_val), 6)},
        )


def _extract_numeric(df: pd.DataFrame, label_field: str, max_samples: int) -> np.ndarray:
    cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != label_field]
    if not cols:
        raise ValueError('没有数值列可用于 MMD 检测')
    data = df[cols].fillna(0)
    if len(data) > max_samples:
        data = data.sample(max_samples, random_state=42)
    return StandardScaler().fit_transform(data.values.astype(float))


def _compute_mmd(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """使用 RBF kernel 计算 MMD²。"""
    n, m = len(X), len(Y)
    idx_x = np.random.choice(n, min(500, n), replace=False)
    idx_y = np.random.choice(m, min(500, m), replace=False)
    X, Y = X[idx_x], Y[idx_y]

    def rbf_kernel(a, b):
        sq_dist = np.sum((a[:, None] - b[None, :]) ** 2, axis=2)
        return np.exp(-sq_dist / (2 * sigma ** 2))

    kxx = rbf_kernel(X, X)
    kyy = rbf_kernel(Y, Y)
    kxy = rbf_kernel(X, Y)
    n, m = len(X), len(Y)
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())
