"""LOF（Local Outlier Factor）异常检测器。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from algorithm_engine.base import BaseDetector
from algorithm_engine.registry import register_detector

logger = logging.getLogger('algorithm_engine')


@register_detector
class LOFDetector(BaseDetector):
    detector_name = 'lof'
    detector_type = 'anomaly'

    def validate_input(self, dataset, config):
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError('dataset 必须是 pandas.DataFrame')

    def run(self, dataset: pd.DataFrame, baseline_dataset=None, config: dict | None = None) -> dict:
        config = config or {}
        label_field = config.get('label_field', '')
        max_samples = config.get('max_samples', 3000)
        n_neighbors = config.get('n_neighbors', 20)
        contamination = config.get('contamination', 0.05)

        df = dataset.copy()
        if len(df) > max_samples:
            df = df.sample(max_samples, random_state=42)

        X = _build_features(df, label_field)

        clf = LocalOutlierFactor(
            n_neighbors=min(n_neighbors, len(df) - 1),
            contamination=contamination,
            n_jobs=-1,
        )
        predictions = clf.fit_predict(X)
        scores = -clf.negative_outlier_factor_  # 越大越异常

        outlier_indices = np.where(predictions == -1)[0]
        max_score = scores.max() if len(scores) > 0 else 1.0
        min_score = scores.min() if len(scores) > 0 else 0.0
        score_range = max_score - min_score if max_score != min_score else 1.0

        samples = []
        for idx in outlier_indices:
            normalized = float((scores[idx] - min_score) / score_range)
            samples.append({
                'sample_id': f'S-{df.index[idx]:06d}',
                'row_index': int(df.index[idx]),
                'risk_type': 'anomaly',
                'confidence': round(min(normalized, 1.0), 4),
                'suggestion': 'review',
                'detail': {
                    'lof_score': round(float(scores[idx]), 6),
                    'normalized_score': round(normalized, 4),
                },
            })

        samples.sort(key=lambda x: x['confidence'], reverse=True)
        r = len(samples) / len(df) if df.__len__() else 0
        risk_level = 'high' if r >= 0.15 else 'medium' if r >= 0.05 else 'low'

        return self.make_result(
            detector_name=self.detector_name,
            detector_type=self.detector_type,
            summary={
                'total_samples': len(df),
                'suspicious_count': len(samples),
                'risk_level': risk_level,
            },
            samples=samples,
        )


def _build_features(df: pd.DataFrame, label_field: str) -> np.ndarray:
    cols = [c for c in df.columns if c != label_field]
    parts = []
    for col in cols:
        if df[col].dtype == object:
            le = LabelEncoder()
            enc = le.fit_transform(df[col].astype(str).fillna('__nan__'))
            parts.append(enc.reshape(-1, 1))
        else:
            parts.append(df[col].fillna(0).values.reshape(-1, 1))
    if not parts:
        raise ValueError('没有可用特征列')
    X = np.hstack(parts).astype(float)
    return StandardScaler().fit_transform(X)
