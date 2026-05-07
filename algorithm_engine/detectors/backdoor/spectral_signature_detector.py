"""Spectral Signature 后门检测器（BackdoorBench 适配器）。第一版返回接口完整的占位实现。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from algorithm_engine.base import BaseDetector
from algorithm_engine.registry import register_detector

logger = logging.getLogger('algorithm_engine')


@register_detector
class SpectralSignatureDetector(BaseDetector):
    """
    Spectral Signature 后门检测。
    - 真实实现需要 torch + 预训练模型的 feature representations。
    - 第一版：如果 torch 可用，尝试使用 SVD 方法；否则返回占位提示。
    """

    detector_name = 'spectral_signature'
    detector_type = 'backdoor'

    def validate_input(self, dataset, config):
        pass

    def run(self, dataset, baseline_dataset=None, config: dict | None = None) -> dict:
        config = config or {}

        # 仅支持图像数据集或特征矩阵（numpy array / DataFrame with numeric features）
        if isinstance(dataset, pd.DataFrame):
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                return self._run_svd(dataset[numeric_cols].fillna(0).values, config)

        return self.format_error(
            '图像后门检测（Spectral Signature）需要特征矩阵或图像数据集。'
            '对于 CSV 数据集，建议使用 cleanlab 或 isolation_forest 检测器。',
            status='not_applicable',
        )

    def _run_svd(self, features: np.ndarray, config: dict) -> dict:
        """基于奇异值分解的简化 Spectral Signature 检测。"""
        max_samples = config.get('max_samples', 3000)
        top_k = config.get('spectral_top_k', 1)

        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]

        # 中心化
        features = features - features.mean(axis=0)

        # SVD
        try:
            U, S, Vt = np.linalg.svd(features, full_matrices=False)
        except np.linalg.LinAlgError:
            return self.format_error('SVD 计算失败')

        # Spectral Signature: 每个样本在 top-k 奇异向量上的投影能量
        top_vt = Vt[:top_k]  # (top_k, feature_dim)
        projections = features @ top_vt.T  # (n, top_k)
        scores = np.linalg.norm(projections, axis=1)  # (n,)

        threshold_pct = config.get('spectral_threshold_pct', 95)
        threshold = np.percentile(scores, threshold_pct)
        suspicious_mask = scores > threshold

        samples = []
        for idx in np.where(suspicious_mask)[0]:
            score = float(scores[idx])
            max_score = float(scores.max()) if scores.max() > 0 else 1.0
            samples.append({
                'sample_id': f'S-{idx:06d}',
                'row_index': int(idx),
                'risk_type': 'backdoor',
                'confidence': round(score / max_score, 4),
                'suggestion': 'review',
                'detail': {
                    'spectral_score': round(score, 6),
                    'threshold': round(float(threshold), 6),
                },
            })

        samples.sort(key=lambda x: x['confidence'], reverse=True)
        r = len(samples) / len(features) if len(features) else 0
        risk_level = 'high' if r >= 0.1 else 'medium' if r >= 0.03 else 'low'

        return self.make_result(
            detector_name=self.detector_name,
            detector_type=self.detector_type,
            summary={
                'total_samples': len(features),
                'suspicious_count': len(samples),
                'risk_level': risk_level,
                'method': 'SVD-SpectralSignature',
                'note': '基于 CSV 数值特征的简化实现；真实图像后门检测需要 torch 模型特征',
            },
            samples=samples,
            metrics={
                'top_singular_value': round(float(S[0]), 4),
                'threshold_pct': threshold_pct,
            },
        )
