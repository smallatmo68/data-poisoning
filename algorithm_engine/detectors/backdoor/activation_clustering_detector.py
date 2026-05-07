"""Activation Clustering 后门检测器（占位实现）。"""

from __future__ import annotations

from algorithm_engine.base import BaseDetector
from algorithm_engine.registry import register_detector


@register_detector
class ActivationClusteringDetector(BaseDetector):
    """
    Activation Clustering 后门检测。
    需要神经网络模型的激活值，第一版返回 not_implemented。
    """

    detector_name = 'activation_clustering'
    detector_type = 'backdoor'

    def validate_input(self, dataset, config):
        pass

    def run(self, dataset, baseline_dataset=None, config: dict | None = None) -> dict:
        return self.format_error(
            'Activation Clustering 需要预训练神经网络的激活值，'
            '当前版本不支持此检测器。请使用 spectral_signature 或 isolation_forest。',
            status='not_implemented',
        )
