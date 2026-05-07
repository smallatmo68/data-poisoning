"""Influence Function 检测器（占位实现）。"""

from __future__ import annotations

from algorithm_engine.base import BaseDetector
from algorithm_engine.registry import register_detector


@register_detector
class InfluenceDetector(BaseDetector):
    """
    基于 Influence Function 的标签异常/毒化样本检测。
    需要可微分模型，第一版返回 not_implemented。
    后续可接入 Influence-Based-Glitch-Detection-main 适配器。
    """

    detector_name = 'influence'
    detector_type = 'influence'

    def validate_input(self, dataset, config):
        pass

    def run(self, dataset, baseline_dataset=None, config: dict | None = None) -> dict:
        return self.format_error(
            'Influence Function 检测需要可微分模型（PyTorch/TensorFlow），'
            '当前版本不支持。请使用 cleanlab 或 isolation_forest。',
            status='not_implemented',
        )
