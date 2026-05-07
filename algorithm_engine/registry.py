"""
algorithm_engine.registry
──────────────────────────
检测器注册中心。通过 @register_detector 装饰器注册，通过 get_detector() 获取实例。
"""

from __future__ import annotations

import importlib
import logging
from typing import Type

from .base import BaseDetector

logger = logging.getLogger('algorithm_engine')

_REGISTRY: dict[str, Type[BaseDetector]] = {}

# 检测器名称 → 模块路径映射（延迟加载，避免在 Django 启动时全部 import）
_DETECTOR_MODULES: dict[str, str] = {
    'cleanlab': 'algorithm_engine.detectors.label_poison.cleanlab_detector',
    'isolation_forest': 'algorithm_engine.detectors.anomaly.isolation_forest_detector',
    'lof': 'algorithm_engine.detectors.anomaly.lof_detector',
    'ks_drift': 'algorithm_engine.detectors.distribution.ks_detector',
    'mmd_drift': 'algorithm_engine.detectors.distribution.mmd_detector',
    'spectral_signature': 'algorithm_engine.detectors.backdoor.spectral_signature_detector',
    'activation_clustering': 'algorithm_engine.detectors.backdoor.activation_clustering_detector',
    'influence': 'algorithm_engine.detectors.influence.influence_detector',
    # 原有 legacy 检测器（兼容 datasets/services.py）
    'onion': 'algorithm_engine.detectors.legacy.onion_adapter',
    'strip': 'algorithm_engine.detectors.legacy.strip_adapter',
    'statistical': 'algorithm_engine.detectors.legacy.statistical_adapter',
    'comprehensive': 'algorithm_engine.detectors.legacy.comprehensive_adapter',
}


def register_detector(cls: Type[BaseDetector]) -> Type[BaseDetector]:
    """装饰器：将检测器类注册到全局注册中心。"""
    _REGISTRY[cls.detector_name] = cls
    return cls


def get_detector(name: str) -> BaseDetector | None:
    """
    按名称获取检测器实例。
    如果尚未加载，先尝试从 _DETECTOR_MODULES 延迟导入。
    """
    if name not in _REGISTRY:
        module_path = _DETECTOR_MODULES.get(name)
        if module_path:
            try:
                importlib.import_module(module_path)
            except Exception as e:
                logger.warning('加载检测器模块 %s 失败: %s', module_path, e)

    cls = _REGISTRY.get(name)
    if cls is None:
        logger.warning('检测器 %s 未注册', name)
        return None
    return cls()


def list_detectors() -> list[str]:
    """返回所有已知检测器名称（包括未加载的）。"""
    return sorted(set(list(_REGISTRY.keys()) + list(_DETECTOR_MODULES.keys())))
