"""
algorithm_engine.base
─────────────────────
所有检测器的抽象基类。检测器必须继承 BaseDetector 并实现 run() 方法。

统一输出格式 DetectorOutput（TypedDict）。
"""

from __future__ import annotations

import abc
import logging
from typing import Any

logger = logging.getLogger('algorithm_engine')


class BaseDetector(abc.ABC):
    """所有检测器的统一抽象接口。"""

    #: 检测器唯一名称（snake_case），必须在子类中覆盖
    detector_name: str = 'base'
    #: 检测器类型：label_poison / distribution / anomaly / backdoor / influence
    detector_type: str = 'unknown'

    # ── 公开接口 ──────────────────────────────────────────────────

    def validate_input(self, dataset, config: dict) -> None:
        """校验输入合法性，失败时抛出 ValueError。子类可覆盖以增加额外校验。"""

    @abc.abstractmethod
    def run(self, dataset, baseline_dataset=None, config: dict | None = None) -> dict:
        """
        执行检测逻辑。

        Parameters
        ----------
        dataset : pandas.DataFrame
            待检测数据集。
        baseline_dataset : pandas.DataFrame | None
            基准数据集（分布漂移检测使用）。
        config : dict | None
            运行时参数，覆盖默认参数。

        Returns
        -------
        dict
            符合统一输出格式的字典（见 format_results）。
        """

    def safe_run(self, dataset, baseline_dataset=None, config: dict | None = None) -> dict:
        """
        带异常保护的 run()。单个检测器异常不影响整体任务。
        """
        try:
            self.validate_input(dataset, config or {})
            return self.run(dataset, baseline_dataset, config)
        except ImportError as e:
            logger.warning('[%s] 依赖缺失: %s', self.detector_name, e)
            return self.format_error(f'依赖缺失: {e}', status='dependency_missing')
        except Exception as e:
            logger.exception('[%s] 检测失败: %s', self.detector_name, e)
            return self.format_error(str(e))

    def format_results(self, raw_output: Any) -> dict:
        """将原始算法输出转换为统一格式（子类可覆盖）。"""
        raise NotImplementedError

    # ── 工具方法 ──────────────────────────────────────────────────

    def format_error(self, error_message: str, status: str = 'failed') -> dict:
        return {
            'detector_name': self.detector_name,
            'detector_type': self.detector_type,
            'detector_status': status,
            'risk_type': self.detector_type,
            'summary': {},
            'samples': [],
            'metrics': {},
            'artifacts': {},
            'error_message': error_message,
        }

    @staticmethod
    def make_result(
        detector_name: str,
        detector_type: str,
        summary: dict,
        samples: list[dict],
        metrics: dict | None = None,
        artifacts: dict | None = None,
    ) -> dict:
        """构建标准成功结果字典。"""
        return {
            'detector_name': detector_name,
            'detector_type': detector_type,
            'detector_status': 'success',
            'risk_type': detector_type,
            'summary': summary,
            'samples': samples,
            'metrics': metrics or {},
            'artifacts': artifacts or {},
            'error_message': None,
        }
