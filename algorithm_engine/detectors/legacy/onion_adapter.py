"""
ONION 文本异常检测器（适配 CSV 数据集）。
基于罕见词频率和文本长度异常检测后门触发器。
"""

from __future__ import annotations

import logging
import math
from collections import Counter

import numpy as np
import pandas as pd

from algorithm_engine.base import BaseDetector
from algorithm_engine.registry import register_detector

logger = logging.getLogger('algorithm_engine')


@register_detector
class OnionDetector(BaseDetector):
    """
    ONION (Outlier Neutralization for Identifying Poisoned Samples) 适配器。
    对 CSV 数据集中的文本列进行异常词频率分析。
    """

    detector_name = 'onion'
    detector_type = 'backdoor'

    def validate_input(self, dataset: pd.DataFrame, config: dict) -> None:
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError('dataset 必须是 pandas.DataFrame')

    def run(self, dataset: pd.DataFrame, baseline_dataset=None, config: dict | None = None) -> dict:
        config = config or {}
        text_field = config.get('text_field') or self._detect_text_field(dataset)
        max_samples = config.get('max_samples', 5000)

        df = dataset.copy()
        if len(df) > max_samples:
            df = df.sample(max_samples, random_state=42)

        if not text_field or text_field not in df.columns:
            raise ValueError(f'未找到文本列（尝试: {text_field}），请在 config 中指定 text_field')

        texts = df[text_field].dropna().astype(str)

        # 构建全局词频
        all_words = []
        for t in texts:
            all_words.extend(t.lower().split())
        word_freq = Counter(all_words)
        total_words = sum(word_freq.values())

        # 计算每个样本的异常分数
        samples = []
        for idx, text in texts.items():
            words = text.lower().split()
            if not words:
                continue

            # 罕见词比例
            rare_count = sum(1 for w in words if word_freq.get(w, 0) <= 2)
            rare_ratio = rare_count / len(words)

            # 文本长度 z-score
            lengths = texts.str.split().str.len()
            mean_len = lengths.mean()
            std_len = lengths.std() or 1
            len_zscore = abs(len(words) - mean_len) / std_len

            # 综合异常分数
            anomaly_score = rare_ratio * 0.6 + min(len_zscore / 3, 1) * 0.4

            if anomaly_score > 0.3:
                samples.append({
                    'sample_id': f'S-{idx:06d}',
                    'row_index': int(idx),
                    'risk_type': 'backdoor',
                    'confidence': round(min(anomaly_score, 1.0), 4),
                    'suggestion': 'review',
                    'detail': {
                        'rare_word_ratio': round(rare_ratio, 4),
                        'text_length': len(words),
                        'length_zscore': round(len_zscore, 4),
                        'rare_words': [w for w in words if word_freq.get(w, 0) <= 2][:10],
                    },
                })

        samples.sort(key=lambda x: x['confidence'], reverse=True)
        risk_level = self._calc_risk_level(len(samples), len(texts))

        return self.make_result(
            detector_name=self.detector_name,
            detector_type=self.detector_type,
            summary={
                'total_samples': len(texts),
                'suspicious_count': len(samples),
                'risk_level': risk_level,
                'text_field': text_field,
            },
            samples=samples,
            metrics={
                'vocabulary_size': len(word_freq),
                'avg_text_length': float(texts.str.split().str.len().mean()),
            },
        )

    @staticmethod
    def _detect_text_field(df: pd.DataFrame) -> str:
        for candidate in ('text', 'content', 'sentence', 'input', 'review', 'comment', 'message'):
            if candidate in df.columns:
                return candidate
        # 选择字符串列中文本最长的
        str_cols = df.select_dtypes(include=['object']).columns
        if len(str_cols) > 0:
            avg_lens = {col: df[col].astype(str).str.len().mean() for col in str_cols}
            return max(avg_lens, key=avg_lens.get)
        return ''

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
