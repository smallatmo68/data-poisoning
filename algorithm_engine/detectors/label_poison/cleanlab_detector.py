"""
Cleanlab 标签投毒检测器。
依赖 cleanlab>=2.5.0（可选导入）。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from algorithm_engine.base import BaseDetector
from algorithm_engine.registry import register_detector

logger = logging.getLogger('algorithm_engine')

try:
    from cleanlab.filter import find_label_issues
    from cleanlab.count import get_confident_thresholds
    _CLEANLAB_AVAILABLE = True
except ImportError:
    _CLEANLAB_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import LabelEncoder
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


@register_detector
class CleanlabDetector(BaseDetector):
    """
    基于 Confident Learning（Cleanlab）的标签投毒检测器。
    对 CSV 数据集，自动训练一个逻辑回归分类器获取预测概率，
    然后用 cleanlab 识别标签异常样本。
    """

    detector_name = 'cleanlab'
    detector_type = 'label_poison'

    def validate_input(self, dataset: pd.DataFrame, config: dict) -> None:
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError('dataset 必须是 pandas.DataFrame')

    def run(self, dataset: pd.DataFrame, baseline_dataset=None, config: dict | None = None) -> dict:
        if not _SKLEARN_AVAILABLE:
            raise ImportError('scikit-learn 未安装')

        config = config or {}
        label_field = config.get('label_field') or _auto_detect_label(dataset)
        if not label_field or label_field not in dataset.columns:
            raise ValueError(f'未找到标签列（尝试: {label_field}），请在 config 中指定 label_field')

        max_samples = config.get('max_samples', 5000)
        df = dataset.copy()
        if len(df) > max_samples:
            df = df.sample(max_samples, random_state=42)

        df = df.dropna(subset=[label_field])

        # 编码 label
        le = LabelEncoder()
        y = le.fit_transform(df[label_field].astype(str))
        n_classes = len(le.classes_)

        if n_classes < 2:
            raise ValueError('标签类别数 < 2，无法进行标签质量检测')

        # 构建特征矩阵（仅数值列）
        X = _build_feature_matrix(df, label_field)

        # 获取预测概率（交叉验证）
        clf = LogisticRegression(max_iter=500, random_state=42, C=1.0)
        pred_probs = cross_val_predict(clf, X, y, cv=min(5, n_classes), method='predict_proba')

        samples = []

        if _CLEANLAB_AVAILABLE:
            # 使用真实 cleanlab
            try:
                issue_indices = find_label_issues(
                    labels=y,
                    pred_probs=pred_probs,
                    return_indices_ranked_by='self_confidence',
                )
                label_quality_scores = 1.0 - pred_probs[np.arange(len(y)), y]

                for rank, idx in enumerate(issue_indices):
                    original_label = le.classes_[y[idx]]
                    predicted_label = le.classes_[np.argmax(pred_probs[idx])]
                    lq_score = float(label_quality_scores[idx])
                    confidence = float(1.0 - lq_score)

                    samples.append({
                        'sample_id': f'S-{df.index[idx]:06d}',
                        'row_index': int(df.index[idx]),
                        'risk_type': 'label_poison',
                        'confidence': round(confidence, 4),
                        'suggestion': 'relabel',
                        'recommended_label': str(predicted_label),
                        'detail': {
                            'original_label': str(original_label),
                            'predicted_label': str(predicted_label),
                            'label_quality': round(lq_score, 4),
                        },
                    })
            except Exception as e:
                logger.warning('Cleanlab find_label_issues 失败，降级到自实现: %s', e)
                samples = _fallback_detection(y, pred_probs, le, df)
        else:
            # cleanlab 未安装，使用 self-confidence 方法
            samples = _fallback_detection(y, pred_probs, le, df)

        risk_level = _calc_risk_level(len(samples), len(df))

        return self.make_result(
            detector_name=self.detector_name,
            detector_type=self.detector_type,
            summary={
                'total_samples': len(df),
                'suspicious_count': len(samples),
                'risk_level': risk_level,
                'n_classes': n_classes,
                'cleanlab_available': _CLEANLAB_AVAILABLE,
            },
            samples=samples,
            metrics={
                'label_classes': list(le.classes_),
                'avg_label_quality': float(np.mean(pred_probs[np.arange(len(y)), y])),
            },
        )


# ── 工具函数 ──────────────────────────────────────────────────────────────

def _auto_detect_label(df: pd.DataFrame) -> str:
    for candidate in ('label', 'class', 'target', 'y', 'Label', 'Class'):
        if candidate in df.columns:
            return candidate
    return ''


def _build_feature_matrix(df: pd.DataFrame, label_field: str) -> np.ndarray:
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    feature_df = df.drop(columns=[label_field])
    result_cols = []

    for col in feature_df.columns:
        if feature_df[col].dtype == object:
            le = LabelEncoder()
            col_enc = le.fit_transform(feature_df[col].astype(str).fillna('__nan__'))
            result_cols.append(col_enc.reshape(-1, 1))
        else:
            col_vals = feature_df[col].fillna(0).values.reshape(-1, 1)
            result_cols.append(col_vals)

    if not result_cols:
        raise ValueError('数据集没有可用特征列')

    X = np.hstack(result_cols).astype(float)
    X = StandardScaler().fit_transform(X)
    return X


def _fallback_detection(y, pred_probs, le, df, threshold: float = 0.3) -> list[dict]:
    """当 cleanlab 不可用时，用 self-confidence 做简单标签质量检测。"""
    self_confidence = pred_probs[np.arange(len(y)), y]
    low_conf_mask = self_confidence < threshold
    indices = np.where(low_conf_mask)[0]

    samples = []
    for idx in indices:
        original_label = le.classes_[y[idx]]
        predicted_label = le.classes_[np.argmax(pred_probs[idx])]
        lq_score = float(self_confidence[idx])
        samples.append({
            'sample_id': f'S-{df.index[idx]:06d}',
            'row_index': int(df.index[idx]),
            'risk_type': 'label_poison',
            'confidence': round(1.0 - lq_score, 4),
            'suggestion': 'relabel',
            'recommended_label': str(predicted_label),
            'detail': {
                'original_label': str(original_label),
                'predicted_label': str(predicted_label),
                'label_quality': round(lq_score, 4),
            },
        })

    return sorted(samples, key=lambda x: x['confidence'], reverse=True)


def _calc_risk_level(suspicious: int, total: int) -> str:
    if total == 0:
        return 'unknown'
    ratio = suspicious / total
    if ratio >= 0.15:
        return 'high'
    if ratio >= 0.05:
        return 'medium'
    return 'low'
