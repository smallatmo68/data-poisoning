import logging
import os
import uuid

import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .models import PreprocessResult

logger = logging.getLogger('dpds.preprocessing')


def run_preprocess(preprocess_id: str):
    """执行预处理任务（被 Celery task 或同步调用）。"""
    try:
        result = PreprocessResult.objects.select_related('dataset').get(id=preprocess_id)
    except PreprocessResult.DoesNotExist:
        logger.error('PreprocessResult %s 不存在', preprocess_id)
        return

    result.status = PreprocessResult.STATUS_RUNNING
    result.save(update_fields=['status'])

    try:
        dataset = result.dataset
        params = result.params_json

        if dataset.dataset_type in ('csv', 'text'):
            summary = _preprocess_csv(dataset, result, params)
        else:
            summary = {'msg': '图像数据集预处理暂不支持，跳过'}

        result.summary_json = summary
        result.status = PreprocessResult.STATUS_SUCCESS
        result.save(update_fields=['summary_json', 'status', 'output_path'])

    except Exception as e:
        logger.exception('预处理失败: %s', e)
        result.status = PreprocessResult.STATUS_FAILED
        result.error_message = str(e)
        result.save(update_fields=['status', 'error_message'])


def _preprocess_csv(dataset, result, params: dict) -> dict:
    storage_path = dataset.storage_path or str(settings.MEDIA_ROOT / str(dataset.file))
    df = pd.read_csv(storage_path)
    original_count = len(df)

    # 1. 去重
    if params.get('dedup', True):
        df = df.drop_duplicates()

    # 2. 缺失值处理
    missing_strategy = params.get('missing_strategy', 'drop')
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy == 'fill_mean':
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].mean())
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else '')

    # 3. 标准化数值列（排除 label 列）
    label_field = dataset.label_field
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != label_field
    ]
    if params.get('normalize', True) and numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 4. 类别编码（排除文本列和 label 列）
    text_field = dataset.text_field
    cat_cols = [
        c for c in df.select_dtypes(include=['object']).columns
        if c not in (label_field, text_field) and df[c].nunique() < 50
    ]
    if params.get('encode_categorical', True) and cat_cols:
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # 保存预处理结果
    out_dir = settings.MEDIA_ROOT / 'preprocessed'
    os.makedirs(out_dir, exist_ok=True)
    out_path = str(out_dir / f'{uuid.uuid4().hex}.csv')
    df.to_csv(out_path, index=False)

    result.output_path = out_path

    return {
        'original_count': original_count,
        'processed_count': len(df),
        'removed_count': original_count - len(df),
        'numeric_cols_normalized': numeric_cols,
        'categorical_cols_encoded': cat_cols,
        'output_path': out_path,
    }
