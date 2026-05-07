import hashlib
import io
import logging
import os
import zipfile

import pandas as pd
from django.conf import settings

from .models import Dataset

logger = logging.getLogger('dpds.datasets')


def compute_md5(file_obj) -> str:
    h = hashlib.md5()
    for chunk in file_obj.chunks():
        h.update(chunk)
    return h.hexdigest()


def parse_csv_meta(file_path: str) -> dict:
    """读取 CSV 并返回列元数据 + 样本数。"""
    try:
        df = pd.read_csv(file_path, nrows=5000)
        sample_count = len(pd.read_csv(file_path, usecols=[0]))
        columns = {}
        for col in df.columns:
            col_lower = col.lower()
            is_target = col_lower in ('label', 'class', 'target', 'y')
            is_text = col_lower in ('text', 'content', 'sentence', 'input', 'review', 'comment')
            dtype = str(df[col].dtype)
            columns[col] = {
                'dtype': dtype,
                'is_target': is_target,
                'is_text': is_text,
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique()),
            }
        return {'columns': columns, 'sample_count': sample_count}
    except Exception as e:
        logger.warning('解析 CSV 元数据失败: %s', e)
        return {'columns': {}, 'sample_count': 0}


def detect_label_text_fields(column_meta: dict) -> tuple[str, str]:
    """从列元数据推断 label_field 和 text_field。"""
    label_field = ''
    text_field = ''
    for col, info in column_meta.get('columns', {}).items():
        if info.get('is_target') and not label_field:
            label_field = col
        if info.get('is_text') and not text_field:
            text_field = col
    return label_field, text_field


def get_dataset_preview(dataset: Dataset, page: int = 1, page_size: int = 20) -> dict:
    """返回分页预览数据。"""
    try:
        path = dataset.storage_path or str(settings.MEDIA_ROOT / dataset.file.name)
        if dataset.dataset_type == Dataset.TYPE_CSV:
            df = pd.read_csv(path)
            total = len(df)
            start = (page - 1) * page_size
            end = start + page_size
            rows = df.iloc[start:end].to_dict(orient='records')
            return {
                'total': total,
                'page': page,
                'page_size': page_size,
                'columns': list(df.columns),
                'rows': rows,
            }
        return {'error': '该数据类型暂不支持在线预览'}
    except Exception as e:
        logger.error('数据集预览失败: %s', e)
        return {'error': str(e)}
