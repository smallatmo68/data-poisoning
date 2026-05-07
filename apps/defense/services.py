import logging
import os
import uuid

import pandas as pd
from django.conf import settings

from apps.detection.models import DetectionResult, DetectionTask
from .models import CleanResult

logger = logging.getLogger('dpds.defense')


def apply_defense(task_id: str, strategy: dict) -> CleanResult:
    """
    执行无害化处理。
    strategy 格式：{'default': 'remove', 'relabel': {'S-000123': 'new_label'}, 'ignore': ['S-000456']}
    """
    task = DetectionTask.objects.select_related('dataset').get(id=task_id)
    dataset = task.dataset

    storage_path = dataset.storage_path or str(settings.MEDIA_ROOT / str(dataset.file))
    df = pd.read_csv(storage_path)

    # 构建 sample_id → row_index 映射
    results = DetectionResult.objects.filter(task=task)
    sample_map = {r.sample_id: r for r in results}

    default_action = strategy.get('default', 'remove')
    relabel_map = strategy.get('relabel', {})
    ignore_set = set(strategy.get('ignore', []))

    remove_indices = []
    relabel_count = 0
    ignored_count = 0

    for sample_id, result in sample_map.items():
        row_idx = _parse_row_index(sample_id)
        if row_idx is None or row_idx >= len(df):
            continue

        action = relabel_map.get(sample_id, None) and 'relabel' or \
                 ('ignore' if sample_id in ignore_set else default_action)

        if action == 'remove':
            remove_indices.append(row_idx)
        elif action == 'relabel' and relabel_map.get(sample_id):
            label_field = dataset.label_field
            if label_field and label_field in df.columns:
                df.at[row_idx, label_field] = relabel_map[sample_id]
                relabel_count += 1
        elif action == 'ignore':
            ignored_count += 1

    clean_df = df.drop(index=remove_indices).reset_index(drop=True)

    out_dir = settings.MEDIA_ROOT / 'clean'
    os.makedirs(out_dir, exist_ok=True)
    out_path = str(out_dir / f'{uuid.uuid4().hex}_clean.csv')
    clean_df.to_csv(out_path, index=False)

    return CleanResult.objects.create(
        task=task,
        dataset=dataset,
        clean_dataset_path=out_path,
        strategy_json=strategy,
        removed_count=len(remove_indices),
        relabel_count=relabel_count,
        ignored_count=ignored_count,
    )


def _parse_row_index(sample_id: str) -> int | None:
    try:
        return int(sample_id.split('-')[-1])
    except (ValueError, IndexError):
        return None
