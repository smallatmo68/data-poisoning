"""
从 third_party_algorithms 目录导入内置数据集到数据库。
已存在的数据集按文件路径做幂等跳过。
"""

import os
import shutil
import threading
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile

from apps.dpds_datasets.models import Dataset
from apps.dpds_datasets.services import parse_csv_meta, detect_label_text_fields
from apps.preprocessing.models import PreprocessResult
from apps.preprocessing.services import run_preprocess

BASE_DIR = Path(settings.BASE_DIR)
BUILTIN_DIR = BASE_DIR / 'third_party_algorithms' / 'BackdoorDetection-main' / 'BackdoorDetection-main'


def _auto_preprocess(dataset_id: str):
    """自动预处理（后台线程）"""
    try:
        from apps.preprocessing.models import PreprocessResult
        from apps.preprocessing.services import run_preprocess
        dataset = Dataset.objects.get(id=dataset_id)
        preprocess_result = PreprocessResult.objects.create(
            dataset=dataset,
            status=PreprocessResult.STATUS_PENDING,
        )
        run_preprocess(str(preprocess_result.id))
        preprocess_result.refresh_from_db()
        if preprocess_result.status == PreprocessResult.STATUS_SUCCESS:
            out_path = preprocess_result.output_path or preprocess_result.summary_json.get('output_path', '')
            if out_path and os.path.exists(out_path):
                dataset.storage_path = out_path
                dataset.status = Dataset.STATUS_READY
                dataset.save(update_fields=['storage_path', 'status'])
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning('自动预处理失败: dataset=%s, error=%s', dataset_id, e)


class Command(BaseCommand):
    help = '从 third_party_algorithms 目录导入内置数据集'

    def add_arguments(self, parser):
        parser.add_argument(
            '--limit',
            type=int,
            default=0,
            help='最多导入 N 个数据集（0=不限制）',
        )
        parser.add_argument(
            '--skip-preprocess',
            action='store_true',
            help='跳过自动预处理',
        )

    def handle(self, *args, **options):
        if not BUILTIN_DIR.exists():
            self.stdout.write(self.style.ERROR(f'内置数据集目录不存在: {BUILTIN_DIR}'))
            return

        csv_files = sorted(BUILTIN_DIR.rglob('*.csv'))
        if not csv_files:
            self.stdout.write(self.style.WARNING('未找到 CSV 文件'))
            return

        limit = options['limit']
        if limit > 0:
            csv_files = csv_files[:limit]

        self.stdout.write(f'找到 {len(csv_files)} 个 CSV 文件')

        media_dir = Path(settings.MEDIA_ROOT) / 'datasets'
        media_dir.mkdir(parents=True, exist_ok=True)

        created_count = 0
        skipped_count = 0
        error_count = 0

        for csv_path in csv_files:
            rel_path = csv_path.relative_to(BUILTIN_DIR)
            dataset_name = f'[内置] {rel_path.stem}'

            # 检查是否已导入（按文件路径）
            existing = Dataset.objects.filter(
                dataset_name=dataset_name,
                dataset_type=Dataset.TYPE_CSV,
            ).first()
            if existing:
                skipped_count += 1
                continue

            try:
                # 复制文件到 media 目标
                dest_path = media_dir / f'builtin_{rel_path.stem}_{csv_path.stat().st_size}.csv'
                if not dest_path.exists():
                    shutil.copy2(str(csv_path), str(dest_path))

                # 创建数据集记录
                file_size = csv_path.stat().st_size
                dataset = Dataset(
                    dataset_name=dataset_name,
                    dataset_type=Dataset.TYPE_CSV,
                    status=Dataset.STATUS_READY,
                    file_size=file_size,
                    storage_path=str(dest_path),
                )
                dataset.save()

                # 解析元数据
                try:
                    meta = parse_csv_meta(str(dest_path))
                    dataset.column_meta = meta.get('columns', {})
                    dataset.sample_count = meta.get('sample_count', 0)
                    lf, tf = detect_label_text_fields(meta)
                    dataset.label_field = lf
                    dataset.text_field = tf
                    dataset.save(update_fields=['column_meta', 'sample_count', 'label_field', 'text_field'])
                except Exception as e:
                    self.stdout.write(f'  [警告] 元数据解析失败 {rel_path}: {e}')

                # 自动预处理
                if not options['skip_preprocess']:
                    threading.Thread(
                        target=_auto_preprocess,
                        args=(str(dataset.id),),
                        daemon=True,
                    ).start()

                created_count += 1
                self.stdout.write(f'  [新增] {dataset_name} ({file_size} bytes)')

            except Exception as e:
                error_count += 1
                self.stdout.write(self.style.ERROR(f'  [错误] {rel_path}: {e}'))

        self.stdout.write(self.style.SUCCESS(
            f'\n完成：新增 {created_count}，跳过 {skipped_count}，错误 {error_count}，共 {len(csv_files)} 个文件'
        ))
