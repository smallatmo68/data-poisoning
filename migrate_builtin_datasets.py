"""
数据集迁移脚本 - 用于初始化内置数据集到数据库
"""
import os
import sys
import django
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DataPoisoningDetection.settings')
django.setup()

from datasets.models import DatasetCategory, DataSource, Dataset, DatasetSample
from django.conf import settings

def calculate_checksum(file_path):
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()

def parse_csv(file_path, max_samples=100):
    import pandas as pd
    df = pd.read_csv(file_path, nrows=max_samples + 1)
    column_names = df.columns.tolist()
    row_count = len(df)
    column_count = len(column_names)

    text_column = ''
    label_column = ''
    for col in column_names:
        col_lower = col.lower()
        if 'text' in col_lower or 'content' in col_lower or 'sentence' in col_lower:
            text_column = col
        if 'label' in col_lower or 'class' in col_lower or 'target' in col_lower:
            label_column = col

    samples = []
    for idx, row in df.head(100).iterrows():
        samples.append({
            'row_index': int(idx),
            'data': row.to_dict()
        })

    return {
        'column_names': column_names,
        'row_count': int(row_count),
        'column_count': int(column_count),
        'text_column': text_column,
        'label_column': label_column,
        'samples': samples
    }

def migrate_builtin_datasets():
    base_path = settings.DATA_SOURCE_PATH
    print(f"数据源路径: {base_path}")

    structure = {
        'CBADatasets': 'CBA数据集',
        'VPIDatasets': 'VPI数据集',
        'backdoor_datas': '后门数据集',
        'badchain_datasets': 'BadChain数据集',
        'badedit_datasets': 'BadEdit数据集',
        'datasets_experiment': '实验数据集',
        'multi_level_trigger': '多级触发器数据集',
        'sleepagent_dataset': 'SleepAgent数据集'
    }

    source, created = DataSource.objects.get_or_create(
        name='内置数据源',
        defaults={'source_type': 'builtin', 'description': 'BackdoorDetection项目内置数据集'}
    )
    print(f"数据来源: {'创建' if created else '已存在'}")

    total_imported = 0
    total_skipped = 0

    for category_code, category_name in structure.items():
        category_path = os.path.join(base_path, category_code)

        if not os.path.exists(category_path):
            print(f"[跳过] 目录不存在: {category_path}")
            continue

        category, created = DatasetCategory.objects.get_or_create(
            code=category_code,
            defaults={'name': category_name}
        )
        print(f"\n分类: {category_name} ({category_code})")

        if os.path.isdir(category_path):
            for filename in sorted(os.listdir(category_path)):
                if not filename.endswith('.csv'):
                    continue

                full_path = os.path.join(category_path, filename)
                file_size = os.path.getsize(full_path)

                checksum = calculate_checksum(full_path)
                if Dataset.objects.filter(checksum=checksum).exists():
                    print(f"  [跳过] {filename} (已存在)")
                    total_skipped += 1
                    continue

                try:
                    parse_result = parse_csv(full_path)

                    dataset = Dataset.objects.create(
                        name=filename.replace('.csv', ''),
                        file_path=full_path,
                        original_filename=filename,
                        category=category,
                        source=source,
                        data_format='csv',
                        row_count=parse_result['row_count'],
                        column_count=parse_result['column_count'],
                        column_names=parse_result['column_names'],
                        text_column=parse_result['text_column'],
                        label_column=parse_result['label_column'],
                        status='imported',
                        file_size=file_size,
                        checksum=checksum,
                        metadata=parse_result
                    )

                    for sample_data in parse_result['samples']:
                        DatasetSample.objects.create(
                            dataset=dataset,
                            row_index=sample_data['row_index'],
                            data=sample_data['data']
                        )

                    print(f"  [导入] {filename} - {parse_result['row_count']}行")
                    total_imported += 1

                except Exception as e:
                    print(f"  [错误] {filename}: {str(e)}")

    print(f"\n导入完成: {total_imported} 个数据集, {total_skipped} 个已跳过")

if __name__ == '__main__':
    migrate_builtin_datasets()
