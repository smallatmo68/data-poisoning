import os
import hashlib
import pandas as pd
from django.conf import settings
from django.utils import timezone
from django.db.models import Q, Count
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import (
    DatasetFile, DataColumn, Dataset, UploadRecord, DataLineage,
    DetectionMethod, DetectionTask, PoisonedRecord
)
from .serializers import (
    DatasetFileSerializer, DataColumnSerializer, DatasetSerializer,
    DatasetDetailSerializer, UploadRecordSerializer, DataLineageSerializer,
    DetectionMethodSerializer, DetectionTaskSerializer, PoisonedRecordSerializer,
    DatasetUploadSerializer, DatasetImportBuiltinSerializer
)


class DatasetFileViewSet(viewsets.ModelViewSet):
    queryset = DatasetFile.objects.all()
    serializer_class = DatasetFileSerializer


class DataColumnViewSet(viewsets.ModelViewSet):
    queryset = DataColumn.objects.all()
    serializer_class = DataColumnSerializer

    def get_queryset(self):
        queryset = DataColumn.objects.all()
        dataset_id = self.request.query_params.get('dataset')
        column_type = self.request.query_params.get('column_type')
        quality_status = self.request.query_params.get('quality_status')

        if dataset_id:
            queryset = queryset.filter(dataset_id=dataset_id)
        if column_type:
            queryset = queryset.filter(column_type=column_type)
        if quality_status:
            queryset = queryset.filter(quality_status=quality_status)
        return queryset


class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return DatasetDetailSerializer
        return DatasetSerializer

    def get_queryset(self):
        queryset = Dataset.objects.all().select_related('file', 'uploaded_by')
        search = self.request.query_params.get('search')
        source_type = self.request.query_params.get('source_type')
        status_filter = self.request.query_params.get('status')
        poisoning_type = self.request.query_params.get('poisoning_type')
        domain = self.request.query_params.get('domain')
        tags = self.request.query_params.get('tags')

        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) | Q(description__icontains=search)
            )
        if source_type:
            queryset = queryset.filter(source_type=source_type)
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        if poisoning_type:
            queryset = queryset.filter(poisoning_type=poisoning_type)
        if domain:
            queryset = queryset.filter(domain=domain)
        if tags:
            queryset = queryset.filter(tags__contains=[tags])

        return queryset.annotate(
            column_count=Count('columns'),
            lineage_count=Count('lineage_records'),
            detection_count=Count('detection_tasks')
        )

    def perform_destroy(self, instance):
        if instance.file and instance.file.file_path:
            try:
                if os.path.exists(instance.file.file_path):
                    os.remove(instance.file.file_path)
            except Exception:
                pass
        instance.delete()

    @action(detail=True, methods=['get'])
    def columns(self, request, pk=None):
        dataset = self.get_object()
        columns = dataset.columns.all()
        serializer = DataColumnSerializer(columns, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def lineages(self, request, pk=None):
        dataset = self.get_object()
        lineages = dataset.lineage_records.all()
        serializer = DataLineageSerializer(lineages, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def detections(self, request, pk=None):
        dataset = self.get_object()
        tasks = dataset.detection_tasks.all()
        serializer = DetectionTaskSerializer(tasks, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def preview(self, request, pk=None):
        dataset = self.get_object()
        
        if not dataset.file or not dataset.file.file_path:
            return Response({'error': '数据文件不存在'}, status=status.HTTP_404_NOT_FOUND)
        
        if not os.path.exists(dataset.file.file_path):
            return Response({'error': '数据文件路径无效'}, status=status.HTTP_404_NOT_FOUND)
        
        try:
            page = int(request.query_params.get('page', 1))
            page_size = int(request.query_params.get('page_size', 50))
            page_size = min(page_size, 200)
        except ValueError:
            return Response({'error': '无效的分页参数'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            df = pd.read_csv(dataset.file.file_path)
            total_rows = len(df)
            total_pages = (total_rows + page_size - 1) // page_size
            
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_data = df.iloc[start_idx:end_idx]
            
            columns = list(df.columns)
            data = []
            for idx, row in page_data.iterrows():
                row_dict = {'_index': int(idx)}
                for col in columns:
                    value = row[col]
                    if pd.isna(value):
                        row_dict[col] = None
                    elif isinstance(value, (int, float)):
                        row_dict[col] = value
                    else:
                        row_dict[col] = str(value)
                data.append(row_dict)
            
            return Response({
                'dataset_id': dataset.id,
                'dataset_name': dataset.name,
                'columns': columns,
                'total_rows': total_rows,
                'total_pages': total_pages,
                'current_page': page,
                'page_size': page_size,
                'data': data
            })
            
        except Exception as e:
            return Response({'error': f'读取数据失败: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['get'])
    def preview_range(self, request, pk=None):
        dataset = self.get_object()
        
        if not dataset.file or not dataset.file.file_path:
            return Response({'error': '数据文件不存在'}, status=status.HTTP_404_NOT_FOUND)
        
        if not os.path.exists(dataset.file.file_path):
            return Response({'error': '数据文件路径无效'}, status=status.HTTP_404_NOT_FOUND)
        
        try:
            start_row = int(request.query_params.get('start', 0))
            end_row = int(request.query_params.get('end', 100))
            end_row = min(end_row, start_row + 200)
        except ValueError:
            return Response({'error': '无效的范围参数'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            df = pd.read_csv(dataset.file.file_path)
            total_rows = len(df)
            
            if start_row < 0 or start_row >= total_rows:
                return Response({'error': '起始行索引超出范围'}, status=status.HTTP_400_BAD_REQUEST)
            
            end_row = min(end_row, total_rows)
            page_data = df.iloc[start_row:end_row]
            
            columns = list(df.columns)
            data = []
            for idx, row in page_data.iterrows():
                row_dict = {'_index': int(idx)}
                for col in columns:
                    value = row[col]
                    if pd.isna(value):
                        row_dict[col] = None
                    elif isinstance(value, (int, float)):
                        row_dict[col] = value
                    else:
                        row_dict[col] = str(value)
                data.append(row_dict)
            
            return Response({
                'dataset_id': dataset.id,
                'dataset_name': dataset.name,
                'columns': columns,
                'total_rows': total_rows,
                'start_row': start_row,
                'end_row': end_row,
                'data': data
            })
            
        except Exception as e:
            return Response({'error': f'读取数据失败: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['get'])
    def poisoned_records(self, request, pk=None):
        dataset = self.get_object()
        records = dataset.poisoned_records.all()
        serializer = PoisonedRecordSerializer(records, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def statistics(self, request):
        total_datasets = Dataset.objects.count()
        total_rows = sum(ds.total_rows for ds in Dataset.objects.all())
        total_poisoned = sum(ds.poisoned_rows for ds in Dataset.objects.all())
        by_source = dict(
            Dataset.objects.values_list('source_type')
            .annotate(count=Count('id'))
            .values_list('source_type', 'count')
        )
        by_status = dict(
            Dataset.objects.values_list('status')
            .annotate(count=Count('id'))
            .values_list('status', 'count')
        )
        return Response({
            'total_datasets': total_datasets,
            'total_rows': total_rows,
            'total_poisoned': total_poisoned,
            'contamination_rate': total_poisoned / total_rows if total_rows > 0 else 0,
            'by_source': by_source,
            'by_status': by_status
        })


class UploadRecordViewSet(viewsets.ModelViewSet):
    queryset = UploadRecord.objects.all()
    serializer_class = UploadRecordSerializer

    def get_queryset(self):
        queryset = UploadRecord.objects.all().select_related('dataset', 'uploader')
        dataset_id = self.request.query_params.get('dataset')
        source = self.request.query_params.get('source')

        if dataset_id:
            queryset = queryset.filter(dataset_id=dataset_id)
        if source:
            queryset = queryset.filter(source=source)
        return queryset


class DataLineageViewSet(viewsets.ModelViewSet):
    queryset = DataLineage.objects.all()
    serializer_class = DataLineageSerializer

    def get_queryset(self):
        queryset = DataLineage.objects.all().select_related('dataset', 'operator', 'parent_lineage')
        dataset_id = self.request.query_params.get('dataset')
        operation_type = self.request.query_params.get('operation_type')

        if dataset_id:
            queryset = queryset.filter(dataset_id=dataset_id)
        if operation_type:
            queryset = queryset.filter(operation_type=operation_type)
        return queryset

    @action(detail=True, methods=['get'])
    def tree(self, request, pk=None):
        lineage = self.get_object()
        descendants = lineage.descendant_lineages.all()
        return Response(DataLineageSerializer(descendants, many=True).data)


class DetectionMethodViewSet(viewsets.ModelViewSet):
    queryset = DetectionMethod.objects.all()
    serializer_class = DetectionMethodSerializer

    @action(detail=False, methods=['get'])
    def active(self, request):
        methods = self.queryset.filter(is_active=True)
        serializer = self.get_serializer(methods, many=True)
        return Response(serializer.data)


class DetectionTaskViewSet(viewsets.ModelViewSet):
    queryset = DetectionTask.objects.all()
    serializer_class = DetectionTaskSerializer

    def get_queryset(self):
        queryset = DetectionTask.objects.all().select_related('dataset', 'method', 'created_by')
        dataset_id = self.request.query_params.get('dataset')
        method_id = self.request.query_params.get('method')
        status_filter = self.request.query_params.get('status')

        if dataset_id:
            queryset = queryset.filter(dataset_id=dataset_id)
        if method_id:
            queryset = queryset.filter(method_id=method_id)
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        return queryset

    @action(detail=True, methods=['post'])
    def start(self, request, pk=None):
        task = self.get_object()
        if task.status != 'queued':
            return Response({'error': '任务状态不允许启动'}, status=status.HTTP_400_BAD_REQUEST)

        task.status = 'running'
        task.started_at = timezone.now()
        task.save()

        try:
            from datasets.tasks import detection_task
            detection_task.delay(task.id)
        except Exception:
            from datasets.services import run_detection
            from threading import Thread
            Thread(target=run_detection, args=(task.id,), daemon=True).start()

        return Response(DetectionTaskSerializer(task).data)

    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        task = self.get_object()
        if task.status not in ['queued', 'running']:
            return Response({'error': '任务状态不允许取消'}, status=status.HTTP_400_BAD_REQUEST)

        task.status = 'cancelled'
        task.completed_at = timezone.now()
        task.save()

        return Response(DetectionTaskSerializer(task).data)


class PoisonedRecordViewSet(viewsets.ModelViewSet):
    queryset = PoisonedRecord.objects.all()
    serializer_class = PoisonedRecordSerializer

    def get_queryset(self):
        queryset = PoisonedRecord.objects.all().select_related('task', 'dataset', 'verified_by')
        dataset_id = self.request.query_params.get('dataset')
        task_id = self.request.query_params.get('task')
        severity = self.request.query_params.get('severity')
        verification_status = self.request.query_params.get('verification_status')

        if dataset_id:
            queryset = queryset.filter(dataset_id=dataset_id)
        if task_id:
            queryset = queryset.filter(task_id=task_id)
        if severity:
            queryset = queryset.filter(severity=severity)
        if verification_status:
            queryset = queryset.filter(verification_status=verification_status)
        return queryset

    @action(detail=True, methods=['post'])
    def verify(self, request, pk=None):
        record = self.get_object()
        new_status = request.data.get('verification_status')
        notes = request.data.get('notes', '')

        if new_status not in ['verified', 'false_positive', 'confirmed', 'repaired', 'ignored']:
            return Response({'error': '无效的验证状态'}, status=status.HTTP_400_BAD_REQUEST)

        record.verification_status = new_status
        record.verified_by = request.user if request.user.is_authenticated else None
        record.verified_at = timezone.now()
        record.verification_notes = notes
        record.save()

        return Response(PoisonedRecordSerializer(record).data)


class DetectionReportView(APIView):
    def get(self, request, task_id):
        from datasets.services import generate_detection_report
        try:
            report = generate_detection_report(task_id)
            return Response(report)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_404_NOT_FOUND)


class DatasetUploadView(APIView):
    def calculate_checksum(self, file_path):
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()

    def parse_dataset(self, df, file_obj):
        columns_info = []
        target_column = None
        text_column = None

        for idx, col in enumerate(df.columns):
            col_data = df[col]
            col_type = 'text'
            is_target = False
            is_text = False

            if col.lower() in ['label', 'class', 'target']:
                col_type = 'label'
                is_target = True
                target_column = col
            elif col.lower() in ['text', 'content', 'sentence', 'input']:
                col_type = 'text'
                is_text = True
                text_column = col
            elif pd.api.types.is_numeric_dtype(col_data):
                col_type = 'numeric'

            null_count = int(col_data.isnull().sum())
            unique_vals = col_data.dropna().unique()
            label_dist = {}

            if col_type == 'label':
                label_dist = col_data.value_counts().head(20).to_dict()

            columns_info.append({
                'name': col,
                'ordinal': idx,
                'column_type': col_type,
                'is_target': is_target,
                'is_text': is_text,
                'null_count': null_count,
                'null_rate': null_count / len(df) if len(df) > 0 else 0,
                'unique_count': len(unique_vals),
                'unique_rate': len(unique_vals) / len(df) if len(df) > 0 else 0,
                'avg_length': col_data.astype(str).str.len().mean() if is_text else 0,
                'min_value': str(col_data.min()) if len(unique_vals) > 0 else '',
                'max_value': str(col_data.max()) if len(unique_vals) > 0 else '',
                'label_distribution': label_dist
            })

        return {
            'columns': columns_info,
            'target_column': target_column,
            'text_column': text_column,
            'row_count': len(df),
            'column_count': len(df.columns)
        }

    def post(self, request):
        serializer = DatasetUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        uploaded_file = serializer.validated_data['file']
        name = serializer.validated_data['name']
        description = serializer.validated_data.get('description', '')
        domain = serializer.validated_data.get('domain', '')
        task_type = serializer.validated_data.get('task_type', '')
        tags = serializer.validated_data.get('tags', [])
        notes = serializer.validated_data.get('notes', '')

        upload_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
        os.makedirs(upload_dir, exist_ok=True)

        original_filename = uploaded_file.name
        file_path = os.path.join(upload_dir, original_filename)

        with open(file_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        file_size = os.path.getsize(file_path)
        checksum = self.calculate_checksum(file_path)

        existing_file = DatasetFile.objects.filter(checksum=checksum).first()
        if existing_file:
            os.remove(file_path)
            dataset_obj = Dataset.objects.filter(file=existing_file).first()
            if dataset_obj:
                return Response({
                    'message': '数据集已存在',
                    'dataset_id': dataset_obj.id,
                    'dataset_name': dataset_obj.name
                }, status=status.HTTP_200_OK)

        df = pd.read_csv(file_path)
        parse_result = self.parse_dataset(df, uploaded_file)

        file_obj = DatasetFile.objects.create(
            filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            checksum=checksum,
            data_format='csv',
            row_count=parse_result['row_count'],
            column_count=parse_result['column_count']
        )

        dataset_obj = Dataset.objects.create(
            name=name,
            file=file_obj,
            source_type='upload',
            status='ready',
            description=description,
            domain=domain,
            task_type=task_type,
            total_rows=parse_result['row_count'],
            tags=tags,
            metadata={'target_column': parse_result['target_column'], 'text_column': parse_result['text_column']},
            uploaded_by=request.user if request.user.is_authenticated else None
        )

        for col_info in parse_result['columns']:
            DataColumn.objects.create(dataset=dataset_obj, **col_info)

        UploadRecord.objects.create(
            dataset=dataset_obj,
            source='web',
            uploader=request.user if request.user.is_authenticated else None,
            original_filename=original_filename,
            file_size=file_size,
            ip_address=self.get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            notes=notes
        )

        DataLineage.objects.create(
            dataset=dataset_obj,
            operation_type='import',
            operation_name=f'导入数据集: {name}',
            description=f'从文件 {original_filename} 导入数据集',
            input_datasets=[],
            output_datasets=[{'id': dataset_obj.id, 'name': dataset_obj.name}],
            operator=request.user if request.user.is_authenticated else None,
            rows_affected=parse_result['row_count'],
            result_summary={
                'row_count': parse_result['row_count'],
                'column_count': parse_result['column_count']
            }
        )

        return Response(DatasetSerializer(dataset_obj).data, status=status.HTTP_201_CREATED)

    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


class BuiltinDatasetImportView(APIView):
    def get_builtin_structure(self):
        base_path = settings.DATA_SOURCE_PATH
        structure = {}

        if not os.path.exists(base_path):
            return structure

        category_map = {
            'CBADatasets': {'name': 'CBA数据集', 'domain': '情感分析'},
            'VPIDatasets': {'name': 'VPI数据集', 'domain': '文本分类'},
            'backdoor_datas': {'name': '后门投毒数据', 'domain': '后门攻击'},
            'badchain_datasets': {'name': 'BadChain数据集', 'domain': '_chain攻击'},
            'badedit_datasets': {'name': 'BadEdit数据集', 'domain': '编辑投毒'},
            'datasets_experiment': {'name': '实验数据集', 'domain': '综合实验'},
            'multi_level_trigger': {'name': '多级触发器数据', 'domain': '多级触发'},
            'sleepagent_dataset': {'name': 'SleepAgent数据', 'domain': 'Agent投毒'}
        }

        for category_key, category_info in category_map.items():
            category_path = os.path.join(base_path, category_key)
            if os.path.exists(category_path):
                structure[category_key] = {
                    'name': category_info['name'],
                    'domain': category_info['domain'],
                    'files': []
                }
                for filename in sorted(os.listdir(category_path)):
                    if filename.endswith('.csv'):
                        file_path = os.path.join(category_path, filename)
                        file_size = os.path.getsize(file_path)
                        structure[category_key]['files'].append({
                            'name': filename,
                            'path': os.path.join(category_key, filename),
                            'size': file_size
                        })

        return structure

    def get(self, request):
        structure = self.get_builtin_structure()
        return Response(structure)

    def calculate_checksum(self, file_path):
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()

    def parse_dataset(self, df):
        columns_info = []
        target_column = None
        text_column = None

        for idx, col in enumerate(df.columns):
            col_data = df[col]
            col_type = 'text'
            is_target = False
            is_text = False

            if col.lower() in ['label', 'class', 'target']:
                col_type = 'label'
                is_target = True
                target_column = col
            elif col.lower() in ['text', 'content', 'sentence', 'input']:
                col_type = 'text'
                is_text = True
                text_column = col
            elif pd.api.types.is_numeric_dtype(col_data):
                col_type = 'numeric'

            null_count = int(col_data.isnull().sum())
            unique_vals = col_data.dropna().unique()
            label_dist = {}

            if col_type == 'label':
                label_dist = col_data.value_counts().head(20).to_dict()

            columns_info.append({
                'name': col,
                'ordinal': idx,
                'column_type': col_type,
                'is_target': is_target,
                'is_text': is_text,
                'null_count': null_count,
                'null_rate': null_count / len(df) if len(df) > 0 else 0,
                'unique_count': len(unique_vals),
                'unique_rate': len(unique_vals) / len(df) if len(df) > 0 else 0,
                'avg_length': col_data.astype(str).str.len().mean() if is_text else 0,
                'min_value': str(col_data.min()) if len(unique_vals) > 0 else '',
                'max_value': str(col_data.max()) if len(unique_vals) > 0 else '',
                'label_distribution': label_dist
            })

        return {
            'columns': columns_info,
            'target_column': target_column,
            'text_column': text_column,
            'row_count': len(df),
            'column_count': len(df.columns)
        }

    def post(self, request):
        import_all = request.data.get('import_all', False)
        categories = request.data.get('categories', [])
        structure = self.get_builtin_structure()

        if not import_all and not categories:
            return Response({'error': '请指定要导入的分类或设置import_all为true'}, status=status.HTTP_400_BAD_REQUEST)

        target_categories = structure.keys() if import_all else categories
        imported = []
        skipped = []
        errors = []

        for category_key in target_categories:
            if category_key not in structure:
                errors.append(f"分类不存在: {category_key}")
                continue

            category_info = structure[category_key]
            domain = category_info.get('domain', '')

            for file_info in category_info['files']:
                full_path = os.path.join(settings.DATA_SOURCE_PATH, file_info['path'])

                try:
                    if not os.path.exists(full_path):
                        errors.append(f"文件不存在: {file_info['path']}")
                        continue

                    checksum = self.calculate_checksum(full_path)
                    existing_file = DatasetFile.objects.filter(checksum=checksum).first()

                    if existing_file:
                        dataset_obj = Dataset.objects.filter(file=existing_file).first()
                        if dataset_obj:
                            skipped.append({'name': dataset_obj.name, 'reason': '已存在'})
                            continue

                    df = pd.read_csv(full_path)
                    parse_result = self.parse_dataset(df)

                    file_obj = DatasetFile.objects.create(
                        filename=file_info['name'],
                        file_path=full_path,
                        file_size=file_info['size'],
                        checksum=checksum,
                        data_format='csv',
                        row_count=parse_result['row_count'],
                        column_count=parse_result['column_count']
                    )

                    poisoning_type = 'none'
                    if 'backdoor' in category_key.lower():
                        poisoning_type = 'backdoor'
                    elif 'bad' in category_key.lower():
                        poisoning_type = 'mixed'

                    dataset_obj = Dataset.objects.create(
                        name=file_info['name'].replace('.csv', ''),
                        file=file_obj,
                        source_type='builtin',
                        status='ready',
                        poisoning_type=poisoning_type,
                        domain=domain,
                        task_type='文本分类',
                        total_rows=parse_result['row_count'],
                        metadata={
                            'target_column': parse_result['target_column'],
                            'text_column': parse_result['text_column'],
                            'category': category_key
                        }
                    )

                    for col_info in parse_result['columns']:
                        DataColumn.objects.create(dataset=dataset_obj, **col_info)

                    DataLineage.objects.create(
                        dataset=dataset_obj,
                        operation_type='import',
                        operation_name=f'内置数据导入: {dataset_obj.name}',
                        description=f'从 {category_info["name"]} 导入内置数据集',
                        input_datasets=[],
                        output_datasets=[{'id': dataset_obj.id, 'name': dataset_obj.name}],
                        operator=None,
                        rows_affected=parse_result['row_count'],
                        result_summary={
                            'row_count': parse_result['row_count'],
                            'column_count': parse_result['column_count'],
                            'category': category_key
                        }
                    )

                    imported.append({
                        'id': dataset_obj.id,
                        'name': dataset_obj.name,
                        'category': category_key,
                        'row_count': parse_result['row_count']
                    })

                except Exception as e:
                    errors.append(f"导入失败 {file_info['name']}: {str(e)}")

        return Response({
            'imported_count': len(imported),
            'imported': imported,
            'skipped_count': len(skipped),
            'skipped': skipped,
            'errors': errors
        })
