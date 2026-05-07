import logging
import threading

from django.conf import settings
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet

from .models import Dataset
from .serializers import DatasetSerializer, DatasetUploadSerializer
from .services import compute_md5, detect_label_text_fields, get_dataset_preview, parse_csv_meta

logger = logging.getLogger('dpds.datasets')


def ok(data=None, msg='OK'):
    return Response({'code': 0, 'msg': msg, 'data': data})


def err(msg, code=4001, http_status=status.HTTP_400_BAD_REQUEST):
    return Response({'code': code, 'msg': msg, 'data': None}, status=http_status)


def _auto_preprocess(dataset_id: str):
    """上传后自动执行默认预处理（去重 + 删除缺失值 + 标准化 + 类别编码）。"""
    from apps.preprocessing.models import PreprocessResult
    from apps.preprocessing.services import run_preprocess

    try:
        dataset = Dataset.objects.get(id=dataset_id)
    except Dataset.DoesNotExist:
        return

    preprocess_result = PreprocessResult.objects.create(
        dataset=dataset,
        params_json={
            'dedup': True,
            'missing_strategy': 'drop',
            'normalize': True,
            'encode_categorical': True,
        },
        status=PreprocessResult.STATUS_PENDING,
    )
    logger.info('自动预处理已创建: dataset=%s, preprocess=%s', dataset_id, preprocess_result.id)

    try:
        run_preprocess(str(preprocess_result.id))
        preprocess_result.refresh_from_db()
        if preprocess_result.status == PreprocessResult.STATUS_SUCCESS:
            out_path = preprocess_result.output_path or preprocess_result.summary_json.get('output_path', '')
            if out_path:
                dataset.processed_path = out_path
                dataset.save(update_fields=['processed_path'])
            logger.info('自动预处理完成: dataset=%s', dataset_id)
        else:
            logger.warning('自动预处理未成功: dataset=%s, status=%s', dataset_id, preprocess_result.status)
    except Exception as e:
        logger.exception('自动预处理异常: dataset=%s, error=%s', dataset_id, e)


class DatasetViewSet(ModelViewSet):
    queryset = Dataset.objects.all().order_by('-created_at')
    serializer_class = DatasetSerializer
    permission_classes = [IsAuthenticated]

    def list(self, request, *args, **kwargs):
        qs = self.get_queryset().filter(owner=request.user)
        page = self.paginate_queryset(qs)
        if page is not None:
            data = DatasetSerializer(page, many=True).data
            resp = self.get_paginated_response(data)
            resp.data = {'code': 0, 'msg': 'OK', 'data': resp.data}
            return resp
        return ok(DatasetSerializer(qs, many=True).data)

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        return ok(DatasetSerializer(instance).data)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance.owner != request.user and not request.user.is_staff:
            return err('无权删除他人数据集', http_status=status.HTTP_403_FORBIDDEN)
        instance.delete()
        return ok(msg='数据集已删除')


class DatasetUploadView(APIView):
    parser_classes = [MultiPartParser]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = DatasetUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return err(str(serializer.errors))

        file = serializer.validated_data['file']
        max_size = getattr(settings, 'MAX_UPLOAD_SIZE', 2 * 1024 * 1024 * 1024)
        if file.size > max_size:
            return err(f'文件过大，最大允许 {max_size // (1024 ** 2)} MB')

        md5 = compute_md5(file)
        file.seek(0)

        dup = Dataset.objects.filter(file_md5=md5, owner=request.user).first()
        if dup:
            return ok({'dataset_id': str(dup.id), 'duplicate': True, 'msg': '检测到重复文件，已返回已有数据集'})

        dataset = Dataset.objects.create(
            dataset_name=serializer.validated_data.get('dataset_name') or file.name,
            dataset_type=serializer.validated_data['dataset_type'],
            file=file,
            file_md5=md5,
            file_size=file.size,
            label_field=serializer.validated_data.get('label_field', ''),
            text_field=serializer.validated_data.get('text_field', ''),
            status=Dataset.STATUS_PARSING,
            owner=request.user,
        )

        file_path = str(settings.MEDIA_ROOT / str(dataset.file))
        dataset.storage_path = file_path

        if dataset.dataset_type in (Dataset.TYPE_CSV, Dataset.TYPE_TEXT):
            meta = parse_csv_meta(file_path)
            dataset.column_meta = meta.get('columns', {})
            dataset.sample_count = meta.get('sample_count', 0)
            if not dataset.label_field or not dataset.text_field:
                lf, tf = detect_label_text_fields(meta)
                dataset.label_field = dataset.label_field or lf
                dataset.text_field = dataset.text_field or tf

        dataset.status = Dataset.STATUS_READY
        dataset.save()

        # 自动预处理（后台线程，不阻塞响应）
        if dataset.dataset_type in (Dataset.TYPE_CSV, Dataset.TYPE_TEXT):
            threading.Thread(
                target=_auto_preprocess,
                args=(str(dataset.id),),
                daemon=True,
            ).start()

        return ok({'dataset_id': str(dataset.id)}, msg='上传成功')


class DatasetPreviewView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, dataset_id):
        try:
            dataset = Dataset.objects.get(id=dataset_id)
        except Dataset.DoesNotExist:
            return err('数据集不存在', http_status=status.HTTP_404_NOT_FOUND)

        page = int(request.query_params.get('page', 1))
        page_size = min(int(request.query_params.get('page_size', 20)), 100)
        return ok(get_dataset_preview(dataset, page, page_size))
