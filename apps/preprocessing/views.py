import threading

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.dpds_datasets.models import Dataset
from .models import PreprocessResult
from .services import run_preprocess


def ok(data=None, msg='OK'):
    return Response({'code': 0, 'msg': msg, 'data': data})


def err(msg, code=4001, http_status=status.HTTP_400_BAD_REQUEST):
    return Response({'code': code, 'msg': msg, 'data': None}, status=http_status)


class CreatePreprocessView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, dataset_id):
        try:
            dataset = Dataset.objects.get(id=dataset_id)
        except Dataset.DoesNotExist:
            return err('数据集不存在', http_status=status.HTTP_404_NOT_FOUND)

        params = request.data.get('params', {})
        result = PreprocessResult.objects.create(dataset=dataset, params_json=params)

        # 在独立线程中执行预处理
        t = threading.Thread(target=run_preprocess, args=(str(result.id),), daemon=True)
        t.start()

        return ok({'preprocess_id': str(result.id)}, msg='预处理任务已创建')


class PreprocessDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, preprocess_id):
        try:
            result = PreprocessResult.objects.select_related('dataset').get(id=preprocess_id)
        except PreprocessResult.DoesNotExist:
            return err('预处理结果不存在', http_status=status.HTTP_404_NOT_FOUND)

        return ok({
            'preprocess_id': str(result.id),
            'dataset_id': str(result.dataset_id),
            'status': result.status,
            'summary': result.summary_json,
            'output_path': result.output_path,
            'error_message': result.error_message,
            'created_at': result.created_at.isoformat(),
        })
