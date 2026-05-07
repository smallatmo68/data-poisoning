import mimetypes
import os

import pandas as pd
from django.conf import settings
from django.http import FileResponse, Http404
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.detection.models import DetectionResult, DetectionTask
from .models import CleanResult, DefenseSampleAction
from .services import apply_defense


def ok(data=None, msg='OK'):
    return Response({'code': 0, 'msg': msg, 'data': data})


def err(msg, code=4001, http_status=status.HTTP_400_BAD_REQUEST):
    return Response({'code': code, 'msg': msg, 'data': None}, status=http_status)


class ApplyDefenseView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, task_id):
        try:
            task = DetectionTask.objects.get(id=task_id)
        except DetectionTask.DoesNotExist:
            return err('检测任务不存在', http_status=status.HTTP_404_NOT_FOUND)

        if task.status != DetectionTask.STATUS_SUCCESS:
            return err('检测任务尚未完成，无法执行无害化')

        strategy = request.data.get('strategy', {'default': 'remove'})
        clean_result = apply_defense(str(task_id), strategy)

        return ok({
            'clean_result_id': str(clean_result.id),
            'removed_count': clean_result.removed_count,
            'relabel_count': clean_result.relabel_count,
            'ignored_count': clean_result.ignored_count,
            'output_path': clean_result.clean_dataset_path,
        }, msg='无害化处理完成')


class SampleActionView(APIView):
    """对单个可疑样本执行复查操作。"""
    permission_classes = [IsAuthenticated]

    def post(self, request, result_id):
        try:
            det_result = DetectionResult.objects.select_related('task__dataset').get(id=result_id)
        except DetectionResult.DoesNotExist:
            return err('检测结果不存在', http_status=status.HTTP_404_NOT_FOUND)

        action = request.data.get('action')
        if action not in dict(DefenseSampleAction.ACTION_CHOICES):
            return err(f'无效操作: {action}，可选: {", ".join(dict(DefenseSampleAction.ACTION_CHOICES).keys())}')

        corrected_label = request.data.get('corrected_label', '')
        reason = request.data.get('reason', '')

        sample_action = DefenseSampleAction.objects.create(
            detection_result=det_result,
            action=action,
            original_label=det_result.risk_type,
            corrected_label=corrected_label,
            operator=request.user,
            reason=reason,
        )

        return ok({
            'action_id': str(sample_action.id),
            'sample_id': det_result.sample_id,
            'action': action,
        }, msg='操作已记录')


class BatchSampleActionView(APIView):
    """批量执行样本复查操作。"""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        actions = request.data.get('actions', [])
        if not actions:
            return err('未提供操作列表')

        created = []
        errors = []
        for item in actions:
            result_id = item.get('result_id')
            action = item.get('action')
            if action not in dict(DefenseSampleAction.ACTION_CHOICES):
                errors.append({'result_id': result_id, 'error': f'无效操作: {action}'})
                continue

            try:
                det_result = DetectionResult.objects.get(id=result_id)
            except DetectionResult.DoesNotExist:
                errors.append({'result_id': result_id, 'error': '检测结果不存在'})
                continue

            sample_action = DefenseSampleAction.objects.create(
                detection_result=det_result,
                action=action,
                original_label=det_result.risk_type,
                corrected_label=item.get('corrected_label', ''),
                operator=request.user,
                reason=item.get('reason', ''),
            )
            created.append({
                'action_id': str(sample_action.id),
                'sample_id': det_result.sample_id,
                'action': action,
            })

        return ok({
            'created': created,
            'errors': errors,
            'success_count': len(created),
            'error_count': len(errors),
        }, msg=f'批量操作完成: 成功 {len(created)}，失败 {len(errors)}')


class SampleDetailView(APIView):
    """获取可疑样本详情（原始数据 + 处理后数据）。"""
    permission_classes = [IsAuthenticated]

    def get(self, request, result_id):
        try:
            det_result = DetectionResult.objects.select_related('task__dataset').get(id=result_id)
        except DetectionResult.DoesNotExist:
            return err('检测结果不存在', http_status=status.HTTP_404_NOT_FOUND)

        task = det_result.task
        dataset = task.dataset
        sample_data = {}

        # 从原始数据集读取该行数据
        row_idx = _parse_row_index(det_result.sample_id)
        if row_idx is not None:
            storage_path = dataset.storage_path or str(settings.MEDIA_ROOT / str(dataset.file))
            if storage_path and os.path.exists(storage_path):
                try:
                    df = pd.read_csv(storage_path)
                    if row_idx < len(df):
                        row = df.iloc[row_idx]
                        sample_data['raw'] = {col: _safe_value(val) for col, val in row.items()}
                except Exception:
                    pass

            # 从预处理数据集读取
            if dataset.processed_path and os.path.exists(dataset.processed_path):
                try:
                    df_proc = pd.read_csv(dataset.processed_path)
                    if row_idx < len(df_proc):
                        row_proc = df_proc.iloc[row_idx]
                        sample_data['processed'] = {col: _safe_value(val) for col, val in row_proc.items()}
                except Exception:
                    pass

        # 该样本的所有检测结果
        all_results = DetectionResult.objects.filter(
            task=task, sample_id=det_result.sample_id
        ).values('id', 'detector_name', 'risk_type', 'confidence', 'suggestion', 'reason', 'triggered_features')

        # 该样本的操作历史
        action_history = DefenseSampleAction.objects.filter(
            detection_result__task=task,
            detection_result__sample_id=det_result.sample_id,
        ).values('action', 'corrected_label', 'reason', 'created_at', 'operator__username')

        return ok({
            'sample_id': det_result.sample_id,
            'row_index': row_idx,
            'data': sample_data,
            'detections': list(all_results),
            'action_history': list(action_history),
        })


def _parse_row_index(sample_id: str) -> int | None:
    try:
        return int(sample_id.split('-')[-1])
    except (ValueError, IndexError):
        return None


def _safe_value(val):
    """确保值可 JSON 序列化。"""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float, str, bool)):
        return val
    return str(val)


class DownloadCleanDatasetView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, clean_result_id):
        try:
            clean_result = CleanResult.objects.get(id=clean_result_id)
        except CleanResult.DoesNotExist:
            return err('净化结果不存在', http_status=status.HTTP_404_NOT_FOUND)

        path = clean_result.clean_dataset_path
        if not path or not os.path.exists(path):
            raise Http404('净化数据集文件不存在')

        response = FileResponse(
            open(path, 'rb'),
            content_type='text/csv',
            as_attachment=True,
            filename=os.path.basename(path),
        )
        return response
