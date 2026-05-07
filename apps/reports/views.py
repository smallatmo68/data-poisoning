import logging
import os

from django.http import FileResponse, Http404
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.detection.models import DetectionTask
from .models import Report
from .services import generate_report, _compute_analysis

logger = logging.getLogger('dpds.reports')


def ok(data=None, msg='OK'):
    return Response({'code': 0, 'msg': msg, 'data': data})


def err(msg, code=4001, http_status=status.HTTP_400_BAD_REQUEST):
    return Response({'code': code, 'msg': msg, 'data': None}, status=http_status)


class CreateReportView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        task_id = request.data.get('task_id')
        report_type = request.data.get('report_type', 'html')

        try:
            task = DetectionTask.objects.get(id=task_id)
        except DetectionTask.DoesNotExist:
            return err('检测任务不存在')

        if task.status != DetectionTask.STATUS_SUCCESS:
            return err('检测任务尚未完成，无法生成报告')

        report = generate_report(str(task_id), report_type, request.user)
        return ok({
            'report_id': str(report.id),
            'report_type': report.report_type,
        }, msg='报告已生成')


class CreateLLMReportView(APIView):
    """调用 LLM 生成深度分析报告。"""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        task_id = request.data.get('task_id')

        # 检查是否启用 LLM
        enable_llm = os.getenv('ENABLE_LLM_REPORT', 'false').lower() in ('true', '1', 'yes')
        if not enable_llm:
            return Response({
                'code': 4003,
                'msg': 'AI 报告生成功能未配置，请在 .env 中配置 ENABLE_LLM_REPORT=true 和 DEEPSEEK_API_KEY',
                'data': {'enabled': False},
            })

        try:
            task = DetectionTask.objects.get(id=task_id)
        except DetectionTask.DoesNotExist:
            return err('检测任务不存在')

        if task.status != DetectionTask.STATUS_SUCCESS:
            return err('检测任务尚未完成，无法生成报告')

        try:
            from .llm_report_service import generate_llm_report
            report = generate_llm_report(str(task_id), request.user)
        except ValueError as e:
            return err(str(e))
        except RuntimeError as e:
            logger.error('LLM 报告生成失败: %s', e)
            return err(str(e), code=5001, http_status=status.HTTP_502_BAD_GATEWAY)

        return ok({
            'enabled': True,
            'task_id': str(task.id),
            'report_id': str(report.id),
            'title': report.title,
            'summary': report.summary,
            'llm_content': report.llm_content,
            'generated_at': report.created_at.isoformat() if report.created_at else None,
            'model': os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'),
        }, msg='LLM 报告已生成')


class ReportListView(APIView):
    """获取任务关联的报告列表。"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        task_id = request.query_params.get('task_id')
        qs = Report.objects.select_related('task').all()
        if task_id:
            qs = qs.filter(task_id=task_id)
        qs = qs.order_by('-created_at')[:50]

        data = []
        for r in qs:
            data.append({
                'id': str(r.id),
                'task_id': str(r.task_id),
                'task_no': r.task.task_no if r.task else '',
                'title': r.title,
                'report_type': r.report_type,
                'summary': r.summary,
                'has_llm': bool(r.llm_content),
                'created_at': r.created_at.isoformat() if r.created_at else None,
            })
        return ok(data)


class DownloadReportView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, report_id):
        try:
            report = Report.objects.get(id=report_id)
        except Report.DoesNotExist:
            return err('报告不存在', http_status=status.HTTP_404_NOT_FOUND)

        if not report.file_path or not os.path.exists(report.file_path):
            raise Http404('报告文件不存在')

        content_type = 'text/html' if report.report_type == 'html' else 'text/markdown'
        return FileResponse(
            open(report.file_path, 'rb'),
            content_type=content_type,
            as_attachment=True,
            filename=os.path.basename(report.file_path),
        )
