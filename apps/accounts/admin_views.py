import logging

from django.contrib.auth.models import User
from django.db.models import Count, Q
from django.utils import timezone
from rest_framework import status
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.audit.models import AuditLog
from apps.detection.models import DetectionTask
from apps.dpds_datasets.models import Dataset
from apps.defense.models import CleanResult
from apps.reports.models import Report

logger = logging.getLogger('dpds.accounts')


def ok(data=None, msg='OK'):
    return Response({'code': 0, 'msg': msg, 'data': data})


def err(msg, code=4001, http_status=status.HTTP_400_BAD_REQUEST):
    return Response({'code': code, 'msg': msg, 'data': None}, status=http_status)


class AdminUserListView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        users = User.objects.select_related('profile').all().order_by('-date_joined')
        data = []
        for u in users:
            profile = getattr(u, 'profile', None)
            data.append({
                'id': u.id,
                'username': u.username,
                'email': u.email,
                'is_active': u.is_active,
                'is_staff': u.is_staff,
                'is_superuser': u.is_superuser,
                'role': profile.role if profile else 'analyst',
                'department': profile.department if profile else '',
                'date_joined': u.date_joined.isoformat(),
                'last_login': u.last_login.isoformat() if u.last_login else None,
            })
        return ok(data)


class AdminUserToggleView(APIView):
    permission_classes = [IsAdminUser]

    def post(self, request, user_id):
        try:
            target = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return err('用户不存在', http_status=status.HTTP_404_NOT_FOUND)

        if target.is_superuser:
            return err('不能禁用超级管理员')

        target.is_active = not target.is_active
        target.save(update_fields=['is_active'])
        action = '已启用' if target.is_active else '已禁用'
        return ok({'user_id': target.id, 'is_active': target.is_active}, msg=f'用户 {target.username} {action}')


class AdminDashboardView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        now = timezone.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # 总数统计
        total_users = User.objects.count()
        total_datasets = Dataset.objects.count()
        total_tasks = DetectionTask.objects.count()
        total_reports = Report.objects.count()

        # 今日统计
        today_datasets = Dataset.objects.filter(created_at__gte=today_start).count()
        today_tasks = DetectionTask.objects.filter(created_at__gte=today_start).count()

        # 任务状态分布
        task_stats = dict(
            DetectionTask.objects.values_list('status')
            .annotate(count=Count('id'))
            .values_list('status', 'count')
        )

        # 最近任务
        recent_tasks = []
        for t in DetectionTask.objects.select_related('dataset', 'created_by').order_by('-created_at')[:10]:
            recent_tasks.append({
                'task_id': str(t.id),
                'task_no': t.task_no,
                'dataset_name': t.dataset.dataset_name if t.dataset else '-',
                'status': t.status,
                'risk_score': t.risk_score,
                'created_by': t.created_by.username if t.created_by else '-',
                'created_at': t.created_at.isoformat(),
            })

        # 最近审计日志
        recent_logs = []
        for log in AuditLog.objects.select_related('user').order_by('-created_at')[:10]:
            recent_logs.append({
                'id': log.id,
                'username': log.user.username if log.user else '-',
                'action': log.action,
                'detail': log.detail,
                'created_at': log.created_at.isoformat(),
            })

        # 风险等级分布
        risk_dist = {
            'high': DetectionTask.objects.filter(risk_score__gte=0.15).count(),
            'medium': DetectionTask.objects.filter(risk_score__gte=0.05, risk_score__lt=0.15).count(),
            'low': DetectionTask.objects.filter(risk_score__lt=0.05, risk_score__isnull=False).count(),
        }

        return ok({
            'total_users': total_users,
            'total_datasets': total_datasets,
            'total_tasks': total_tasks,
            'total_reports': total_reports,
            'today_datasets': today_datasets,
            'today_tasks': today_tasks,
            'task_stats': task_stats,
            'risk_distribution': risk_dist,
            'recent_tasks': recent_tasks,
            'recent_logs': recent_logs,
        })


class AdminAuditLogView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        page = int(request.query_params.get('page', 1))
        page_size = min(int(request.query_params.get('page_size', 20)), 100)
        action_filter = request.query_params.get('action', '')
        user_filter = request.query_params.get('user', '')

        qs = AuditLog.objects.select_related('user').all()
        if action_filter:
            qs = qs.filter(action=action_filter)
        if user_filter:
            qs = qs.filter(user__username__icontains=user_filter)

        total = qs.count()
        start = (page - 1) * page_size
        logs = qs.order_by('-created_at')[start:start + page_size]

        data = []
        for log in logs:
            data.append({
                'id': log.id,
                'username': log.user.username if log.user else '-',
                'action': log.action,
                'target_type': log.target_type,
                'target_id': log.target_id,
                'detail': log.detail,
                'ip_address': log.ip_address,
                'created_at': log.created_at.isoformat(),
            })

        return ok({
            'total': total,
            'page': page,
            'page_size': page_size,
            'results': data,
        })
