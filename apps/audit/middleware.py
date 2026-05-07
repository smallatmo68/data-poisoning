import logging

logger = logging.getLogger('dpds.audit')


class AuditLogMiddleware:
    """记录写操作到审计日志（仅 POST/PUT/PATCH/DELETE）。"""

    WRITE_METHODS = {'POST', 'PUT', 'PATCH', 'DELETE'}
    SKIP_PATHS = {'/api/auth/token/refresh/'}

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        if (
            request.method in self.WRITE_METHODS
            and request.path not in self.SKIP_PATHS
            and hasattr(request, 'user')
            and request.user.is_authenticated
        ):
            self._log(request, response)

        return response

    @staticmethod
    def _log(request, response):
        try:
            from apps.audit.models import AuditLog

            ip = (
                request.META.get('HTTP_X_FORWARDED_FOR', '').split(',')[0].strip()
                or request.META.get('REMOTE_ADDR')
            )

            action = 'other'
            path = request.path.lower()
            if 'login' in path:
                action = 'login'
            elif 'logout' in path:
                action = 'logout'
            elif 'upload' in path or 'datasets' in path:
                action = 'upload' if request.method == 'POST' else 'delete' if request.method == 'DELETE' else 'other'
            elif 'detection' in path:
                action = 'detect'
            elif 'defense' in path:
                action = 'defense'
            elif 'report' in path or 'download' in path:
                action = 'export'
            elif 'algorithm' in path or 'config' in path:
                action = 'config'

            AuditLog.objects.create(
                user=request.user,
                action=action,
                target_type=path.split('/')[2] if len(path.split('/')) > 2 else '',
                detail=f'{request.method} {request.path} → {response.status_code}',
                ip_address=ip or None,
            )
        except Exception as e:
            logger.warning('审计日志写入失败: %s', e)
