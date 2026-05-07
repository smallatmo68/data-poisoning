import logging

from rest_framework.response import Response
from rest_framework.views import exception_handler

logger = logging.getLogger('dpds')


def custom_exception_handler(exc, context):
    """将 DRF 默认异常格式统一转换为 {code, msg, data} 格式。"""
    response = exception_handler(exc, context)

    if response is not None:
        code = response.status_code
        if isinstance(response.data, dict):
            detail = response.data.get('detail', str(response.data))
        else:
            detail = str(response.data)

        response.data = {
            'code': code,
            'msg': detail,
            'data': None,
        }

    return response
