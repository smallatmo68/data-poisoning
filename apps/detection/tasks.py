import logging
from celery import shared_task

from .services import run_detection

logger = logging.getLogger('dpds.detection')


@shared_task(bind=True, name='detection.run_detection_task')
def run_detection_task(self, task_id: str):
    logger.info('开始检测任务 %s', task_id)
    try:
        run_detection(task_id)
        return {'task_id': task_id, 'status': 'success'}
    except Exception as e:
        logger.exception('检测 Celery 任务失败: %s', e)
        raise
