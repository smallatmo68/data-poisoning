import logging

from celery import shared_task

from .services import run_preprocess

logger = logging.getLogger('dpds.preprocessing')


@shared_task(bind=True, name='preprocessing.run_preprocess_task')
def run_preprocess_task(self, preprocess_id: str):
    logger.info('开始预处理任务 %s', preprocess_id)
    try:
        run_preprocess(preprocess_id)
        return {'preprocess_id': preprocess_id, 'status': 'success'}
    except Exception as e:
        logger.exception('预处理 Celery 任务失败: %s', e)
        raise
