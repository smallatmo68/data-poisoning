from celery import shared_task
from datasets.services import run_detection


@shared_task(bind=True)
def detection_task(self, task_id):
    run_detection(task_id)
    return {'task_id': task_id, 'status': 'completed'}
