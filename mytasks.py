from celery_config import celery_app


@celery_app.task(name='mytasks.add')
def add(a, b):
    return a + b

