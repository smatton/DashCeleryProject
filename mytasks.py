from celery_config import celery_app

@celery_app.task(name='mytasks.hello')
def hello(a, b):
    return a + b
