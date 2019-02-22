from celery import Celery
import appConfig

BACKEND = appConfig.app_backend
BROKER = appConfig.app_broker

celery_app = Celery('celery_config',include=['mytasks'], backend=BACKEND, broker=BROKER)
