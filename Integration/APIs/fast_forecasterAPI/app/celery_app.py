from celery import Celery

# import os

app_celery = Celery(
    name="celery_fast_forecaster",
    broker="redis://localhost:6379/",
    backend="redis://localhost:6379/",
)

"""
if "REDIS_URL" in os.environ:
    app_celery.conf.update(
        name="celery_fast_forecaster",
        broker_url=os.environ["REDIS_URL"],
        result_backend=os.environ["REDIS_URL"],
    )
"""
