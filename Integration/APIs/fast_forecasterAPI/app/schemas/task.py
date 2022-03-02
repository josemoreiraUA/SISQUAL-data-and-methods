from pydantic import BaseModel


class Task(BaseModel):
    """ Celery task representation """

    task_id: str
    status: str
