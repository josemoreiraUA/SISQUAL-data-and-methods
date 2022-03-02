from pydantic import BaseModel


class Prediction(BaseModel):
    """ Prediction task result """

    task_id: str
    status: str
    forecast: str
