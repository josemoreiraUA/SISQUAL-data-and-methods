from pydantic import BaseModel
from datetime import datetime

class TTaskBase(BaseModel):
    client_pkey: int
    model_type: str
    state: str

class TTaskCreate(TTaskBase):
    pass

class TTask(TTaskBase):
    id: int
    time_created: datetime
    time_started: datetime
    time_finished: datetime

    class Config:
        orm_mode = True
