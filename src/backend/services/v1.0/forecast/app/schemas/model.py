from pydantic import BaseModel
from datetime import datetime

class ModelBase(BaseModel):
    type: str
    model_name: str
    storage_name: str
    metrics: str
    forecast_period: int
    train_params: str

class ModelCreate(ModelBase):
    pass

class Model(ModelBase):
    id: int
    time_trained: datetime
    client_pkey: int

    class Config:
        orm_mode = True
