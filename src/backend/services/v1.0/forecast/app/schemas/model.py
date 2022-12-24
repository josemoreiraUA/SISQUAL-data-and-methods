""" 
    Model schema classes.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
    authors: jd
    version: 1.0
    date:    29/11/2022
"""

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
