from typing import List, Union, Optional
from pydantic import BaseModel
from datetime import datetime

class ModelBase(BaseModel):
    type: str
    model_name: str
    storage_name: str
    metrics: str
    forecast_period: int
    n_lags: Optional[int] = None
    train_params: str

class ModelCreate(ModelBase):
    pass

class Model(ModelBase):
    id: int
    time_trained: datetime
    client_pkey: int

    class Config:
        orm_mode = True

class ClientParametersIn(BaseModel):
    culture: str
    is_active: Optional[bool] = True

class ClientBase(BaseModel):
    id: str
    culture: str
    is_active: bool = True

class ClientCreate(ClientBase):
    pass

class Client(ClientBase):
    client_pkey: int

    class Config:
        orm_mode = True

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
