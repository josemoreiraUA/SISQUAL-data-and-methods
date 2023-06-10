""" 
    Domain classes for the application.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    10/06/2023
"""

from enum import Enum
from pydantic import BaseModel
from typing import List, Optional
from datetime import date, datetime, time, timedelta

class RoundingMethod(str, Enum):
    Raw = 'Raw'
    Floor = 'Floor'
    Ceil = 'Ceil'
    Round = 'Round'

class MessageLevel(str, Enum):
    Trace = 'Trace'
    Success = 'Success'
    Info = 'Info'
    Warning = 'Warning'
    Error = 'Error'
    Critical = 'Critical'

class TypeForecast(str, Enum):
    CustomerFlow = 'Customer Flow'
    Sizing = 'Team Size (Sizing)'
    CustomerFlowAndSizing = 'Customer Flow And Team Size (Sizing)'

class ForecastingModels(str, Enum):
    MLPRegressor = 'MLPRegressor'
    HistGradientBoostingRegressor = 'HistGradientBoostingRegressor'
    Prophet = 'Prophet'
    RandomForestRegressor = 'RandomForestRegressor'

class CustomerFlowModels(str, Enum):
    MLPRegressor = 'MLPRegressor'
    HistGradientBoostingRegressor = 'HistGradientBoostingRegressor'
    Prophet = 'Prophet'

class SizingModels(str, Enum):
    MLPRegressor = 'MLPRegressor'
    RandomForestRegressor = 'RandomForestRegressor'

class CustomerFlowAndSizingModels(str, Enum):
    MLPRegressor = 'MLPRegressor'

class TaskState(str, Enum):
    Created = 'Created'
    Pending = 'Pending'
    Executing = 'Executing'
    Finished = 'Finished'
    Error = 'Error'

class MyException(Exception):
    pass

class TrainIn(BaseModel):
    client_id: str
    model_name: str
    input_data: dict
    forecast_period: int
    n_lags: Optional[int] = 720

class TrainTaskOut(BaseModel):
    id: int
    state: str