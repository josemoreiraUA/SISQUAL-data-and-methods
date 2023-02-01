""" 
    Domain classes for the application.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022
"""

from enum import Enum
from pydantic import BaseModel
from typing import List, Optional
from datetime import date, datetime, time, timedelta

class MessageLevel(str, Enum):
    Trace = 'Trace'
    Success = 'Success'
    Info = 'Info'
    Warning = 'Warning'
    Error = 'Error'
    Critical = 'Critical'

class ForecastModels(str, Enum):
    MLPRegressor = 'MLPRegressor'
    HistGradientBoostingRegressor = 'HistGradientBoostingRegressor'
    Prophet = 'Prophet'

class TaskState(str, Enum):
    Created = 'Created'
    Pending = 'Pending'
    Executing = 'Executing'
    Finished = 'Finished'
    Error = 'Error'

class MyException(Exception):
    pass

"""
class MLPRegressorIn(BaseModel):
    client_id: int
    model_imput_data: List[int]
    forecast_period: int

class MLPRegressorOut(BaseModel):
    forecast: List[int]
    outparams: Optional[List[int]] = None

class TrainMLPRegressorIn(BaseModel):
    client_id: int
    ts_imput_data: List[datetime]
    visitors_imput_data: List[float]
"""

"""
class ForecastIn(BaseModel):
    client_id: int
    model_imput_data: List[int]
    forecast_period: int

class ForecastOut(BaseModel):
    forecast: List[int]
    outparams: Optional[List[int]] = None
"""

class TrainIn(BaseModel):
    client_id: str
    model_name: str
    input_data: dict
    forecast_period: int
    n_lags: Optional[int] = 720

"""
class TrainOut(BaseModel):
    outparams: Optional[List[int]] = None
"""

"""
class ParametersIn(BaseModel):
    culture: str
    forecast_period: int

class ParametersOut(BaseModel):
    culture: str
    forecast_period: int
"""

class TrainTaskOut(BaseModel):
    id: int
    state: str
