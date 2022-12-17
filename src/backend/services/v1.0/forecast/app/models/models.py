""" Domain classes for the application.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022
"""

from enum import Enum
from pydantic import BaseModel
from typing import List, Optional
from datetime import date, datetime, time, timedelta

class ForecastModels(Enum):
    MLPRegressor = 1

class ForecastModels2(str, Enum):
    MLPRegressor = 'MLPRegressor'

class MyException(Exception):
    pass

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

class ForecastIn(BaseModel):
    client_id: str
    model_imput_data: List[int]
    forecast_period: int

class ForecastOut(BaseModel):
    forecast: List[int]
    outparams: Optional[List[int]] = None

class TrainIn(BaseModel):
    client_id: str
    model_imput_data: List[int]
    forecast_period: int

class TrainOut(BaseModel):
    outparams: Optional[List[int]] = None

class ParametersIn(BaseModel):
    culture: str
    forecast_period: int

class ParametersOut(BaseModel):
    culture: str
    forecast_period: int