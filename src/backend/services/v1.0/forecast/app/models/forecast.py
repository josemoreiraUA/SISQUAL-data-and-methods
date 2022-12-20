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

class ForecastModels(str, Enum):
    MLPRegressor = 'MLPRegressor'

class ForecastIn(BaseModel):
    client_id: str
    model_imput_data: List[int]
    forecast_period: int

class ForecastOut(BaseModel):
    forecast: List[int]
    outparams: Optional[List[int]] = None
