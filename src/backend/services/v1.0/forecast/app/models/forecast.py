""" 
    Forecast domain class.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022
"""

from enum import Enum
from pydantic import BaseModel
from typing import List, Optional, Union

class ForecastModels(str, Enum):
    MLPRegressor = 'MLPRegressor'
    HistGradientBoostingRegressor = 'HistGradientBoostingRegressor'
    Prophet = 'Prophet'

class ForecastIn(BaseModel):
    client_id: str
    model_input_data: Union[dict, List[int]]
    forecast_period: int

class ForecastOut(BaseModel):
    forecast: List[int]
    #outparams: Optional[List[int]] = []
