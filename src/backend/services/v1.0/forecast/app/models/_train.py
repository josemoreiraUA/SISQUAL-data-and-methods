""" 
    Domain classes for the application.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022
"""

from pydantic import BaseModel
from typing import List, Optional

class TrainIn(BaseModel):
    client_id: str
    model_imput_data: List[int]
    forecast_period: int

class TrainOut(BaseModel):
    outparams: Optional[List[int]] = None
