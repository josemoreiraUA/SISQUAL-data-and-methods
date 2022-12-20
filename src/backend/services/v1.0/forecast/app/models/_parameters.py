""" 
    Domain classes for the application.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022
"""

from pydantic import BaseModel

class ParametersIn(BaseModel):
    culture: str
    forecast_period: int

class ParametersOut(BaseModel):
    culture: str
    forecast_period: int