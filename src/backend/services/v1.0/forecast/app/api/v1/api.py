""" 
    API V1 endpoints.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022
"""

from fastapi import APIRouter
from app.api.v1.endpoints import forecast

api_router = APIRouter()

api_router.include_router(
			forecast.router, 
			prefix='/models/{model_id}/forecast', 
			tags=['Forecast using a trained model'])
