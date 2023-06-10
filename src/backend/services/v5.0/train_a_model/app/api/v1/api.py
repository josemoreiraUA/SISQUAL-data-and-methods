""" 
    API V1 endpoints.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022
"""

from fastapi import APIRouter
from app.api.v1.endpoints import train_btask, models, clients, tasks

api_router = APIRouter()

#api_router.include_router(
#			train_btask.router, 
#			prefix='/models/{model_type}/train', 
#			tags=['Train a Model Service API Endpoints'])

api_router.include_router(
			models.router, 
			prefix='/models', 
			tags=['Models API Endpoints'])

api_router.include_router(
			clients.router, 
			prefix='/clients', 
			tags=['Clients API Endpoints'])

api_router.include_router(
			tasks.router, 
			prefix='/tasks', 
			tags=['Tasks API Endpoints'])

api_router.include_router(
			train_btask.router, 
			prefix='', 
			tags=['Train a Model Service API Endpoints'])