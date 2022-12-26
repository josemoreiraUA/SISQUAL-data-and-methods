""" 
	WFM train model integration web service.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022

	APIs:
		/app/api/v1/models/{model_type}/train
"""

from typing import List, Union, Optional, Any
from fastapi import FastAPI, Request, Header, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.models.models import MyException#, ForecastModels
#from app.endpoints import train_btask
from app.dependencies.auth import has_authorization
from app.dependencies.db import get_db
#from core.config import settings
from app.core.config import settings, global_file_logger, global_console_logger
from app.db import crud, schemas
from app.api.v1.api import api_router

#train_app = FastAPI(root_path="/api/v1")

train_app = FastAPI(
		title='Train Model Web Service', 
		description='Train Model Integration API', 
		version='1.0', 
		dependencies=[Depends(has_authorization)])

# allowed origins

if settings.BACKEND_CORS_ORIGINS:
    train_app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_origin_regex=settings.BACKEND_CORS_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# startup and shutdown application events

@train_app.on_event("startup")
async def startup():
    global_console_logger.info('train model service startup event.')
    global_file_logger.info('train model service startup event.')

    #pass

@train_app.on_event("shutdown")
async def shutdown():
    global_console_logger.info('train model service shutdown event.')
    global_file_logger.info('train model service shutdown event.')

    #pass

# handle web server unhandled exceptions

@train_app.exception_handler(MyException)
async def web_server_exception_handler(request: Request, exc: MyException):
    return JSONResponse(
        status_code=418,
        content={'detail': 'An error occurred! Please contact the system admin.'},
    )

# routes/endpoints

train_app.include_router(api_router, prefix=settings.API_V1_STR)

# inject rhe JWT Bearer Authorization token in the response

@train_app.middleware("http")
async def add_jwt_bearer_token_header(request: Request, call_next):
    response = await call_next(request)
    response.headers['Authorization'] = 'Bearer ' + settings.JWT_TOKEN
    return response

# ---------------------------------------------------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    """
		run:

            poetry run hypercorn main:train_app -b 127.0.0.1:8001 --worker-class trio --workers 1

            poetry run daphne -b 127.0.0.1 -p 8001 main:train_app

            poetry run uvicorn main:train_app --workers 1 --host 127.0.0.1 --port 8001

			uvicorn main:train_app --reload
			uvicorn main:train_app --root-path /api/v1 --reload

		API docs available at:
		
			http://127.0.0.1:8001/docs
			http://127.0.0.1:8001/redoc
			http://127.0.0.1:8001/openapi.json

    """

    #import uvicorn
    #uvicorn.run('main:train_app')