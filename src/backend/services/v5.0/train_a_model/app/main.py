""" 
	WFM train a model integration web service.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    09/06/2023

	APIs:
		/app/api/v1/customer/flow/models/{model_type}/train
        /app/api/v1/sizing/models/{model_type}/train
        /app/api/v1/customer/flow/and/sizing/models/{model_type}/train
"""

from typing import List, Union, Optional, Any
from fastapi import FastAPI, Request, Header, Depends, HTTPException, status

# security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

# db
from sqlalchemy.orm import Session

# custom imports
from app.models.models import MyException
from app.dependencies.auth import has_authorization
from app.dependencies.db import get_db
from app.core.config import settings, setup_app_logging
from app.db import crud, schemas
from app.api.v1.api import api_router

setup_app_logging(settings)

#train_app = FastAPI(root_path="/api/v1")

train_app = FastAPI(
		title='Train a Model Web Service', 
		description='Train a Model Integration API', 
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
    settings.console_logger.info('Train a Model Service startup event.')
    settings.file_logger.info('startup event.')

@train_app.on_event("shutdown")
async def shutdown():
    settings.console_logger.info('Train a Model Service shutdown event.')
    settings.file_logger.info('shutdown event.')

# handle web server unhandled exceptions
@train_app.exception_handler(MyException)
async def web_server_exception_handler(request: Request, exc: MyException):
    return JSONResponse(
        status_code=418,
        content={'detail': 'Train a Model Service: An error occurred! Please contact the system admin.'},
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