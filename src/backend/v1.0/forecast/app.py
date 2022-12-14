""" WFM forecast integration web service.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022
	
	APIs:
		/app/api/v1/model/{id}/forecast
		/app/api/v1/model/MLPRegressor/getforecast

"""

from fastapi import FastAPI, Request, Header, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# custom imports
from models.models import MyException
from routers import mlp_route, forecast_route#, parameters_route

from pathlib import Path
from dependencies import has_authorization
from core.config import settings

from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session
from db import crud, models, schemas

from api import deps

app = FastAPI(title='MLForecast', description='MLForecast Integration API', version='1.0', dependencies=[Depends(has_authorization)])

if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_origin_regex=settings.BACKEND_CORS_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.on_event("startup")
async def startup():
    pass

@app.on_event("shutdown")
async def shutdown():
    pass

@app.exception_handler(MyException)
async def unicorn_exception_handler(request: Request, exc: MyException):
    return JSONResponse(
        status_code=418,
        content={'detail': 'An error occurred! Please contact the system admin.'},
    )

# routes

app.include_router(forecast_route.router, prefix='/app/api/v1/models/{model_id}/forecast', tags=['Forecast using a trained model'])

if __name__ == '__main__':
    """
		run:
			uvicorn app:app --workers 1 --host 127.0.0.1 --port 8000
			
			daphne -b 127.0.0.1 -p 8000 app:app
			
			hypercorn app:app -b 127.0.0.1:8000 --worker-class trio --workers 1

			uvicorn app:app --workers 1 --port 8000
			uvicorn app:app --reload
			uvicorn app:app --root-path /app/api/v1 --reload

		api docs available at:
			http://127.0.0.1:8000/docs
			http://127.0.0.1:8000/redoc
			http://127.0.0.1:8000/openapi.json

    """

    uvicorn.run('app:app')