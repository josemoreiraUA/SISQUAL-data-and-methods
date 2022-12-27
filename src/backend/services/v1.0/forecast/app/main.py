""" 
    Forecast integration web service.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022
	
	APIs:

		/api/v1/models/{model_id}/forecast
"""

from fastapi import FastAPI, Request, Header, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session

# custom imports

from app.models.exception import MyException
from app.api.dependencies.auth import has_authorization
from app.core.config import settings, setup_app_logging#, global_file_logger, global_console_logger
from app.api.v1.api import api_router

setup_app_logging(settings)

"""
app = FastAPI(
			title='Forecast Web Service', 
			description='Forecast Integration API.', 
			version='1.0', 
			root_path='/app/api/v1')
"""

app = FastAPI(
			title='Forecast Web Service', 
			description='Forecast Integration API.', 
			version='1.0', 
			dependencies=[Depends(has_authorization)])

# allowed external connections

if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_origin_regex=settings.BACKEND_CORS_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# startup / shutdown events

@app.on_event("startup")
async def startup():
    settings.console_logger.info('startup event.')
    settings.file_logger.info('startup event.')
    #pass

@app.on_event("shutdown")
async def shutdown():
    settings.console_logger.info('shutdown event.')
    settings.file_logger.info('shutdown event.')
    #pass

# handle web server unexpected exceptions

@app.exception_handler(MyException)
async def web_server_exception_handler(request: Request, exc: MyException):
    return JSONResponse(
        status_code=418,
        content={'detail': 'An error occurred. Please contact the system admin.'},
    )

# available endpoints (routes)

app.include_router(api_router, prefix=settings.API_V1_STR)

# inject the JWT Authorization Bearer token in the response

@app.middleware("http")
async def add_jwt_bearer_token_header(request: Request, call_next):
    response = await call_next(request)
    response.headers['Authorization'] = 'Bearer ' + settings.JWT_TOKEN
    return response

# -------------------------------------------------------------------------------------------------------
# 
# -------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    """
		run service:

			poetry run hypercorn main:app -b 127.0.0.1:8000 --worker-class trio --workers 1

			poetry run daphne -b 127.0.0.1 -p 8000 main:app

			poetry run uvicorn main:app --workers 1 --host 127.0.0.1 --port 8000


			uvicorn main:app --workers 1 --port 8000
			uvicorn main:app --reload
			uvicorn main:app --root-path /api/v1 --reload

		API docs available at:

			http://127.0.0.1:8000/docs
			http://127.0.0.1:8000/redoc
			http://127.0.0.1:8000/openapi.json
    """

    #import uvicorn
    #uvicorn.run('main:app')