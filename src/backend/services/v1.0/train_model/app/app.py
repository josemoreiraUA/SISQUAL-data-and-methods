""" WFM train integration web service.

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

#from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware

# custom imports
from models.models import MyException, ForecastModels, ForecastModels2
#from routers import train, train_btask, train_bproc
from routers import train_btask
from dependencies import has_authorization
from core.config import settings

from sqlalchemy.orm import Session
from db import crud, models, schemas

from api import deps

available_forecast_models = []

for model in ForecastModels2:
    available_forecast_models += [{'model_type': model.value, 'model_name': model.name}]

#train_app = FastAPI(root_path="/api/v1")
train_app = FastAPI(
		title='Train Model Web Service', 
		description='Train Model Integration API', 
		version='1.0', 
		dependencies=[Depends(has_authorization)])

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    train_app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_origin_regex=settings.BACKEND_CORS_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@train_app.on_event("startup")
async def startup():
    #train_app.state.executor = ProcessPoolExecutor(max_workers=None)
    pass

@train_app.on_event("shutdown")
async def shutdown():
    #train_app.state.executor.shutdown()
    pass

@train_app.exception_handler(MyException)
async def web_server_exception_handler(request: Request, exc: MyException):
    return JSONResponse(
        status_code=418,
        content={'detail': 'An error occurred! Please contact the system admin.'},
    )

#@train_app.get("/app/api/v1/clients/", response_model=List[schemas.Client])
@train_app.get("/app/api/v1/clients/")
def get_clients(db: Session = Depends(deps.get_db)):
    db_clients = crud.get_clients(db)

    if not db_clients:
        return {"clients": []}

    return {"clients": db_clients}

@train_app.get("/app/api/v1/clients/{client_id}", response_model=schemas.Client)
def get_client(client_id: str, db: Session = Depends(deps.get_db)):
    client = crud.get_client(db, client_id)

    if not client:
        raise HTTPException(
		        status_code=404, detail=f'Client {client_id} not found!'
			)

    return client

@train_app.post("/app/api/v1/clients/{client_id}")
def update_client_parameters(client_id: str, params: schemas.ClientParametersIn, db: Session = Depends(deps.get_db)):
    client = crud.update_client_parameters(db, client_id, params)

    return {'detail': '1'}

@train_app.get("/app/api/v1/clients/count/")
def get_number_clients(db: Session = Depends(deps.get_db)):
    num_clients = crud.get_number_clients(db)
    return {'nun_clients': num_clients}

#@train_app.post("/clients/", response_model=schemas.Client)
@train_app.put("/app/api/v1/clients/")
def create_client(client: schemas.ClientCreate, db: Session = Depends(deps.get_db)):
    result = crud.create_client(db=db, client=client)

    if result == -1:
        return {'detail': f'Client {client.id} already exists!'}

    return {'detail': '1'}

@train_app.delete("/app/api/v1/clients/{client_id}")
def delete_client(client_id: str, db: Session = Depends(deps.get_db)):
    num_deleted_clients = crud.delete_client(db, client_id)

    if num_deleted_clients == 1:
        return {'detail': '1'}

    return {'detail': f'Client {client_id} not found!'}

@train_app.get("/app/api/v1/models/")
def get_available_forecast_models():
    """
    Gets the list of models available for training and forecast.
    """

    return {"models": available_forecast_models}

@train_app.get("/app/api/v1/client/{client_id}/models")
def get_client_models(client_id: str, db: Session = Depends(deps.get_db)):
    """Gets the list of models of a client."""

    client_models = crud.get_client_models(db, client_id)

    if not client_models:
        return {'models': []}

    return {'models': client_models}

@train_app.get("/app/api/v1/models/{model_id}")
def get_model(model_id: str, db: Session = Depends(deps.get_db)):
    model = crud.get_model(db, model_id)

    if not model:
        raise HTTPException(
		        status_code=404, detail=f'Model {model_id} not found'
			)

    return model

@train_app.put("/app/api/v1/models/{client_id}")
def create_model(client_id: str, model: schemas.ModelCreate, db: Session = Depends(deps.get_db)):
    result = crud.create_model(db=db, model=model, client_id=client_id)

    if result == -1:
        return {'detail': f'Client {client.id} does not exist!'}

    return {'detail': '1'}

@train_app.get("/app/api/v1/tasks/{task_id}")
def get_task(task_id: int, db: Session = Depends(deps.get_db)):
    task = crud.get_task(db, task_id)

    if not task:
        raise HTTPException(
		        status_code=404, detail=f'Task {task_id} not found'
			)

    return task

@train_app.get("/app/api/v1/client/{client_id}/tasks")
def get_client_tasks(client_id: str, db: Session = Depends(deps.get_db)):
    """Gets the list of client tasks."""

    client_tasks = crud.get_client_tasks(db, client_id)

    if not client_tasks:
        return {'tasks': []}

    return {'tasks': client_tasks}

@train_app.get("/app/api/v1/tasks/{task_id}/state/")
def get_task_state(task_id: int, db: Session = Depends(deps.get_db)) -> Any:
    #task = crud.get_task_state(db, task_id)
    task = crud.get_task(db, task_id)
    if not task:
        raise HTTPException(
		        status_code=404, detail=f'Task {task_id} not found'
			)

    return task.state

@train_app.put("/app/api/v1/tasks/")
def create_task(task: schemas.TTaskCreate, db: Session = Depends(deps.get_db)):
    result = crud.create_task(db=db, task=task)

    if result == -1:
        return {'detail': 'Error creating task!'}

    return {'detail': '1'}

# routes/endpoints

train_app.include_router(train_btask.router, prefix='/app/api/v1/models/{model_type}/train', tags=['Train a model'])

if __name__ == '__main__':
    """
		run:
			
			uvicorn app:train_app --workers 1 --host 127.0.0.1 --port 8001
			
			daphne -b 127.0.0.1 -p 8001 app:train_app
			
			hypercorn app:train_app -b 127.0.0.1:8001 --worker-class trio --workers 1
			
			poetry run hypercorn app:train_app -b 127.0.0.1:8001 --worker-class trio --workers 1
			
			uvicorn app:train_app --reload
			uvicorn app:train_app --root-path /app/api/v1 --reload

		API docs available at:
		
			http://127.0.0.1:8001/docs
			http://127.0.0.1:8001/redoc
			http://127.0.0.1:8001/openapi.json

    """

    #import uvicorn
    #uvicorn.run('app:train_app')