""" 
	Clents integration API.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022

	Services:

        Get the list of clients.
        Create (register) a new client.
        Update client parameters.
        Get client details.
        List of models of a client.
        List of tasks of a client.

	APIs:

        /
        /
        /{client_id}
        /{client_id}
        /{client_id}/models
        /{client_id}/tasks        

"""

from fastapi import APIRouter, HTTPException, Depends

from sqlalchemy.orm import Session
from app.dependencies.db import get_db
from app.db import crud, schemas

router = APIRouter()

@router.get("")
def get_list_known_clients(db: Session = Depends(get_db)):
    db_clients = crud.get_clients(db)

    if not db_clients:
        return {'clients': []}

    return {'clients': db_clients}

@router.put("")
def create_client(client: schemas.ClientCreate, db: Session = Depends(get_db)):
    result = crud.create_client(db=db, client=client)

    if result == -1:
        raise HTTPException(status_code=409, detail=f'Client {client.id} already exists!')
        #return {'detail': f'Client {client.id} already exists!'}

    return {'detail': '1'}

@router.post("/{client_id}")
def update_client_parameters(client_id: str, params: schemas.ClientParametersIn, db: Session = Depends(get_db)):
    client = crud.update_client_parameters(db, client_id, params)

    return {'detail': '1'}

@router.get("/{client_id}", response_model=schemas.Client)
def get_client_details(client_id: str, db: Session = Depends(get_db)):
    client = crud.get_client(db, client_id)

    if not client:
        raise HTTPException(status_code=404, detail=f'Client {client_id} not found!')

    return client

@router.get("/{client_id}/models")
def get_client_models(client_id: str, db: Session = Depends(get_db)):
    client_models = crud.get_client_models(db, client_id)

    if not client_models:
        return {'models': []}

    return {'models': client_models}

@router.get("/{client_id}/tasks")
def get_client_tasks(client_id: str, db: Session = Depends(get_db)):
    client_tasks = crud.get_client_tasks(db, client_id)

    if not client_tasks:
        return {'tasks': []}

    return {'tasks': client_tasks}