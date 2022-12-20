from sqlalchemy.orm import Session
from typing import Union, Any

from db import models

def get_client_pkey(db: Session, client_id: str) -> Union[int, None]:

    client = db.query(models.Client.pkey).filter(models.Client.id == client_id).first()

    if not client:
        return None

    return client.pkey

def get_model_details(db: Session, model_id: int, client_pkey: int) -> Any:
    return db.query(models.Model.type.label('type') \
        , models.Model.storage_name.label('storage_name') \
        , models.Model.forecast_period.label('forecast_period')
        ).filter(models.Model.id == model_id, models.Model.client_pkey == client_pkey).first()

"""
def get_client(db: Session, client_id: str):
    return db.query(models.Client.pkey.label('client_pkey') \
        , models.Client.id.label('id') \
        , models.Client.culture.label('culture') \
        , models.Client.is_active.label('is_active') \
        ).filter(models.Client.id == client_id).first()

def get_clients(db: Session):
    return db.query(models.Client.pkey.label('client_pkey') \
        , models.Client.id.label('id') \
        , models.Client.culture.label('culture') \
        , models.Client.is_active.label('is_active') \
        ).all()

def get_number_clients(db: Session):
    return db.query(models.Client).count()

def create_client(db: Session, client: schemas.ClientCreate):
    client_exists = db.query(models.Client).filter(models.Client.id == client.id).first()

    if not client_exists:
        db_client = models.Client(**client.dict())
        db.add(db_client)
        db.commit()

        return 1

    return -1

def update_client_parameters(db: Session, client_id: str, params: schemas.ClientParametersIn):
    client = db.query(models.Client.pkey).filter(models.Client.id == client_id).first()

    if not client:
        raise HTTPException(
		        status_code=404, detail=f'Client {client_id} not found!'
			)

    new_culture = params.culture

    db.query(models.Client).filter(models.Client.pkey == client.pkey).update({'culture': new_culture})
    db.commit()
    db.flush()
    return True

def delete_client(db: Session, client_id: str):
    num_deleted_clients = db.query(models.Client).filter(models.Client.id == client_id).delete()
    db.commit()

    return num_deleted_clients

def get_model(db: Session, model_id: int):
    return db.query(models.Model.id.label('id') \
        , models.Model.type.label('type') \
        , models.Model.model_name.label('model_name') \
        , models.Model.storage_name.label('storage_name') \
        , models.Model.time_trained.label('time_trained') \
        , models.Model.metrics.label('metrics') \
        , models.Model.forecast_period.label('forecast_period') \
        , models.Model.train_params.label('train_params') \
        ).filter(models.Model.id == model_id).first()

def get_client_models(db: Session, client_id: str):
    client = db.query(models.Client.pkey).filter(models.Client.id == client_id).first()
    if not client:
        raise HTTPException(
		        status_code=404, detail=f'Client {client_id} not found!'
			)

    return db.query(models.Model.id.label('id') \
        , models.Model.type.label('type') \
        , models.Model.model_name.label('model_name') \
        , models.Model.time_trained.label('time_trained') \
        , models.Model.metrics.label('metrics') \
        , models.Model.forecast_period.label('forecast_period') \
        , models.Model.train_params.label('train_params')
        ).filter(models.Model.client_pkey == client.pkey).all()

def create_model(db: Session, model_params: schemas.ModelCreate, client_pkey: int):
    model = models.Model(**model_params.dict(), client_pkey=client_pkey)
    db.add(model)
    db.commit()
    db.refresh(model)
    return model

def get_task(db: Session, task_id: int):
    return db.query(models.TrainTask.id.label('id') \
        , models.TrainTask.time_created.label('time_created') \
        , models.TrainTask.time_started.label('time_started') \
        , models.TrainTask.time_finished.label('time_finished') \
        , models.TrainTask.client_pkey.label('client_pkey') \
        , models.TrainTask.model_type.label('model_type') \
        , models.TrainTask.state.label('state')
        ).filter(models.TrainTask.id == task_id).first()

def get_client_tasks(db: Session, client_id: str):
    client = db.query(models.Client.pkey).filter(models.Client.id == client_id).first()
    if not client:
        raise HTTPException(
		        status_code=404, detail=f'Client {client_id} not found!'
			)

    return db.query(models.TrainTask.id.label('id') \
        , models.TrainTask.time_created.label('time_created') \
        , models.TrainTask.time_started.label('time_started') \
        , models.TrainTask.time_finished.label('time_finished') \
        , models.TrainTask.model_type.label('model_type')
        , models.TrainTask.state.label('state')
        ).filter(models.TrainTask.client_pkey == client.pkey).all()

def create_task(db: Session, task_params: schemas.TTaskCreate):
    task = models.TrainTask(**task_params.dict())
    db.add(task)
    db.commit()
    db.refresh(task)
    return task

def update_task_state(db: Session, task_id: int, new_task_state: str, flag: int) -> bool:
    if flag == 1:
        db.query(models.TrainTask).filter(models.TrainTask.id == task_id).update({'state': new_task_state, 'time_started': func.now()})
    elif flag == 0:
        db.query(models.TrainTask).filter(models.TrainTask.id == task_id).update({'state': new_task_state, 'time_finished': func.now()})
    else:
        db.query(models.TrainTask).filter(models.TrainTask.id == task_id).update({'state': new_task_state})

    db.commit()
    db.flush()
    return True
"""