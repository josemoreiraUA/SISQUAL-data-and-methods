from sqlalchemy.orm import Session
from sqlalchemy import select, func
from fastapi import HTTPException

import datetime

from app.db import models, schemas
from app.models.models import TypeForecast

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
        raise HTTPException(status_code=404, detail=f'Client {client_id} not found!')

    new_culture = params.culture
    new_state = params.is_active

    db.query(models.Client).filter(models.Client.pkey == client.pkey).update({'culture': new_culture, 'is_active': new_state})
    db.commit()
    db.flush()
    return True

def get_client_registration_key(db: Session, client_id: str):
    return db.query(models.Client.pkey).filter(models.Client.id == client_id).first()

def update_model_html_report(db: Session, html_report: str, model_id: int):
    upd_output = db.query(models.Model).filter(models.Model.id == model_id).update({'html_report': html_report})
    db.commit()
    db.flush()
    return upd_output

def update_task_error_report(db: Session, error_report: str, task_id: int):
    upd_output = db.query(models.TrainTask).filter(models.TrainTask.id == task_id).update({'error_report': error_report})
    db.commit()
    db.flush()
    return upd_output    

def get_model(db: Session, model_id: int):
    return db.query(models.Model.id.label('id') \
        , models.Model.type.label('type') \
        , models.Model.model_name.label('model_name') \
        , models.Model.time_trained.label('time_trained') \
        , models.Model.metrics.label('metrics') \
        , models.Model.forecast_period.label('forecast_period') \
        , models.Model.train_params.label('train_params') \
        , models.Model.type_forecast.label('type_forecast') \
        , models.Model.html_report.label('html_report')
        ).filter(models.Model.id == model_id).first()

def get_client_models(db: Session, client_id: str, type_of_forecast: TypeForecast = None):
    client = db.query(models.Client.pkey).filter(models.Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail=f'Client {client_id} not found!')

    if type_of_forecast is None:
        return db.query(models.Model.id.label('id') \
            , models.Model.type.label('type') \
            , models.Model.model_name.label('model_name') \
            , models.Model.time_trained.label('time_trained') \
            , models.Model.metrics.label('metrics') \
            , models.Model.forecast_period.label('forecast_period') \
            , models.Model.train_params.label('train_params') \
            , models.Model.type_forecast.label('type_forecast') \
            , models.Model.html_report.label('html_report')
            ).filter(models.Model.client_pkey == client.pkey).all()
    else:
        return db.query(models.Model.id.label('id') \
            , models.Model.type.label('type') \
            , models.Model.model_name.label('model_name') \
            , models.Model.time_trained.label('time_trained') \
            , models.Model.metrics.label('metrics') \
            , models.Model.forecast_period.label('forecast_period') \
            , models.Model.train_params.label('train_params') \
            , models.Model.type_forecast.label('type_forecast') \
            , models.Model.html_report.label('html_report')
            ).filter(models.Model.client_pkey == client.pkey, models.Model.type_forecast == type_of_forecast.name).all()

def get_client_pkey(db: Session, client_id: str):
    client = db.query(models.Client.pkey).filter(models.Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail=f'Client {client_id} not found!')

    return client.pkey

def create_model(db: Session, model_params: schemas.ModelCreate, client_pkey: int):
    model = models.Model(**model_params.dict(), client_pkey=client_pkey)
    db.add(model)
    db.commit()
    db.refresh(model)
    return model

def delete_model(db: Session, model_id: int) -> int:
    model = db.query(models.Model).filter(models.Model.id == model_id).first()
    
    if not model:
        return 0

    r = db.delete(model)
    db.commit()

    return r

def get_task_state(db: Session, task_id: int):
    return db.query(models.TrainTask.state.label('state')
        ).filter(models.TrainTask.id == task_id).first()

def get_task(db: Session, task_id: int):
    return db.query(models.TrainTask.id.label('id') \
        , models.TrainTask.time_created.label('time_created') \
        , models.TrainTask.time_started.label('time_started') \
        , models.TrainTask.time_finished.label('time_finished') \
        , models.TrainTask.client_pkey.label('client_pkey') \
        , models.TrainTask.model_type.label('model_type') \
        , models.TrainTask.state.label('state')
        ).filter(models.TrainTask.id == task_id).first()

def get_report(db: Session, task_id: int):
    return db.query(models.Model.html_report.label('html_report')
        ).join(models.TrainTask, models.Model.id == models.TrainTask.model_id).filter(models.TrainTask.id == task_id).first()

def get_task_error_report(db: Session, task_id: int):
    return db.query(models.TrainTask.error_report.label('error_report')
        ).filter(models.TrainTask.id == task_id).first()

def get_client_tasks(db: Session, client_id: str):
    client = db.query(models.Client.pkey).filter(models.Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail=f'Client {client_id} not found!')

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

def update_task_state(db: Session, task_id: int, new_task_state: str, flag: int, model_id: int) -> bool:
    #datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    if flag == 1:
        db.query(models.TrainTask).filter(models.TrainTask.id == task_id).update({'state': new_task_state, 'time_started': func.now()})
        #db.query(models.TrainTask).filter(models.TrainTask.id == task_id).update({'state': new_task_state, 'time_started': datetime.datetime.now()})
    elif flag == 0:
        db.query(models.TrainTask).filter(models.TrainTask.id == task_id).update({'state': new_task_state, 'time_finished': func.now(), 'model_id': model_id})
        #db.query(models.TrainTask).filter(models.TrainTask.id == task_id).update({'state': new_task_state, 'time_finished': datetime.datetime.now(), 'model_id': model_id})
    else:
        db.query(models.TrainTask).filter(models.TrainTask.id == task_id).update({'state': new_task_state})

    db.commit()
    db.flush()
    return True