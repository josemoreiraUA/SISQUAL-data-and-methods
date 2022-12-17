from sqlalchemy.orm import Session
from sqlalchemy import select, func
from fastapi import HTTPException

from db import models, schemas
#from db import schemas

def get_client(db: Session, client_id: str):
    return db.query(models.Client.pkey.label('client_pkey') \
        , models.Client.id.label('id') \
        , models.Client.culture.label('culture') \
        , models.Client.is_active.label('is_active') \
        ).filter(models.Client.id == client_id).first()
    #return db.query(models.Client).filter(models.Client.id == client_id).first()

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

    #print(client.id)
    if not client_exists:
        db_client = models.Client(**client.dict())
        db.add(db_client)
        db.commit()
        #db.refresh(db_client)
        return 1

    return -1

def update_client_parameters(db: Session, client_id: str, params: schemas.ClientParametersIn):
    #client_exists = db.query(models.Client).filter(models.Client.id == client_id).first()
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
    #client = models.Client.delete().where(models.Client.client_id == client_id)
    #db.commit(client)

    #delete_sql = delete(models.Client).where(models.Client.client_id == client_id)
    #db.session.execute(delete_sql)
    #db.session.commit()

    num_deleted_clients = db.query(models.Client).filter(models.Client.id == client_id).delete()
    db.commit()

    #engine.execute(x)
    #db_client = models.Client(**client.dict())
    #db.add(db_client)
    #db.commit()
    #db.refresh(db_client)
    return num_deleted_clients
    #return db_client

def get_model(db: Session, model_id: int):
    return db.query(models.Model.id.label('id') \
        , models.Model.type.label('type') \
        , models.Model.model_name.label('model_name') \
        #, models.Model.storage_name.label('storage_name') \
        , models.Model.time_trained.label('time_trained') \
        , models.Model.metrics.label('metrics') \
        , models.Model.forecast_period.label('forecast_period') \
        , models.Model.train_params.label('train_params') \
        #, models.Model.time_trained.label('time_trained') \
        ).filter(models.Model.id == model_id).first()
    #return db.query(models.Model).filter(models.Model.id == model_id).first()

def get_client_models(db: Session, client_id: str):
    client = db.query(models.Client.pkey).filter(models.Client.id == client_id).first()
    if not client:
        raise HTTPException(
		        status_code=404, detail=f'Client {client_id} not found!'
			)

    """
    result = db.execute(select(models.Client.pkey).filter(models.Client.id == client_id))

    print('*****************', client_id)
    print('*****************', client_pkey.pkey)
    print('*****************', result)
    print('*****************', type(result))

    for id in result:
        print('*****************', id, id.pkey)
    """
    return db.query(models.Model.id.label('id') \
        , models.Model.type.label('type') \
        , models.Model.model_name.label('model_name') \
        #, models.Model.storage_name.label('storage_name') \
        , models.Model.time_trained.label('time_trained') \
        , models.Model.metrics.label('metrics') \
        , models.Model.forecast_period.label('forecast_period') \
        , models.Model.train_params.label('train_params')
        ).filter(models.Model.client_pkey == client.pkey).all()

def get_client_pkey(db: Session, client_id: str):
    client = db.query(models.Client.pkey).filter(models.Client.id == client_id).first()
    if not client:
        raise HTTPException(
		        status_code=404, detail=f'Client {client_id} not found!'
			)

    return client.pkey

"""
def create_model(db: Session, model: schemas.ModelCreate, client_id: str):
    client = db.query(models.Client.pkey).filter(models.Client.id == client_id).first()
    if not client:
        raise HTTPException(
		        status_code=404, detail=f'Client {client_id} not found'
			)
"""
"""
    client_pkey = db.query(models.Client) \
        .with_entities(models.Client.pkey) \
        .filter(models.Client.id == client_id).first()
"""

"""
    client_pkey = db.query(models.Client) \
        .with_entities(models.Client.pkey) \
        .filter(models.Client.id == client_id).one()
"""
"""
    db_model = models.Model(**model.dict(), client_pkey=client.pkey)
    db.add(db_model)
    db.commit()
    #db.refresh(db_model)
    return 1
"""

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
    #return db.query(models.Model).filter(models.Model.id == model_id).first()

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
    """
    client = db.query(models.Client.pkey).filter(models.Client.id == client_id).first()
    if not client:
        raise HTTPException(
		        status_code=404, detail=f'Client {client_id} not found'
			)
    """

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
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.UserCreate):
    fake_hashed_password = user.password + "notreallyhashed"
    db_user = models.User(email=user.email, hashed_password=fake_hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_items(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Item).offset(skip).limit(limit).all()


def create_user_item(db: Session, item: schemas.ItemCreate, user_id: int):
    db_item = models.Item(**item.dict(), owner_id=user_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
"""