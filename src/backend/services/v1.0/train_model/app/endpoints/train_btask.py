""" 
	WFM train model integration web service.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022

	Services:
		train a forecast model.
	
	APIs:
		/

	Designed options:
		...
"""

import numpy as np
import datetime
import pandas as pd
import os
from http import HTTPStatus
from typing import Union

from joblib import dump
from sklearn.neural_network import MLPRegressor
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request

from sqlalchemy.orm import Session

from models.models import TrainIn, ForecastModels, TrainTaskOut, TaskState, ForecastModels2
import tools.auxiliary_functions as myfuncs
from dependencies.db import get_db
from db import crud, models, schemas
from core.config import settings

router = APIRouter()

def check_model_type(model_type: str) -> bool:
    for model in ForecastModels2:
        if model_type == str(model.value):
            return True

    raise HTTPException(
            status_code=404, 
			detail=f'Model of type {model_type} not available!'
        )

def get_dir(models_storage_dir: str, client_id: str) -> str:
    dir = models_storage_dir + client_id

    if not os.path.exists(dir):
        os.mkdir(dir)

    return dir + '/'

def get_name(model_type: str, model_name: str) -> str:
    """
	Gets a name for a new trained model.
	"""

    current_time = datetime.datetime.now()
    day = str(current_time.day)
    month = str(current_time.month)
    year = str(current_time.year)
    hour = str(current_time.hour)
    minute = str(current_time.minute)
    second = str(current_time.second)
    microsecond = str(current_time.microsecond)

    return model_type + '_' + model_name + '_' + day + '_' + month + '_' + year + '_' + hour + '_' + minute + '_' + second + '_' + microsecond

def store(model: object, dir: str, name: str) -> bool:
    """
	Stores a model.
    """

    dump(model, dir + name + '.joblib')
    return True

def train_mlp(params: TrainIn) -> dict:
    """
	Trains a MLPRegressor model.
	"""

    data = pd.DataFrame.from_dict(params.imput_data)
    nprev = params.forecast_period

    # preprocess data
    data['ts'] = pd.to_datetime(data['ts'])
    data.shape

    forecast_data = pd.DataFrame(data.resample('H', on = 'ts').values.sum())
    forecast_data.reset_index(inplace=True)

    # remove outliers
    outliers, new_ts = myfuncs.streming_outlier_detection(forecast_data['values'].values, k=53, s=168, alpha=2)
    forecast_data['filtered_values'] = new_ts

    # preprocess data
    X, y = myfuncs.to_supervised(forecast_data['filtered_values'].values, n_lags=nprev, n_output=nprev)

    # prepare data for training
    X_train = X[:-1]
    y_train = y[:-1]
    X_test = X[-1].reshape(1,-1)
    y_test = y[-1].reshape(1,-1)

    # create model
    mlp_model = MLPRegressor(max_iter=10)

    # train model
    mlp_model.fit(X_train,y_train)

    # predict
    mlp_model_forecast = mlp_model.predict(X_test)

    # model evaluation metrics
    mlp_model_metrics = myfuncs.model_metrics(y_test.flatten(), mlp_model_forecast.flatten())

    train_params = {'k': 53, 's': 168, 'alpha': 2, 'max_iter': 10}

    return {'model': mlp_model, 'forecast': mlp_model_forecast, 'metrics': mlp_model_metrics, 'train_params': train_params}

def switch(model_type: str, params: TrainIn) -> Union[dict, None]:
    """
	Chooses which type of model to train for a request.
	"""

    switch={
       ForecastModels2.MLPRegressor.value : train_mlp(params),
       }

    return switch.get(model_type, None)

def train_model(db: Session, client_pkey: int, model_type: str, params: TrainIn, task_id: int) -> bool:
    """
	Trains a model.
	"""

    is_model_created_in_db = False
    is_model_stored_in_hd = False

    model_db_id = -1
    model_hd_name = ''

    try:
        update_result = crud.update_task_state(db, task_id, TaskState.Executing, 1)

        train_output = switch(model_type, params)

        if train_output != None:
            client_id = params.client_id

            model_name = params.model_name

            # create model in db
            model_storage_name = get_name(model_type, model_name)
            forecast_period = params.forecast_period
            model = train_output['model']
            model_metrics = train_output['metrics']
            model_train_params = train_output['train_params']

            metrics = ''
            for key, value in model_metrics.items():
                metrics += key + ':' + str(value) + ';'

            train_params = ''
            for key, value in model_train_params.items():
                train_params += key + ':' + str(value) + ';'

            model_params = schemas.ModelCreate(type = model_type \
			    , model_name = model_name \
			    , storage_name = model_storage_name + '.joblib' \
			    , metrics = metrics \
			    , forecast_period = params.forecast_period \
			    , train_params = train_params
		    )

            model_in_db = crud.create_model(db, model_params, client_pkey)
            if model_in_db:
                is_model_created_in_db = True
                model_db_id = model_in_db.id

            dir = get_dir(settings.MODELS_STORAGE_DIR, client_id)

            if store(model, dir, model_storage_name):
                is_model_stored_in_hd = True
                model_hd_name = dir + model_storage_name + '.joblib'

            update_result = crud.update_task_state(db, task_id, TaskState.Finished, 0)
            return True
    except:
        update_result = crud.update_task_state(db, task_id, TaskState.Error, -1)
        if is_model_created_in_db:
            pass
        if model_hd_name:
            pass
    finally:
        pass

    return False

@router.post('/', status_code=HTTPStatus.ACCEPTED)
async def train_model_request(model_type: str, params: TrainIn, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> dict:
    """
	Trains a model.
	"""

    client_id = params.client_id

    client_pkey = crud.get_client_pkey(db, client_id)

    check_model_type(model_type)

    task_state = TaskState.Pending

    task = schemas.TTaskCreate(
              client_pkey=client_pkey \
			, model_type=model_type \
			, state=task_state
        )

    task = crud.create_task(db, task)

    background_tasks.add_task(train_model, db, client_pkey=client_pkey, model_type=model_type, params=params, task_id=task.id)

    return {'detail': '1'}