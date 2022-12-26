""" 
	Train model integration web service.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022

	Services:

		train a forecast model.
	
	APIs:

		/models/{model_type}/train

	Designed options:

		...

    TODOS:

        - This needs a logger!
        - Remove DB records and HD files in case of failure.
        - DB structure needs to be reviewed.
"""

import numpy as np
import datetime
import pandas as pd
import os
from http import HTTPStatus
from typing import Union

from joblib import dump
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request

from sqlalchemy.orm import Session

from app.models.models import TrainIn, ForecastModels, TrainTaskOut, TaskState, ForecastModels
import app.tools.auxiliary_functions as myfuncs
from app.dependencies.db import get_db
from app.db import crud, schemas#, models
from app.core.config import settings

router = APIRouter()

def check_model_type(model_type: str) -> bool:
    for model in ForecastModels:
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

def pre_process_data(input_data: dict) -> dict:
    """
    Pre-processes the input data for training.
    """

    data = pd.DataFrame.from_dict(input_data)

    data['ds'] = pd.to_datetime(data['ds'])
    data.shape

    data_resample = pd.DataFrame(data.resample('H', on = 'ds').values.sum())
    data_resample.reset_index(inplace=True)

    return data_resample

def train_hgbr(params: TrainIn) -> dict:
    """
    Trains a HistGradientBoostingRegressor model.
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
    """

    K = 53
    S = 168
    ALPHA = 2
    SAMPLE_TEST_SIZE = params.forecast_period

    data = pre_process_data(params.input_data)

    data['Month'] = data['ds'].apply(lambda x: x.month)
    data['Day'] = data['ds'].apply(lambda x: x.weekday())
    data['Hour'] = data['ds'].apply(lambda x: x.hour)
    data['Closed'] = data['values'].apply(lambda x: myfuncs.closed(x))

    outliers, filtered_values = myfuncs.streming_outlier_detection(data['values'].values, k=K, s=S, alpha=ALPHA)
    data['filtered_values'] = filtered_values

    X = data[['Month','Day','Hour']]

    X_train = X[:-SAMPLE_TEST_SIZE]
    X_test = X[-SAMPLE_TEST_SIZE:]
    y_train = data[:-SAMPLE_TEST_SIZE]['filtered_values']
    y_test = data[-SAMPLE_TEST_SIZE:]['values']

    # create model instance
    hgbr_model = HistGradientBoostingRegressor()

    # train model
    hgbr_model.fit(X_train, y_train)

    # get params
    model_params = hgbr_model.get_params(deep=True)

    # predict
    hgbr_model_forecast = hgbr_model.predict(X_test)

    """
    ind = ''
    i = 0
    for row in X_test.itertuples():
        ind += str(i) + ','
        i += 1

    print(ind)
    """

    """
    month = ''
    for row in X_test.itertuples():
        month += str(row[1]) + ','

    day = ''
    for row in X_test.itertuples():
        day += str(row[2]) + ','

    hour = ''
    for row in X_test.itertuples():
        hour += str(row[3]) + ','

    print(month)
    print(day)
    print(hour)

    for row in X_test.itertuples():
        print(row)
    """

    # 'pandas.core.frame.DataFrame'
    """
    print('-------------------------------------------')
    print(type(X_test))
    print('-------------------------------------------')
    print(X_test)
    print('-------------------------------------------')
    print('-------------------------------------------')
    """

    # model evaluation metrics
    hgbr_model_metrics = myfuncs.model_metrics(y_test, hgbr_model_forecast)

    outlier_detection_parameters = {'k': K, 's': S, 'alpha': ALPHA}

    custom_train_parameters = {}

    return {'model': hgbr_model, 
            'model_parameters': model_params,    
            'forecast': hgbr_model_forecast, 
            'metrics': hgbr_model_metrics, 
            'outlier_detection_parameters': outlier_detection_parameters,
            'custom_train_parameters': custom_train_parameters
            }

def train_mlp(params: TrainIn) -> dict:
    """
    Trains a MLPRegressor model.
    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    """

    N_PREV = params.forecast_period
    K = 53
    S = 168
    ALPHA = 2
    MAX_N_ITERATIONS = 10

    # pre-process data
    data = pre_process_data(params.input_data)

    # remove outliers
    outliers, filtered_values = myfuncs.streming_outlier_detection(data['values'].values, k=K, s=S, alpha=ALPHA)
    data['filtered_values'] = filtered_values

    # additional pre-process data step
    X, y = myfuncs.to_supervised(data['filtered_values'].values, n_lags=N_PREV, n_output=N_PREV)

    # prepare data for training
    X_train = X[:-1]
    y_train = y[:-1]
    X_test = X[-1].reshape(1,-1)
    y_test = y[-1].reshape(1,-1)

    # create model
    mlp_model = MLPRegressor(max_iter=MAX_N_ITERATIONS)

    # train model
    mlp_model.fit(X_train,y_train)

    # get params
    model_params = mlp_model.get_params(deep=True)

    # predict
    mlp_model_forecast = mlp_model.predict(X_test)

    # model evaluation metrics
    mlp_model_metrics = myfuncs.model_metrics(y_test.flatten(), mlp_model_forecast.flatten())

    outlier_detection_params = {'k': K, 's': S, 'alpha': ALPHA}

    train_params = {'max_number_iterations': MAX_N_ITERATIONS}

    return {'model': mlp_model, 
            'model_parameters': model_params,    
            'forecast': mlp_model_forecast, 
            'metrics': mlp_model_metrics, 
            'outlier_detection_parameters': outlier_detection_params,
            'custom_train_parameters': train_params
            }

def train_mlp_deprecated(params: TrainIn) -> dict:
    """
	Trains a MLPRegressor model.
    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
	"""

    data = pd.DataFrame.from_dict(params.input_data)
    nprev = params.forecast_period

    # preprocess data
    data['ds'] = pd.to_datetime(data['ds'])
    data.shape

    forecast_data = pd.DataFrame(data.resample('H', on = 'ds').values.sum())
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

    # get params
    model_params = mlp_model.get_params(deep=True)
    print(model_params)

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

    if ForecastModels.MLPRegressor.value == model_type:
        return train_mlp(params)
    elif ForecastModels.HistGradientBoostingRegressor.value == model_type:
        return train_hgbr(params)

    """
    switch={
       ForecastModels.MLPRegressor.value : train_mlp(params),
       ForecastModels.HistGradientBoostingRegressor.value : train_hgbr(params),
       }
    """

    #return switch.get(model_type, None)
    return None

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

            # create model in the DB
            model_storage_name = get_name(model_type, model_name)
            forecast_period = params.forecast_period

            model = train_output['model']
            outlier_detection_parameters = train_output['outlier_detection_parameters']
            model_parameters = train_output['model_parameters']
            custom_train_parameters = train_output['custom_train_parameters']
            model_metrics = train_output['metrics']
            forecast = train_output['forecast']

            # >> temp 
            metrics_csv = ''
            for key, value in model_metrics.items():
                metrics_csv += key + ':' + str(value) + ';'

            custom_train_parameters_csv = ''
            for key, value in custom_train_parameters.items():
                custom_train_parameters_csv += key + ':' + str(value) + ';'
            # << temp

            model_params = schemas.ModelCreate(type = model_type \
			    , model_name = model_name \
			    , storage_name = model_storage_name + '.joblib' \
			    , metrics = metrics_csv \
			    , forecast_period = params.forecast_period \
			    , train_params = custom_train_parameters_csv
		    )

            model_in_db = crud.create_model(db, model_params, client_pkey)
            if model_in_db:
                is_model_created_in_db = True
                model_db_id = model_in_db.id

            # store model in the HD
            dir = get_dir(settings.MODELS_STORAGE_DIR, client_id)

            if store(model, dir, model_storage_name):
                is_model_stored_in_hd = True
                model_hd_name = dir + model_storage_name + '.joblib'

            # update the state of the task in the DB
            update_result = crud.update_task_state(db, task_id, TaskState.Finished, 0)
            return True
        else:
            # update the state of the task in the DB
            update_result = crud.update_task_state(db, task_id, TaskState.Error, -1)
    except Exception as e:
        #print(e)
        # update the state of the task in the DB
        update_result = crud.update_task_state(db, task_id, TaskState.Error, -1)
        if is_model_created_in_db:
            # TODO: remove record from the DB
            pass
        if model_hd_name:
            # TODO: remove model from the HD
            pass
    finally:
        pass

    return False

@router.post('', status_code=HTTPStatus.ACCEPTED)
async def train_model_request(model_type: str, params: TrainIn, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> dict:
    """
	Trains a model.
	"""

    # request validation steps

    client_id = params.client_id

    client_pkey = crud.get_client_pkey(db, client_id)

    check_model_type(model_type)

    # create task

    task_state = TaskState.Pending

    task = schemas.TTaskCreate(
              client_pkey=client_pkey \
			, model_type=model_type \
			, state=task_state
        )

    task = crud.create_task(db, task)

    # create background task to train the model

    background_tasks.add_task(train_model, db, client_pkey=client_pkey, model_type=model_type, params=params, task_id=task.id)

    # reply to the client (task accepted)

    return {'detail': '1'}