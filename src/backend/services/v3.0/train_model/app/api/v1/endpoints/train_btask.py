""" 
	Train model integration web service.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022

	Services:

		train a model.
	
	APIs:

		/models/{model_type}/train

    TODOS:
        - DB structure needs to be reviewed.
"""

import numpy as np
import datetime
import pandas as pd
import os
from http import HTTPStatus
from typing import Union
import uuid
#import uuid6
from math import sqrt, ceil, floor

import matplotlib

import shutil

from joblib import dump

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from prophet import Prophet

from prophet.serialize import model_to_json
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request

from sqlalchemy.orm import Session

# custom imports

from app.models.models import TrainIn, ForecastModels, TrainTaskOut, TaskState, ForecastModels, MessageLevel#, RoundingMethod
import app.tools.auxiliary_functions as myfuncs
from app.dependencies.db import get_db
from app.db import crud, schemas
from app.core.config import settings
from app.models.models import MyException

matplotlib.use('AGG')

router = APIRouter()

# --------------------------------------------------------------------
# auxiliary functions.
# --------------------------------------------------------------------

def is_model_available(model_type: str) -> bool:
    for model in ForecastModels:
        if model_type == model.value:
            return True

    return False

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

    return model_type + '_' + model_name.strip().replace(' ', '_') + '_' + day + '_' + month + '_' + year + '_' + hour + '_' + minute + '_' + second + '_' + microsecond

def get_html_report_dir() -> str:
    current_time = datetime.datetime.now()
    day = str(current_time.day)
    month = str(current_time.month)
    year = str(current_time.year)
    hour = str(current_time.hour)
    minute = str(current_time.minute)
    second = str(current_time.second)

    return 'report_' + day + '_' + month + '_' + year + '_' + hour + '_' + minute + '_' + second + '_' + str(uuid.uuid4())

def log_message(message: str, level: MessageLevel, console: bool = True, file: bool = True) -> None:
    if level == MessageLevel.Info:
        if console:
            settings.console_logger.info(message)

        if file:
            settings.file_logger.info(message)
    elif level == MessageLevel.Success:
        if console:
            settings.console_logger.success(message)

        if file:
            settings.file_logger.success(message)
    elif level == MessageLevel.Trace:
        if console:
            settings.console_logger.trace(message)

        if file:
            settings.file_logger.trace(message)
    elif level == MessageLevel.Error:
        if console:
            settings.console_logger.error(message)

        if file:
            settings.file_logger.error(message)
    elif level == MessageLevel.Critical:
        if console:
            settings.console_logger.critical(message)

        if file:
            settings.file_logger.critical(message)
    elif level == MessageLevel.Warning:
        if console:
            settings.console_logger.warning(message)

        if file:
            settings.file_logger.warning(message)
    else:
        if console:
            settings.console_logger.debug(message)

        if file:
            settings.file_logger.debug(message)

def cleanup(task_id: int, model_in_db: int, model_in_hd: str, report_contents_dir: str, db: Session) -> bool:
    log_message(f'Task {task_id}: starting cleanup procedure!', MessageLevel.Info)

    # cleanup
    if model_in_db != None:
        model_id = model_in_db

        r = crud.delete_model(db, model_id)
        if r == 0:
            log_message(f'Task {task_id}: error removing model {model_id} from db!', MessageLevel.Error)
        else:
            log_message(f'Task {task_id}: model {model_id} removed from db!', MessageLevel.Success)

    if model_in_hd != None:
        try:
            os.remove(model_in_hd)
            log_message(f'Task {task_id}: model {model_in_hd} removed from hd!', MessageLevel.Success)
        except Exception as e:
            log_message(f'Task {task_id}: error removing model {model_in_hd} from hd ({e})!', MessageLevel.Error)

    if report_contents_dir != None:
        try:
            shutil.rmtree(report_contents_dir)
            log_message(f'Task {task_id}: report {report_contents_dir} removed from hd!', MessageLevel.Success)
        except Exception as e:
            log_message(f'Task {task_id}: error removing report {report_contents_dir} from hd ({e})!', MessageLevel.Error)



    """
    if os.path.exists(model_path):
        os.remove(model_path)
    else:
        print("The file does not exist")
    """

    log_message(f'Task {task_id}: cleanup procedure finished!', MessageLevel.Info)
    return True

# --------------------------------------------------------------------
# pre processing data functions.
# --------------------------------------------------------------------

def pre_process_data(input_data: dict) -> dict:
    """
    Pre-processes the input data for training.
    """

    data = pd.DataFrame.from_dict(input_data)

    data['ds'] = pd.to_datetime(data['ds'])
    data.shape

    #duplicateRows = data[data.duplicated(['ds'])]

    #settings.file_logger.info(str(data['ds'].size))
    #settings.file_logger.info(str(data['y'].size))

    data_resample = pd.DataFrame(data.resample('H', on = 'ds').y.sum())
    data_resample.reset_index(inplace=True)

    #settings.file_logger.info(str(data_resample.size))
    #settings.file_logger.info(str(data_resample['ds'].size))
    #settings.file_logger.info(str(data_resample['y'].size))

    return data_resample

def pre_process_data_multivariate(input_data: dict) -> dict:
    data = pd.DataFrame.from_dict(input_data)

    data['ds'] = pd.to_datetime(data['ds'])
    data.shape

    data_resample = pd.DataFrame(data.resample('H', on = 'ds').sum())
    data_resample.reset_index(inplace=True)

    return data_resample

# --------------------------------------------------------------------
# html report functions.
# --------------------------------------------------------------------

def model_details_to_html(
    model_in_db, 
    model_details_elem_title: str,
    model_type_elem_title: str,
    model_name_elem_title: str,
    model_time_trained_elem_title: str,
    forecast_period_elem_title: str,
    num_lags_elem_title: str) -> str:

    html = '<h3>' + model_details_elem_title + '</h3>'
    html += "<table>"
    html += '<tbody>'
    html += '<tr>'
    html += '<td>' + model_type_elem_title + '</td>'
    html += '<td>' + model_in_db.type + '</td>'
    html += '</tr>'
    html += '<tr>'
    html += '<td>' + model_name_elem_title + '</td>'
    html += '<td>' + model_in_db.model_name + '</td>'
    html += '</tr>'
    html += '<tr>'
    html += '<td>' + model_time_trained_elem_title + '</td>'
    html += '<td>' + str(model_in_db.time_trained).replace('T', ' ') + '</td>'
    html += '</tr>'
    html += '<tr>'
    html += '<td>' + forecast_period_elem_title + '</td>'
    html += '<td>' + str(model_in_db.forecast_period) + '</td>'
    html += '</tr>'

    if model_in_db.type == ForecastModels.MLPRegressor:
        html += '<tr>'
        html += '<td>' + num_lags_elem_title + '</td>'
        html += '<td>' + str(model_in_db.n_lags) + '</td>'
        html += '</tr>'

    html += '</tbody>'
    html += '</table>'

    return html

def to_html(data: dict, title: str, round_val: bool = False, num_decimals: int = 2) -> str:
    html = ''

    if data is not None and len(data) > 0:
        html += '<h4>' + title + '</h4>'
        html += '<table>'
        html += '<tbody>'

        for key, value in data.items():
            html += '<tr>'
            html += '<td>' + key + '</td>'

            if round_val:
                html += '<td>' + str(round(value, num_decimals)) + '</td>'
            else:
                html += '<td>' + str(value) + '</td>'

            html += '</tr>'

        html += '</tbody>'
        html += '</table>'

    return html

def embed_images_in_html(images: dict, client_id: str, report_uri: str) -> str:

    html = ''
    imgs = images['imgs']
    title = images['title']

    if images is not None and len(images) > 0:
        html += '<h4>' + title + '</h4>'

        for img in imgs:
            html += '<img src="' + settings.IMG_WEBSERVER_URI + '/' + client_id + '/' + report_uri + '/' + img['img_name'] + '" alt="' + img['title'] + '" />'

    return html

def create_html_report(
    model_in_db,
    outlier_detection_parameters: dict,
    model_parameters: dict,
    custom_train_parameters: dict,
    model_metrics: dict,
    forecast_imgs: dict,
    metrics_imgs: dict,
    model_components_imgs: dict,
    client_id: str,
    report_dir_name: str,
    customer_flow_and_team_size: bool
    ) -> str:
    """
    Creates an html report
    """

    # report encapsulating div element
    html_report = "<div class='htmlReport'>"

    # title labels
    model_details_elem_title = 'Model Details'
    model_type_elem_title = 'Type'
    model_name_elem_title = 'Name'
    model_time_trained_elem_title = 'Time Trained'
    forecast_period_elem_title = 'Forecast Period'
    num_lags_elem_title = 'Number of Lags'

    outlier_detection_parameters_elem_title = 'Ooutlier Detection Parameters'
    model_parameters_elem_title = 'Model Parameters'
    custom_train_parameters_elem_title = 'Custom Train Parameters'
    model_metrics_elem_title = 'Model Metrics'

    # model details
    html_report += model_details_to_html(
            model_in_db=model_in_db,
            model_details_elem_title=model_details_elem_title,
            model_type_elem_title=model_type_elem_title,
            model_name_elem_title=model_name_elem_title,
            model_time_trained_elem_title=model_time_trained_elem_title,
            forecast_period_elem_title=forecast_period_elem_title,
            num_lags_elem_title=num_lags_elem_title
        )

    # outlier detection parameters
    #html_report += to_html(data=outlier_detection_parameters, title=outlier_detection_parameters_elem_title)

    # model parameters
    #html_report += to_html(data=model_parameters, title=model_parameters_elem_title)

    # custom train parameters
    #html_report += to_html(data=custom_train_parameters, title=custom_train_parameters_elem_title)

    # model metrics

    #metrics = {}
    #metrics['mae'] = model_metrics['mae']
    #html_report += to_html(data=metrics, title=model_metrics_elem_title, round_val=True, num_decimals=3)


    if model_in_db.type == ForecastModels.RandomForestRegressor:
        d_metrics = {}

        d_metrics['Raw MAE'] = model_metrics['raw']['mae']
        d_metrics['Round MAE'] = model_metrics['round']['mae']
        d_metrics['Floor MAE'] = model_metrics['floor']['mae']
        d_metrics['Ceil MAE'] = model_metrics['ceil']['mae']

        html_report += to_html(data=d_metrics, title=model_metrics_elem_title, round_val=True, num_decimals=3)
    elif customer_flow_and_team_size :
        d_metrics = {}

        d_metrics['Tickets Raw MAE'] = model_metrics['tickets_raw']['mae']
        d_metrics['Tickets Round MAE'] = model_metrics['tickets_round']['mae']
        d_metrics['Tickets Floor MAE'] = model_metrics['tickets_floor']['mae']
        d_metrics['Tickets Ceil MAE'] = model_metrics['tickets_ceil']['mae']

        d_metrics['Team Size Raw MAE'] = model_metrics['team_size_raw']['mae']
        d_metrics['Team Size Round MAE'] = model_metrics['team_size_round']['mae']
        d_metrics['Team Size Floor MAE'] = model_metrics['team_size_floor']['mae']
        d_metrics['Team Size Ceil MAE'] = model_metrics['team_size_ceil']['mae']

        html_report += to_html(data=d_metrics, title=model_metrics_elem_title, round_val=True, num_decimals=3)
    elif model_in_db.type == ForecastModels.Prophet:
        html_report += to_html(data=model_metrics, title=model_metrics_elem_title, round_val=True, num_decimals=3)
    else:
        mae = {'MAE': model_metrics['mae']}
        html_report += to_html(data=mae, title=model_metrics_elem_title, round_val=True, num_decimals=3)

    # add images
    if metrics_imgs is not None:
        html_report += embed_images_in_html(metrics_imgs, client_id, report_dir_name)

    if model_components_imgs is not None:
        html_report += embed_images_in_html(model_components_imgs, client_id, report_dir_name)

    if forecast_imgs is not None:
        html_report += embed_images_in_html(forecast_imgs, client_id, report_dir_name)

    html_report += '</div>'

    return html_report

# --------------------------------------------------------------------
# storing machine learning models functions.
# --------------------------------------------------------------------

def store(model: object, dir: str, name: str) -> bool:
    """
    Stores a model using (joblib) (pickle).
    """

    dump(model, dir + name + '.joblib')
    return True

def store_using_json(model: object, dir: str, name: str) -> bool:
    """
    Stores a model using (json).
    """
    
    with open(dir + name + '.json', 'w') as fout:
        fout.write(model_to_json(model))

    return True

# --------------------------------------------------------------------
# train machine learning models functions.
# --------------------------------------------------------------------

"""
    Input data: ds, x, y
        ds -> datetime
        x  -> number of customers
        y  -> number of cashiers that sold items (caixas que venderam artigos)
"""
def train_RandomForestRegressor(params: TrainIn, task_id: int) -> dict:

    log_message(f'Task {task_id}: pre-process data step!', MessageLevel.Info)

    data = pd.DataFrame.from_dict(params.input_data)
    #rounding_method = params.rounding_method

    forecast_period = params.forecast_period

    data['ds'] = pd.to_datetime(data['ds'])
    #data.shape

    data['Month'] = data['ds'].apply(lambda x: x.month)
    data['Day'] = data['ds'].apply(lambda x: x.weekday())
    data['Hour'] = data['ds'].apply(lambda x: x.hour)

    data.drop(columns=['ds'], inplace=True)
    data = data[['Month', 'Day', 'Hour', 'x', 'y']]

   #log_message(f'Task {task_id}: training model!', MessageLevel.Info)

    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values

    #log_message(f'Task {task_id}: training model!', MessageLevel.Info)

    X_train = X[:-forecast_period]
    y_train = y[:-forecast_period]
    X_test = X[-forecast_period:]
    y_test = y[-forecast_period:]

    """
    N = data.shape[0]
    K = N * 80 / 100
    K = int(K)

    X_train = X[:K]
    y_train = y[:K]
    X_test = X[K:]
    y_test = y[K:]
    """

    # training step
    log_message(f'Task {task_id}: training model!', MessageLevel.Info)

    rfrgr = RandomForestRegressor()
    rfrgr.fit(X_train, y_train)
    #rfrgr.fit(X, y)

    log_message(f'Task {task_id}: model trained!', MessageLevel.Info)

    log_message(f'Task {task_id}: getting model evaluation metrics!', MessageLevel.Info)

    y_pred = rfrgr.predict(X_test)

    y_pred_round = []
    y_pred_ceil = []
    y_pred_floor = []
    
    """
    equal = [0, 0, 0]
    plus_1 = [0, 0, 0]
    minus_1 = [0, 0, 0]
    gt_1 = [0, 0, 0]
    lt_1 = [0, 0, 0]
    """
    
    k = 0
    for el in y_pred:
        _round = round(el)
        _ceil = ceil(el)
        _floor = floor(el)

        """
        v = y_test[k]
        
        if v == _round:
            equal[0] += 1
        elif v == _round + 1:
            plus_1[0] += 1
        elif v == _round - 1:
            minus_1[0] += 1
        elif _round + 1 > v:
            gt_1[0] += 1
        elif _round - 1 < v:
            lt_1[0] += 1
            
        if v == _floor:
            equal[1] += 1
        elif v == _floor + 1:
            plus_1[1] += 1
        elif v == _floor - 1:
            minus_1[1] += 1
        elif _floor + 1 > v:
            gt_1[1] += 1
        elif _floor - 1 < v:
            lt_1[1] += 1
            
        if v == _ceil:
            equal[2] += 1
        elif v == _ceil + 1:
            plus_1[2] += 1
        elif v == _ceil - 1:
            minus_1[2] += 1
        elif _ceil + 1 > v:
            gt_1[2] += 1
        elif _ceil - 1 < v:
            lt_1[2] += 1
        """

        y_pred_round.append(_round)
        y_pred_ceil.append(_ceil)
        y_pred_floor.append(_floor)
        
        k += 1

    raw_model_metrics = myfuncs.model_metrics(y_test, y_pred)
    round_model_metrics = myfuncs.model_metrics(y_test, y_pred_round)
    ceil_model_metrics = myfuncs.model_metrics(y_test, y_pred_ceil)
    floor_model_metrics = myfuncs.model_metrics(y_test, y_pred_floor)

    metrics = {}
    metrics['raw'] = raw_model_metrics
    metrics['round'] = round_model_metrics
    metrics['ceil'] = ceil_model_metrics
    metrics['floor'] = floor_model_metrics    

    return {'model': rfrgr,
            'model_parameters': None,
            'forecast': y_pred,
            'metrics': metrics,
            'outlier_detection_parameters': None,
            'custom_train_parameters': None,
            'report_dir_name': None
            }    

def train_hgbr(params: TrainIn, task_id: int) -> dict:
    """
    Trains a HistGradientBoostingRegressor model.
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
    """

    K = 53
    S = 168
    ALPHA = 2
    SAMPLE_TEST_SIZE = params.forecast_period

    # initial pre-process data step
    log_message(f'Task {task_id}: initial pre-process data step!', MessageLevel.Info)

    data = pre_process_data(params.input_data)

    data['Month'] = data['ds'].apply(lambda x: x.month)
    data['Day'] = data['ds'].apply(lambda x: x.weekday())
    data['Hour'] = data['ds'].apply(lambda x: x.hour)
    #data['Closed'] = data['y'].apply(lambda x: myfuncs.closed(x))

    if data['y'].size <= SAMPLE_TEST_SIZE:
        raise MyException(f'Training data size ({data["y"].size} hours) too small (must be at least {SAMPLE_TEST_SIZE + 1} hours)')

    # remove outliers step
    log_message(f'Task {task_id}: remove (detect) outliers step!', MessageLevel.Info)

    outliers, filtered_values = myfuncs.streming_outlier_detection(data['y'].values, k=K, s=S, alpha=ALPHA)
    data['filtered_values'] = filtered_values

    # prepare data for training/testing
    X = data[['Month','Day','Hour']]

    log_message(f'Task {task_id}: starting model training!', MessageLevel.Info)

    X_train = X[:-SAMPLE_TEST_SIZE]
    X_test = X[-SAMPLE_TEST_SIZE:]
    y_train = data[:-SAMPLE_TEST_SIZE]['filtered_values']
    y_test = data[-SAMPLE_TEST_SIZE:]['y']

    #print(X_train)
    #print(y_train)
    #df.to_csv(sep=' ', index=False, header=False)
    #X_train.to_csv('X_train.txt', sep=' ', mode='a')
    #y_train.to_csv('y_train.txt', sep=' ', mode='a')

    # create model instance
    hgbr_model = HistGradientBoostingRegressor()

    # train model
    hgbr_model.fit(X_train, y_train)

    log_message(f'Task {task_id}: model trained!', MessageLevel.Success)

    # get model params
    model_params = hgbr_model.get_params(deep=True)

    # predict
    hgbr_model_forecast = hgbr_model.predict(X_test)

    # model evaluation metrics
    log_message(f'Task {task_id}: getting model evaluation metrics!', MessageLevel.Info)

    hgbr_model_metrics = myfuncs.model_metrics(y_test, hgbr_model_forecast)

    # parameters
    outlier_detection_parameters = {'k': K, 's': S, 'alpha': ALPHA}

    custom_train_parameters = {}

    return {'model': hgbr_model, 
            'model_parameters': model_params,    
            'forecast': hgbr_model_forecast, 
            'metrics': hgbr_model_metrics, 
            'outlier_detection_parameters': outlier_detection_parameters,
            'custom_train_parameters': custom_train_parameters,
            'report_dir_name': None
            }

def train_mlp(params: TrainIn, task_id: int) -> dict:
    """
    Trains a MLPRegressor model.
    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    """

    K = 53
    S = 168
    ALPHA = 2
    MAX_N_ITERATIONS = 1000

    # initial pre-process data
    log_message(f'Task {task_id}: initial pre-process data step!', MessageLevel.Info)

    data = pre_process_data(params.input_data)

    # remove outliers step
    log_message(f'Task {task_id}: remove (detect) outliers step!', MessageLevel.Info)

    outliers, filtered_values = myfuncs.streming_outlier_detection(data['y'].values, k=K, s=S, alpha=ALPHA)
    data['filtered_values'] = filtered_values

    #settings.file_logger.info(str(data['y'].size))
    #settings.file_logger.info(str(data['filtered_values'].size))

    # additional pre-process data step
    log_message(f'Task {task_id}: additional pre-process data step!', MessageLevel.Info)

    X, y = myfuncs.to_supervised(data['filtered_values'].values, n_lags=params.n_lags, n_output=params.forecast_period)

    log_message(f'Task {task_id}: starting model training!', MessageLevel.Info)
    # prepare data for training/testing
    X_train = X[:-1]
    y_train = y[:-1]
    X_test = X[-1].reshape(1,-1)
    y_test = y[-1].reshape(1,-1)

    #X_train[0].tofile('X_train_mlp.txt', sep=',')
    #y_train[0].tofile('y_train_mlp.txt', sep=',')

    #X_train.to_csv('X_train_mlp.txt', sep=' ')
    #y_train.to_csv('y_train_mlp.txt', sep=' ')

    """
    settings.file_logger.info(str(X.shape))
    settings.file_logger.info(str(y.shape))

    settings.file_logger.info(X)
    settings.file_logger.info(y)

    settings.file_logger.info(X_train)
    settings.file_logger.info(y_train)
    settings.file_logger.info(X_test)
    settings.file_logger.info(y_test)

    settings.file_logger.info(str(X_train.shape))
    settings.file_logger.info(str(y_train.shape))
    settings.file_logger.info(str(X_test.shape))
    settings.file_logger.info(str(y_test.shape))
    """

    # create model
    mlp_model = MLPRegressor(max_iter=MAX_N_ITERATIONS)

    # train model
    mlp_model.fit(X_train, y_train)

    log_message(f'Task {task_id}: model trained!', MessageLevel.Success)

    # get model params
    model_params = mlp_model.get_params(deep=True)

    # predict
    mlp_model_forecast = mlp_model.predict(X_test)

    # model evaluation metrics
    log_message(f'Task {task_id}: getting model evaluation metrics!', MessageLevel.Info)

    mlp_model_metrics = myfuncs.model_metrics(y_test.flatten(), mlp_model_forecast.flatten())

    outlier_detection_params = {'k': K, 's': S, 'alpha': ALPHA}

    train_params = {'max_number_iterations': MAX_N_ITERATIONS}

    return {'model': mlp_model, 
            'model_parameters': model_params,    
            'forecast': mlp_model_forecast, 
            'metrics': mlp_model_metrics, 
            'outlier_detection_parameters': outlier_detection_params,
            'custom_train_parameters': train_params,
            'report_dir_name': None
            }

"""
    Input data: ds, x, y
        ds -> datetime
        y1 -> number of customers
        y2 -> number of cashiers that sold items (caixas que venderam artigos)
"""
def train_mlp_multivariate(params: TrainIn, task_id: int) -> dict:
    """
    Trains a MLPRegressor model to forecast customer flow and team size simultaneously.
    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    """

    K = 53
    S = 168
    ALPHA = 2
    MAX_N_ITERATIONS = 1000
    forecast_period = params.forecast_period

    # initial pre-process data
    log_message(f'Task {task_id}: initial pre-process data step!', MessageLevel.Info)

    data = pre_process_data_multivariate(params.input_data)

    # remove outliers step
    log_message(f'Task {task_id}: remove (detect) outliers step!', MessageLevel.Info)

    train_data = data[:-forecast_period].copy()
    test_data = data[-forecast_period:].copy()

    outliers, filtered_values = myfuncs.streming_outlier_detection(train_data['y1'].values, k=K, s=S, alpha=ALPHA)
    train_data['y1_filtered_values'] = filtered_values

    outliers, filtered_values = myfuncs.streming_outlier_detection(train_data['y2'].values, k=K, s=S, alpha=ALPHA)
    train_data['y2_filtered_values'] = filtered_values    

    # additional pre-process data step
    log_message(f'Task {task_id}: additional pre-process data step!', MessageLevel.Info)

    X, y = myfuncs.to_supervised_multivariate(  
                           train_data['y1_filtered_values'].values
                         , train_data['y2_filtered_values'].values
                         , n_lags=params.n_lags
                         , n_output=params.forecast_period)

    log_message(f'Task {task_id}: starting model training!', MessageLevel.Info)

    # create model
    mlp_model = MLPRegressor(max_iter=MAX_N_ITERATIONS)

    # train model
    mlp_model.fit(X, y)

    log_message(f'Task {task_id}: model trained!', MessageLevel.Success)

    # get model params
    model_params = mlp_model.get_params(deep=True)

    # predict
    y1_last_known = train_data['y1'][-forecast_period:].tolist()
    y2_last_known = train_data['y2'][-forecast_period:].tolist()

    aux = []

    for j in range(0, len(y1_last_known), 1):
        aux += [y1_last_known[j]]
        aux += [y2_last_known[j]]
        
    y_last_known = np.asarray([aux])
    y_last_known.reshape(1, -1)

    mlp_model_forecast = mlp_model.predict(y_last_known)

    # remove negative forecasted values
    mlp_model_forecast = mlp_model_forecast.flatten().tolist()
    mlp_model_forecast = [0 if x < 0 else x for x in mlp_model_forecast]

    # model evaluation metrics
    log_message(f'Task {task_id}: getting model evaluation metrics!', MessageLevel.Info)

    tickets_pred = []
    team_size_pred = []
    
    for j in range(0, len(mlp_model_forecast), 2):
        tickets_pred += [mlp_model_forecast[j]]
        team_size_pred += [mlp_model_forecast[j+1]]

    ind = 0

    # in order to compare results without bias from closed bussiness hours/days added to the data, 
    # we need to remove certain values, i.e. we need to put zeros in certain forecasted values.
    team_size_pred_with_zeros_round = []
    team_size_pred_with_zeros_ceil = []
    team_size_pred_with_zeros_floor = []
    
    tickets_pred_with_zeros_round = []
    tickets_pred_with_zeros_ceil = []
    tickets_pred_with_zeros_floor = []

    """
    equal = [0, 0, 0]
    plus_1 = [0, 0, 0]
    minus_1 = [0, 0, 0]
    gt_1 = [0, 0, 0]
    lt_1 = [0, 0, 0]    
    """

    team_size_pred_with_zeros = []
    tickets_pred_with_zeros = []

    for row in test_data.itertuples():
        # team size
        if row.y2 > 0:
            team_size_pred_with_zeros += [team_size_pred[ind]]
        else:
            team_size_pred_with_zeros += [0]

        el = team_size_pred_with_zeros[ind]

        _round = round(el)
        _ceil = ceil(el)
        _floor = floor(el)

        team_size_pred_with_zeros_round.append(_round)
        team_size_pred_with_zeros_ceil.append(_ceil)
        team_size_pred_with_zeros_floor.append(_floor)

        # tickets
        if row.y1 > 0:
            tickets_pred_with_zeros += [tickets_pred[ind]]
        else:
            tickets_pred_with_zeros += [0]
            
        el = tickets_pred_with_zeros[ind]

        _round = round(el)
        _ceil = ceil(el)
        _floor = floor(el)

        tickets_pred_with_zeros_round.append(_round)
        tickets_pred_with_zeros_ceil.append(_ceil)
        tickets_pred_with_zeros_floor.append(_floor)

        ind += 1
        
        """
        _round = round(el)
        _ceil = ceil(el)
        _floor = floor(el)

        v = row.y2
        
        if v == _round:
            equal[0] += 1
        elif v == _round + 1:
            plus_1[0] += 1
        elif v == _round - 1:
            minus_1[0] += 1
        elif _round + 1 > v:
            gt_1[0] += 1
        elif _round - 1 < v:
            lt_1[0] += 1
            
        if v == _floor:
            equal[1] += 1
        elif v == _floor + 1:
            plus_1[1] += 1
        elif v == _floor - 1:
            minus_1[1] += 1
        elif _floor + 1 > v:
            gt_1[1] += 1
        elif _floor - 1 < v:
            lt_1[1] += 1
            
        if v == _ceil:
            equal[2] += 1
        elif v == _ceil + 1:
            plus_1[2] += 1
        elif v == _ceil - 1:
            minus_1[2] += 1
        elif _ceil + 1 > v:
            gt_1[2] += 1
        elif _ceil - 1 < v:
            lt_1[2] += 1
        """ 

    tickets_raw_model_metrics = myfuncs.model_metrics(test_data['y1'], tickets_pred_with_zeros)
    tickets_round_model_metrics = myfuncs.model_metrics(test_data['y1'], tickets_pred_with_zeros_round)
    tickets_ceil_model_metrics = myfuncs.model_metrics(test_data['y1'], tickets_pred_with_zeros_ceil)
    tickets_floor_model_metrics = myfuncs.model_metrics(test_data['y1'], tickets_pred_with_zeros_floor)

    team_size_raw_model_metrics = myfuncs.model_metrics(test_data['y2'], team_size_pred_with_zeros)
    team_size_round_model_metrics = myfuncs.model_metrics(test_data['y2'], team_size_pred_with_zeros_round)
    team_size_ceil_model_metrics = myfuncs.model_metrics(test_data['y2'], team_size_pred_with_zeros_ceil)
    team_size_floor_model_metrics = myfuncs.model_metrics(test_data['y2'], team_size_pred_with_zeros_floor)

    metrics = {}
    metrics['tickets_raw'] = tickets_raw_model_metrics
    metrics['tickets_round'] = tickets_round_model_metrics
    metrics['tickets_ceil'] = tickets_ceil_model_metrics
    metrics['tickets_floor'] = tickets_floor_model_metrics

    metrics['team_size_raw'] = team_size_raw_model_metrics
    metrics['team_size_round'] = team_size_round_model_metrics
    metrics['team_size_ceil'] = team_size_ceil_model_metrics
    metrics['team_size_floor'] = team_size_floor_model_metrics

    outlier_detection_params = {'k': K, 's': S, 'alpha': ALPHA}

    train_params = {'max_number_iterations': MAX_N_ITERATIONS}

    return {'model': mlp_model, 
            'model_parameters': model_params,    
            'forecast': mlp_model_forecast, 
            'metrics': metrics, 
            'outlier_detection_parameters': outlier_detection_params,
            'custom_train_parameters': train_params,
            'report_dir_name': None
            }

def train_prophet(params: TrainIn, task_id: int) -> dict:
    """
    Trains a Prophet model.

    Prophet: Automatic Forecasting Procedure
        https://github.com/facebook/prophet
    """

    K = 53
    S = 168
    ALPHA = 2

    # initial pre-process data step
    log_message(f'Task {task_id}: initial pre-process data step!', MessageLevel.Info)
    data = pre_process_data(params.input_data)

    # remove (detect) outliers step
    log_message(f'Task {task_id}: remove (detect) outliers step!', MessageLevel.Info)
    outliers, filtered_values = myfuncs.streming_outlier_detection(data['y'].values, k=K, s=S, alpha=ALPHA)

    data['y'] = filtered_values
    N = data['y'].size

    # create model
    model = Prophet()

    log_message(f'Task {task_id}: starting model training!', MessageLevel.Info)

    # train model
    model.fit(data)

    log_message(f'Task {task_id}: model trained!', MessageLevel.Success)

    # cross validation step

    #initial = str(N-720*2) + ' hours'
    initial = str(N-params.forecast_period*2) + ' hours'
    period = f'{params.forecast_period} hours'
    horizon = f'{params.forecast_period} hours'

    #df_cv = cross_validation(model, initial=initial, period='720 hours', horizon='720 hours')
    df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)

    log_message(f'Task {task_id}: model cross validation step finished!', MessageLevel.Info)

    # model performance metrics
    log_message(f'Task {task_id}: getting model performance metrics!', MessageLevel.Info)
    
    df_p = performance_metrics(df_cv)

    # create images for the html report
    client_id = params.client_id

    # create dir to store the images for the report in the web server
    dir = settings.IMG_WEBSERVER_DIR + '/' + client_id

    if not os.path.exists(dir):
        os.mkdir(dir)

    REPORT_DIR_NAME = get_html_report_dir()

    dir += '/' + REPORT_DIR_NAME

    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        log_message(f'Task {task_id}: {dir} already exists!', MessageLevel.Critical)

    forecast_imgs = None

    # prophet metrics plots
    MAE_PLOT_IMG_NAME = 'mae.png'

    metrics_imgs = {'imgs': [
                       {'img_name': MAE_PLOT_IMG_NAME, 'title': 'MAE'}
                    ],
                    'title': 'Error/Performance Metrics'
                }

    # fig.savefig('sales.png', transparent=False, dpi=80, bbox_inches="tight")
    plot_cross_validation_metric(df_cv, metric='mae').savefig(dir + '/' + MAE_PLOT_IMG_NAME, backend='Agg', transparent=False, dpi=300, bbox_inches="tight")
    
    # model components plots
    MODEL_COMPONENTS_PLOT_IMG_NAME = 'model_components.png'

    model_components_imgs = {'imgs': [
                       {'img_name': MODEL_COMPONENTS_PLOT_IMG_NAME, 'title': 'Model Components'}
                    ],
                    'title': 'Model Components'
                }

    log_message(f'Task {task_id}: getting model components!', MessageLevel.Info)

    #future = model.make_future_dataframe(periods=720, freq='H', include_history=False)
    periods = params.forecast_period

    future = model.make_future_dataframe(periods=periods, freq='H', include_history=False)
    f = model.predict(future)
    model.plot_components(f).savefig(dir + '/' + MODEL_COMPONENTS_PLOT_IMG_NAME, backend='Agg', transparent=False, dpi=300, bbox_inches="tight")
    
    return {'model': model,
            'model_parameters': {},
            'forecast': {},
            'metrics': {},
            'outlier_detection_parameters': {},
            'custom_train_parameters': {},
            'forecast_imgs': forecast_imgs,
            'metrics_imgs': metrics_imgs,
            'model_components_imgs': model_components_imgs,
            'report_dir_name': REPORT_DIR_NAME,
            }

# --------------------------------------------------------------------
# 
# --------------------------------------------------------------------

def switch(model_type: str, params: TrainIn, task_id: int) -> Union[dict, None]:
    """
	Chooses which type of model to train for a request.
	"""

    if ForecastModels.MLPRegressor.value == model_type:
        if params.customer_flow_and_team_size:
            return train_mlp_multivariate(params, task_id)
        else:
            return train_mlp(params, task_id)
    elif ForecastModels.HistGradientBoostingRegressor.value == model_type:
        return train_hgbr(params, task_id)
    elif ForecastModels.Prophet.value == model_type:
        return train_prophet(params, task_id)
    elif ForecastModels.RandomForestRegressor.value == model_type:
        return train_RandomForestRegressor(params, task_id)

    return None

def train_model(db: Session, client_pkey: int, model_type: str, params: TrainIn, task_id: int) -> bool:
    """
	Trains a model.
	"""

    model_stored_in_db = None
    model_stored_in_hd = None

    try:
        # update the task state in the db
        update_result = crud.update_task_state(db, task_id, TaskState.Executing, 1, None)

        # update the task state in the log
        log_message(f'Task {task_id} {TaskState.Executing}!', MessageLevel.Info)

        # train the model
        train_output = switch(model_type, params, task_id)

        # if the model was trained successfully
        if train_output != None:
            client_id = params.client_id
            model_name = params.model_name

            # get/prepare parameters
            model_storage_name = get_name(model_type, model_name)
            forecast_period = params.forecast_period

            # train output parameters
            model = train_output['model']
            outlier_detection_parameters = train_output['outlier_detection_parameters']
            model_parameters = train_output['model_parameters']
            custom_train_parameters = train_output['custom_train_parameters']
            model_metrics = train_output['metrics']
            forecast = train_output['forecast']

            # html report images
            forecast_imgs = None
            metrics_imgs = None
            model_components_imgs = None

            report_dir_name = None

            # >> (temp) prepare metrics and train parameters to be stored in the db in csv format
            # model metrics
            metrics_csv = ''

            if 'raw' in model_metrics.keys():
                raw_metrics = model_metrics['raw']
                round_metrics = model_metrics['round']
                ceil_metrics = model_metrics['ceil']
                floor_metrics = model_metrics['floor']

                metrics_csv += 'raw mae' + ':' + str(raw_metrics['mae']) + ';'
                metrics_csv += 'round mae' + ':' + str(round_metrics['mae']) + ';'
                metrics_csv += 'ceil mae' + ':' + str(ceil_metrics['mae']) + ';'
                metrics_csv += 'floor mae' + ':' + str(floor_metrics['mae']) + ';'
            elif params.customer_flow_and_team_size:
                metrics_csv += 'tickets raw mae' + ':' + str(model_metrics['tickets_raw']['mae']) + ';'
                metrics_csv += 'tickets round mae' + ':' + str(model_metrics['tickets_round']['mae']) + ';'
                metrics_csv += 'tickets ceil mae' + ':' + str(model_metrics['tickets_ceil']['mae']) + ';'
                metrics_csv += 'tickets floor mae' + ':' + str(model_metrics['tickets_floor']['mae']) + ';'

                metrics_csv += 'team size raw mae' + ':' + str(model_metrics['team_size_raw']['mae']) + ';'
                metrics_csv += 'team size tickets round mae' + ':' + str(model_metrics['team_size_round']['mae']) + ';'
                metrics_csv += 'team size tickets ceil mae' + ':' + str(model_metrics['team_size_ceil']['mae']) + ';'
                metrics_csv += 'team size tickets floor mae' + ':' + str(model_metrics['team_size_floor']['mae']) + ';'
            else:
                for key, value in model_metrics.items():
                    metrics_csv += key + ':' + str(value) + ';'

            # model custom train parameters
            custom_train_parameters_csv = ''

            if custom_train_parameters is not None:
                for key, value in custom_train_parameters.items():
                    custom_train_parameters_csv += key + ':' + str(value) + ';'
            # << 

            # model file extension on disk
            SAVED_MODEL_FILE_EXTENSION = '.joblib'

            if model_type == ForecastModels.Prophet:
                SAVED_MODEL_FILE_EXTENSION = '.json'

                # html report images
                forecast_imgs = train_output['forecast_imgs']
                metrics_imgs = train_output['metrics_imgs']
                model_components_imgs = train_output['model_components_imgs']

                # html report images physical path on disk
                report_dir_name = train_output['report_dir_name']

            # create new trained model record in the db
            model_params = None

            log_message(f'Task {task_id}: storing the model info in the DB.', MessageLevel.Info)

            if model_type == ForecastModels.MLPRegressor:
                model_params = schemas.ModelCreate(type = model_type \
			        , model_name = model_name \
			        , storage_name = model_storage_name + SAVED_MODEL_FILE_EXTENSION \
			        , metrics = metrics_csv \
			        , forecast_period = params.forecast_period \
                    , n_lags = params.n_lags \
			        , train_params = custom_train_parameters_csv
		        )
            else:
                model_params = schemas.ModelCreate(type = model_type \
                    , model_name = model_name \
                    , storage_name = model_storage_name + SAVED_MODEL_FILE_EXTENSION \
                    , metrics = metrics_csv \
                    , forecast_period = params.forecast_period \
                    , train_params = custom_train_parameters_csv
                )

            # update the db
            model_in_db = crud.create_model(db, model_params, client_pkey)

            if model_in_db:
                model_stored_in_db = model_in_db.id

            log_message(f'Task {task_id}: creating the html report.', MessageLevel.Info)

            # create html report
            html_report = create_html_report(
                model_in_db=model_in_db,
                outlier_detection_parameters=outlier_detection_parameters,
                model_parameters=model_parameters,
                custom_train_parameters=custom_train_parameters,
                model_metrics=model_metrics,
                forecast_imgs=forecast_imgs,
                metrics_imgs=metrics_imgs,
                model_components_imgs=model_components_imgs,
                client_id=client_id,
                report_dir_name=report_dir_name,
                customer_flow_and_team_size=params.customer_flow_and_team_size
                )

            # update html report in db
            update_output = crud.update_model_html_report(db, html_report, model_in_db.id)

            # store trained model in hd
            dir = get_dir(settings.MODELS_STORAGE_DIR, client_id)

            log_message(f'Task {task_id}: storing the model in the HD.', MessageLevel.Info)

            if model_type == ForecastModels.Prophet:
                if store_using_json(model, dir, model_storage_name):
                    model_stored_in_hd = dir + model_storage_name + SAVED_MODEL_FILE_EXTENSION
            else:
                if store(model, dir, model_storage_name):
                    model_stored_in_hd = dir + model_storage_name + SAVED_MODEL_FILE_EXTENSION

            # update the state of the task (task finished)
            update_result = crud.update_task_state(db, task_id, TaskState.Finished, 0, model_in_db.id)

            log_message(f'Task {task_id} {TaskState.Finished}!', MessageLevel.Success)
            return True
        # an error occurred
        else:
            # update the state of the task in the db
            update_result = crud.update_task_state(db, task_id, TaskState.Error, -1, None)

            # save error to log file
            log_message(f'Task {task_id} {TaskState.Error}: equals(train_output, None)!', MessageLevel.Critical)

            # creates an error report and stores it in the db
            error_report = '<div><h3>' + f'Task {task_id} {TaskState.Error}: train model output is None!' + '</h3></div>'
            update_result = crud.update_task_error_report(db, error_report, task_id)
            
            # update log
            log_message(f'Task {task_id} {TaskState.Error}: error report created!', MessageLevel.Critical)
    # an unexpected exception occurred
    except Exception as e:
        # creates an error report and stores it in the db
        error_report = '<div><h3>' + f'Task {task_id} {TaskState.Error}: {e}!' + '</h3></div>'
        update_result = crud.update_task_error_report(db, error_report, task_id)

        # log error to the log file
        log_message(f'Task {task_id} {TaskState.Error}: error report created!', MessageLevel.Critical)
        log_message(f'Task {task_id} {TaskState.Error}: {e}', MessageLevel.Critical)

        # update the state of the task in the db
        update_result = crud.update_task_state(db, task_id, TaskState.Error, -1, None)

        # cleanup (remove intermediate artifacts that have been created)
        report_contents_dir = None

        if report_dir_name != None:
            report_contents_dir = settings.IMG_WEBSERVER_DIR + '/' + client_id + '/' + report_dir_name

        cleanup(task_id, model_stored_in_db, model_stored_in_hd, report_contents_dir, db)
    finally:
        pass

    return False

# --------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------

@router.post('', status_code=HTTPStatus.ACCEPTED)
async def train_model_request(model_type: str, params: TrainIn, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> dict:

    # 1. check if the client id is known (registered)
    client_id = params.client_id

    client = crud.get_client_registration_key(db, client_id)

    if not client:
        # log error
        log_message(f'Client {client_id} not found!', MessageLevel.Critical)

        # send error message
        raise HTTPException(status_code=404, detail=f'Client {client_id} not found!')

    client_pkey = client.pkey

    # 2. check if the model type requested is available (supported)
    if not is_model_available(model_type):
        # log error
        log_message(f'Model type {model_type} not available!', MessageLevel.Critical)

        # send error message
        raise HTTPException(status_code=404, detail=f'Model type {model_type} not available!')

    # validate input data.
    keys = params.input_data.keys()

    if model_type == ForecastModels.RandomForestRegressor:
        if len(keys) != 3:
            log_message(f'Client {client_id} model {model_type}. Unexpected number of keys in input data structure {keys} ({len(keys)}) expected (3)!', MessageLevel.Critical)
            raise HTTPException(status_code=400, detail=f'Unexpected input data structure!')

        if 'ds' not in keys or 'x' not in keys or 'y' not in keys:
            log_message(f'Client {client_id} model {model_type}. Unexpected input data structure (expected key ds, x or y not found) {keys}!', MessageLevel.Critical)
            raise HTTPException(status_code=400, detail=f'Unexpected input data structure!')
    elif model_type == ForecastModels.MLPRegressor and params.customer_flow_and_team_size:
        if len(keys) != 3:
            log_message(f'Client {client_id} model {model_type}. Unexpected number of keys in input data structure {keys} ({len(keys)}) expected (3)!', MessageLevel.Critical)
            raise HTTPException(status_code=400, detail=f'Unexpected input data structure!')

        if 'ds' not in keys or 'y1' not in keys or 'y2' not in keys:
            log_message(f'Client {client_id} model {model_type}. Unexpected input data structure (expected key ds, y1 or y2 not found) {keys}!', MessageLevel.Critical)
            raise HTTPException(status_code=400, detail=f'Unexpected input data structure!')
    else:
        if len(keys) != 2:
            log_message(f'Client {client_id} model {model_type}. Unexpected number of keys in input data structure {keys} ({len(keys)}) expected (2)!', MessageLevel.Critical)
            raise HTTPException(status_code=400, detail=f'Unexpected input data structure!')

        if 'ds' not in keys or 'y' not in keys:
            log_message(f'Client {client_id} model {model_type}. Unexpected input data structure (expected key (ds, y) not found) {keys}!', MessageLevel.Critical)
            raise HTTPException(status_code=400, detail=f'Unexpected input data structure!')

    # 3. create task
    task = schemas.TTaskCreate(client_pkey=client_pkey, model_type=model_type, state=TaskState.Pending)
    task = crud.create_task(db, task)

    # log message
    log_message(f'Task {task.id} for client {client_id} ({client_pkey}) train model {model_type} created!', MessageLevel.Info)

    # 4. create background task to train the model
    background_tasks.add_task(train_model, db, client_pkey=client_pkey, model_type=model_type, params=params, task_id=task.id)

    # 5. reply to the client (task created)
    return {'detail': '1', 'task_id': task.id}