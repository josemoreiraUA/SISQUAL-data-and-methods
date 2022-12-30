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
import uuid
#import uuid6

from joblib import dump
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from prophet import Prophet
from prophet.serialize import model_to_json#, model_from_json
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request

from sqlalchemy.orm import Session

from app.models.models import TrainIn, ForecastModels, TrainTaskOut, TaskState, ForecastModels
import app.tools.auxiliary_functions as myfuncs
from app.dependencies.db import get_db
from app.db import crud, schemas#, models
from app.core.config import settings

router = APIRouter()

# --------------------------------------------------------------------

def check_model_type_deprecated(model_type: str) -> bool:
    """Use is_model_available() instead."""

    for model in ForecastModels:
        if model_type == str(model.value):
            return True

    raise HTTPException(
            status_code=404, 
			detail=f'Model of type {model_type} not available!'
        )

def train_mlp_deprecated(params: TrainIn) -> dict:
    """Use train_mlp() instead."""

    data = pd.DataFrame.from_dict(params.input_data)
    nprev = params.forecast_period

    # preprocess data
    data['dt'] = pd.to_datetime(data['dt'])
    data.shape

    forecast_data = pd.DataFrame(data.resample('H', on = 'dt').values.sum())
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

# --------------------------------------------------------------------

# --------------------------------------------------------------------

def model_details_to_html(
    model_in_db, 
    model_details_elem_title: str,
    model_type_elem_title: str,
    model_name_elem_title: str,
    model_time_trained_elem_title: str,
    forecast_period_elem_title: str) -> str:

    html = '<h3>' + model_details_elem_title + '</h3>'
    #html += "<table class='ModelDetails'>"
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
    html += '</tbody>'
    html += '</table>'

    return html

def to_html(data: dict, title: str, round_val: bool = False, num_decimals: int = 2) -> str:
    html = ''

    if data is not None and len(data) > 0:
        #html += '<div>'
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
        #html += '</div>'

    return html

def embed_images_in_html(images: dict, client_id: str, report_uri: str) -> str:

    html = ''
    imgs = images['imgs']
    title = images['title']

    if images is not None and len(images) > 0:
        #html += '<div>'
        html += '<h4>' + title + '</h4>'
        #html += '<table>'
        #html += '<tbody>'

        for img in imgs:
            html += '<img src="' + settings.IMG_WEBSERVER_URI + '/' + client_id + '/' + report_uri + '/' + img['img_name'] + '" alt="' + img['title'] + '" />'

        #html += '</tbody>'
        #html += '</table>'
        #html += '</div>'

    return html

#model_in_db: models.Model,
def create_html_report(
    model_in_db,
    outlier_detection_parameters: dict,
    model_parameters: dict,
    custom_train_parameters: dict,
    model_metrics: dict,
    forecast_imgs: dict,
    metrics_imgs: dict,
    client_id: str,
    report_dir_name: str
    ) -> str:
    """Creates an html report"""

    # report encapsulating div element
    html_report = "<div class='htmlReport'>"

    # title labels
    model_details_elem_title = 'Model Details'
    model_type_elem_title = 'Type'
    model_name_elem_title = 'Name'
    model_time_trained_elem_title = 'Time Trained'
    forecast_period_elem_title = 'Forecast Period'

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
            forecast_period_elem_title=forecast_period_elem_title
        )

    # outlier detection parameters
    html_report += to_html(data=outlier_detection_parameters, title=outlier_detection_parameters_elem_title)

    # model parameters
    html_report += to_html(data=model_parameters, title=model_parameters_elem_title)

    # custom train parameters
    html_report += to_html(data=custom_train_parameters, title=custom_train_parameters_elem_title)

    # model metrics
    html_report += to_html(data=model_metrics, title=model_metrics_elem_title, round_val=True, num_decimals=3)

    # add images
    html_report += embed_images_in_html(metrics_imgs, client_id, report_dir_name)
    html_report += embed_images_in_html(forecast_imgs, client_id, report_dir_name)

    html_report += '</div>'

    return html_report

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

    return model_type + '_' + model_name + '_' + day + '_' + month + '_' + year + '_' + hour + '_' + minute + '_' + second + '_' + microsecond

def store(model: object, dir: str, name: str) -> bool:
    """
	Stores a model.
    """

    dump(model, dir + name + '.joblib')
    return True

def store_using_json(model: object, dir: str, name: str) -> bool:
    """
    Stores a model.
    """
    
    with open(dir + name + '.json', 'w') as fout:
        fout.write(model_to_json(model))

    return True

def pre_process_data(input_data: dict) -> dict:
    """
    Pre-processes the input data for training.
    """

    data = pd.DataFrame.from_dict(input_data)

    data['ds'] = pd.to_datetime(data['ds'])
    data.shape

    #duplicateRows = data[data.duplicated(['ds'])]

    data_resample = pd.DataFrame(data.resample('H', on = 'ds').y.sum())
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
    data['Closed'] = data['y'].apply(lambda x: myfuncs.closed(x))

    outliers, filtered_values = myfuncs.streming_outlier_detection(data['y'].values, k=K, s=S, alpha=ALPHA)
    data['filtered_values'] = filtered_values

    X = data[['Month','Day','Hour']]

    X_train = X[:-SAMPLE_TEST_SIZE]
    X_test = X[-SAMPLE_TEST_SIZE:]
    y_train = data[:-SAMPLE_TEST_SIZE]['filtered_values']
    y_test = data[-SAMPLE_TEST_SIZE:]['y']

    # create model instance
    hgbr_model = HistGradientBoostingRegressor()

    # train model
    hgbr_model.fit(X_train, y_train)

    # get params
    model_params = hgbr_model.get_params(deep=True)

    # predict
    hgbr_model_forecast = hgbr_model.predict(X_test)

    """
    ds = data[-SAMPLE_TEST_SIZE:]['ds']
    print(type(ds))
    ind = ''
    for row in ds:
        ind += '"' + str(row) + '",'

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
    outliers, filtered_values = myfuncs.streming_outlier_detection(data['y'].values, k=K, s=S, alpha=ALPHA)
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

def train_prophet(params: TrainIn) -> dict:
    """
    Trains a Prophet model.

    Prophet: Automatic Forecasting Procedure
        https://github.com/facebook/prophet
    """

    #N_PREV = params.forecast_period
    K = 53
    S = 168
    ALPHA = 2
    #MAX_N_ITERATIONS = 10

    SAMPLE_TEST_SIZE = params.forecast_period

    # pre-process data
    data = pre_process_data(params.input_data)

    # remove outliers
    outliers, filtered_values = myfuncs.streming_outlier_detection(data['y'].values, k=K, s=S, alpha=ALPHA)
    data['y'] = filtered_values

    # create model
    model = Prophet()

    N = data['y'].size
    #print(type(data['ds']))
    #print(data['ds'].size)
    
    #print(data['ds'])

    # TODO: use cross_validation

    #X_train = data['ds']
    #y_train = data['y']

    #d = {X_train, y_train}
    #df = pd.DataFrame(data=d)
    #print(df)

    # train model
    #model.fit(df)
    model.fit(data)

    # method 1
    """
    future = model.make_future_dataframe(periods=720, freq='H', include_history=False)
    fcst = model.predict(future)
    fig = model.plot(fcst)
    fig2 = model.plot_components(fcst)
    """

    #print(type(future))
    #print(type(fcst))
    #print(type(fig))
    #print(future)

    # method 2
    #df_cv = cross_validation(model, cutoffs=cutoffs, horizon='30 days', parallel="processes")
    #df_p = performance_metrics(df_cv, rolling_window=1)
    #42324
    
    #initial = str((8760 * 3) + (8760 / 2)) + ' hours'
    initial = str(N-720*2) + ' hours'
    print(N, N-720)

    df_cv = cross_validation(model, initial=initial, period='720 hours', horizon='720 hours')
    #df_cv = cross_validation(model, initial=initial, period='1 hour', horizon='720 hours')

    #df_cv = cross_validation(model, horizon='30 days')
    #df_p = performance_metrics(df_cv, rolling_window=1)
    df_p = performance_metrics(df_cv)

    
    print(df_p)
    print(df_cv)


    #fig.savefig('plot2.png')
    #fig2.savefig('plot3.png')


    #pd.plotting.register_matplotlib_converters()
    #df.plot(x='ds', y='y', figsize = (12,8));

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html
    #fig = data.plot(x='ds', y='y',kind='line', figsize=(25, 10), fontsize=20).get_figure()

    #fig.savefig('data.png')

    # >>

    # create images for the report

    #IMG_WEBSERVER_DIR> str = 'C:\\temp\\nginx-1.22.1\\nginx-1.22.1\\files'
    #IMG_WEBSERVER_URI> str = 'http:\\127.0.0.1\\files'
    #settings
    client_id = params.client_id

    # create dir to store the images for the report in the web server

    dir = settings.IMG_WEBSERVER_DIR + '\\' + client_id

    if not os.path.exists(dir):
        os.mkdir(dir)

    current_time = datetime.datetime.now()
    day = str(current_time.day)
    month = str(current_time.month)
    year = str(current_time.year)
    hour = str(current_time.hour)
    minute = str(current_time.minute)
    second = str(current_time.second)
    #microsecond = str(current_time.microsecond)

    REPORT_DIR_NAME = 'report_' + day + '_' + month + '_' + year + '_' + hour + '_' + minute + '_' + second + '_' + str(uuid.uuid4())

    dir += '\\' + REPORT_DIR_NAME

    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        err_message = f'Directory {dir} already exists!'
        settings.console_logger.critical(err_message)
        settings.file_logger.critical(err_message)
    # <<

    FORECAST_PLOT_IMG_NAME = 'forecast_plot.png'
    forecast_imgs = {'imgs': [{
                            'img_name': FORECAST_PLOT_IMG_NAME,
                            'title': 'Predictions vs Observations'
                    }],
                    'title': 'Forecast Performance'
                }

    ax = df_cv.plot(x='ds', y='yhat', label='Predictions', legend=True, figsize=(12,8))
    fig = df_cv.plot(x='ds', y='y', label='Observations', legend=True, ax=ax).get_figure()
    fig.savefig(dir + '\\' + FORECAST_PLOT_IMG_NAME)

    #matplotlib.use('agg')
    #mse, rmse, mae, mdape, smape, coverage

    MSE_PLOT_IMG_NAME = 'mse.png'
    RMSE_PLOT_IMG_NAME = 'rmse.png'
    MAE_PLOT_IMG_NAME = 'mae.png'
    MDAPE_PLOT_IMG_NAME = 'mdape.png'
    SMAPE_PLOT_IMG_NAME = 'smape.png'
    COVERAGE_PLOT_IMG_NAME = 'coverage.png'

    metrics_imgs = {'imgs': [
                       {'img_name': MSE_PLOT_IMG_NAME, 'title': 'MSE'},
                       {'img_name': RMSE_PLOT_IMG_NAME, 'title': 'RMSE'},
                       {'img_name': MAE_PLOT_IMG_NAME, 'title': 'MAE'},
                       {'img_name': MDAPE_PLOT_IMG_NAME, 'title': 'MDAPE'},
                       {'img_name': SMAPE_PLOT_IMG_NAME, 'title': 'SMAPE'},
                       {'img_name': COVERAGE_PLOT_IMG_NAME, 'title': 'Coverage'}
                    ],
                    'title': 'Error/Performance Metrics'
                }

    plot_cross_validation_metric(df_cv, metric='mse').savefig(dir + '\\' + MSE_PLOT_IMG_NAME)
    plot_cross_validation_metric(df_cv, metric='rmse').savefig(dir + '\\' + RMSE_PLOT_IMG_NAME)
    plot_cross_validation_metric(df_cv, metric='mae').savefig(dir + '\\' + MAE_PLOT_IMG_NAME)
    plot_cross_validation_metric(df_cv, metric='mdape').savefig(dir + '\\' + MDAPE_PLOT_IMG_NAME)
    plot_cross_validation_metric(df_cv, metric='smape').savefig(dir + '\\' + SMAPE_PLOT_IMG_NAME)
    plot_cross_validation_metric(df_cv, metric='coverage').savefig(dir + '\\' + COVERAGE_PLOT_IMG_NAME)
    #print(type(fig))
    #fig.savefig('mape.png')

    # forecast
    #forecast = model.predict(future)

    # plot the forecast
    #forecast_plot = model.plot(forecast)

    # plot the forecast components
    #fig2 = m.plot_components(forecast)

    """
    X_train = data[:-SAMPLE_TEST_SIZE]['ds']
    X_test = data[-SAMPLE_TEST_SIZE:]['ds']
    y_train = data[:-SAMPLE_TEST_SIZE]['filtered_values']
    y_test = data[-SAMPLE_TEST_SIZE:]['y']

    d = {X_train, y_train}
    df = pd.DataFrame(data=d)

    # train model
    model.fit(df)

    d = {X_test, y_test}
    future = pd.DataFrame(data=d)

    forecast = model.predict(future)
    print(forecast)
    #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    """

    #future = model.make_future_dataframe(periods=30)

    # additional pre-process data step
    #X, y = myfuncs.to_supervised(data['filtered_values'].values, n_lags=N_PREV, n_output=N_PREV)

    # prepare data for training
    #X_train = X[:-1]
    #y_train = y[:-1]
    #X_test = X[-1].reshape(1,-1)
    #y_test = y[-1].reshape(1,-1)

    # create model
    #mlp_model = MLPRegressor(max_iter=MAX_N_ITERATIONS)

    # train model
    #mlp_model.fit(X_train,y_train)

    # get params
    #model_params = mlp_model.get_params(deep=True)

    # predict
    #mlp_model_forecast = mlp_model.predict(X_test)

    # model evaluation metrics
    #mlp_model_metrics = myfuncs.model_metrics(y_test.flatten(), mlp_model_forecast.flatten())

    #outlier_detection_params = {'k': K, 's': S, 'alpha': ALPHA}

    #train_params = {'max_number_iterations': MAX_N_ITERATIONS}

    model_params = {}
    model_forecast = {}
    model_metrics = {}
    outlier_detection_params = {}
    train_params = {}

    return {'model': model, 
            'model_parameters': model_params,    
            'forecast': model_forecast, 
            'metrics': model_metrics, 
            'outlier_detection_parameters': outlier_detection_params,
            'custom_train_parameters': train_params,
            'forecast_imgs': forecast_imgs,
            'metrics_imgs': metrics_imgs,
            'report_dir_name': REPORT_DIR_NAME
            }

def switch(model_type: str, params: TrainIn) -> Union[dict, None]:
    """
	Chooses which type of model to train for a request.
	"""

    if ForecastModels.MLPRegressor.value == model_type:
        return train_mlp(params)
    elif ForecastModels.HistGradientBoostingRegressor.value == model_type:
        return train_hgbr(params)
    elif ForecastModels.Prophet.value == model_type:
        return train_prophet(params)

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
        update_result = crud.update_task_state(db, task_id, TaskState.Executing, 1, None)

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

            forecast_imgs = None
            metrics_imgs = None
            report_dir_name = None

            # >> temp 
            metrics_csv = ''
            for key, value in model_metrics.items():
                metrics_csv += key + ':' + str(value) + ';'

            custom_train_parameters_csv = ''
            for key, value in custom_train_parameters.items():
                custom_train_parameters_csv += key + ':' + str(value) + ';'
            # << 

            SAVED_MODEL_FILE_EXTENSION = '.joblib'

            if model_type == ForecastModels.Prophet:
                SAVED_MODEL_FILE_EXTENSION = '.json'

                forecast_imgs = train_output['forecast_imgs']
                metrics_imgs = train_output['metrics_imgs']
                report_dir_name = train_output['report_dir_name']

            model_params = schemas.ModelCreate(type = model_type \
			    , model_name = model_name \
			    , storage_name = model_storage_name + SAVED_MODEL_FILE_EXTENSION \
			    , metrics = metrics_csv \
			    , forecast_period = params.forecast_period \
			    , train_params = custom_train_parameters_csv
		    )

            model_in_db = crud.create_model(db, model_params, client_pkey)
            if model_in_db:
                is_model_created_in_db = True
                model_db_id = model_in_db.id

            # >> create html report 

            html_report = create_html_report(
                model_in_db=model_in_db,
                outlier_detection_parameters=outlier_detection_parameters,
                model_parameters=model_parameters,
                custom_train_parameters=custom_train_parameters,
                model_metrics=model_metrics,
                forecast_imgs=forecast_imgs,
                metrics_imgs=metrics_imgs,
                client_id=client_id,
                report_dir_name=report_dir_name
                )

            #print(html_report)
            update_output = crud.update_model_html_report(db, html_report, model_in_db.id)
            #update_output = 1 > success
            #print('------------------->', update_output)
            # << 

            # store model in the HD
            dir = get_dir(settings.MODELS_STORAGE_DIR, client_id)

            if model_type == ForecastModels.Prophet:
                if store_using_json(model, dir, model_storage_name):
                    is_model_stored_in_hd = True
                    model_hd_name = dir + model_storage_name + SAVED_MODEL_FILE_EXTENSION
            else:
                if store(model, dir, model_storage_name):
                    is_model_stored_in_hd = True
                    model_hd_name = dir + model_storage_name + SAVED_MODEL_FILE_EXTENSION

            # update the state of the task in the DB
            update_result = crud.update_task_state(db, task_id, TaskState.Finished, 0, model_in_db.id)
            return True
        else:
            # update the state of the task in the DB
            update_result = crud.update_task_state(db, task_id, TaskState.Error, -1, None)
    except Exception as e:
        print('-------------------------->', e)
        # update the state of the task in the DB
        update_result = crud.update_task_state(db, task_id, TaskState.Error, -1, None)
        if is_model_created_in_db:
            # TODO: remove record from the DB
            pass
        if model_hd_name:
            # TODO: remove model from the HD
            pass
    finally:
        pass

    return False

# --------------------------------------------------------------------

# --------------------------------------------------------------------

@router.post('', status_code=HTTPStatus.ACCEPTED)
async def train_model_request(model_type: str, params: TrainIn, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> dict:

    # validation steps

    # 1. check if the client is known (registered)
    client_id = params.client_id

    client = crud.get_client_registration_key(db, client_id)

    if not client:
        # error message
        err_message = f'Client {client_id} not found!'

        # log error
        settings.console_logger.error(err_message)
        settings.file_logger.error(err_message)

        # send error message
        raise HTTPException(
            status_code=404, 
            detail=err_message
        )

    client_pkey = client.pkey

    # 2. check if the model type requested is available (supported)
    if not is_model_available(model_type):
        # error message
        err_message = f'Model of type {model_type} not available!'
    
        # log error
        settings.console_logger.error(err_message)
        settings.file_logger.error(err_message)

        # send error message
        raise HTTPException(
            status_code=404, 
            detail=err_message
        )

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
    return {'detail': '1', 'task_id': task.id}