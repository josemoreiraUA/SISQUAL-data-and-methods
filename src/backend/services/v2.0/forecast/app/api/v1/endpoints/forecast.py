""" 
	Forecast integration web service.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022

	Services:

		Provides forecasts using previously trained models.
	
	APIs:

		/models/{model_id}/forecast

	Design Options:
	
		DO1.  Models are loaded from disk in every call.
		      Preloading models in memory is another option but may not scale well.

		DO2.  Models are identified using an id.

	Designed Issues:
		
		We are using generic data structures for in/out parameters. 
        This has to work for any model, since it is assumed that models can have different 
        in/out parameters.

    Log Levels:

        logger.trace("trace message.")
        logger.debug("debug message.")
        logger.info("info message.")
        logger.success("success message.")
        logger.warning("warning message.")
        logger.error("error message.")
        logger.critical("critical message.")
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from joblib import load
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from prophet import Prophet
from prophet.serialize import model_from_json
from fastapi import APIRouter, HTTPException, Header, Depends, Request
from typing import Union

from sqlalchemy.orm import Session

# custom imports
from app.models.forecast import ForecastIn, ForecastOut, ForecastModels, MessageLevel
from app.core.config import settings
from app.crud import crud_client, crud_model
from app.api.dependencies.db import get_db

router = APIRouter()

# --------------------------------------------------------------------
# models loading functions
# --------------------------------------------------------------------

async def load_model(client_id: str, model_id: int, model_location_on_disk: str) -> Union[MLPRegressor, HistGradientBoostingRegressor]:
    """
	Loads a stored model.
	"""

    try:
        model = load(model_location_on_disk)
    except Exception as e:
        log_message(f'Forecast: client {client_id} model {model_id}. Error loading model > {e}!', MessageLevel.Critical)
        raise HTTPException(status_code=500, detail=f'Forecast: client {client_id} model {model_id}. Error loading model!')

    return model

# --------------------------------------------------------------------
# models forecast functions
# --------------------------------------------------------------------

async def prophet_forecast(model_id: int, params: ForecastIn, model_storage_name: str) -> ForecastOut:
    """
    Gets a forecast using a stored Prophet model.
    """

    client_id = params.client_id

    # load input forecast data
    forecast_input_data = pd.DataFrame(params.model_input_data)
    forecast_input_data['ds'] = pd.to_datetime(forecast_input_data['ds'])

    forecast = None

    # load model
    fin = None

    try:
        fin = open(settings.MODELS_DIR + client_id + '\\' + model_storage_name, 'r')
        m = model_from_json(fin.read())
        forecast = m.predict(forecast_input_data)        
    except Exception as e:
        log_message(f'Forecast: client {params.client_id} model {model_id}. Error loading model > {e}!', MessageLevel.Critical)
        raise HTTPException(status_code=500, detail=f'Forecast: client {params.client_id} model {model_id}. Error loading model!')
    finally:
        if fin is not None:
            fin.close()

    #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    # postprocess forecast
    forecast = forecast['yhat'].tolist()
    forecast = [0 if x < 0 else x for x in forecast]
    
    return ForecastOut(ds=forecast_input_data['ds'].astype('string').tolist(), forecast=forecast)

async def mlp_forecast(model_id: int, params: ForecastIn, model_storage_name: str, forecast_period: int) -> ForecastOut:
    """
	Gets a forecast using a stored MLPRegressor model.
	"""

    client_id = params.client_id

    # load input forecast data
    #forecast_input_data = np.asarray(params.model_input_data)
    forecast_input_data = pd.DataFrame(params.model_input_data)

    ds = forecast_input_data['ds'].tolist()
    y = np.asarray(forecast_input_data['y'])

    if y.size != len(ds):
        log_message(f'Forecast: client {params.client_id} model {model_id}. The sizes of the input datatimes and the corresponding observations do not match {y.size} != {len(ds)}!', MessageLevel.Critical)
        raise HTTPException(status_code=400, detail=f'Forecast: client {params.client_id} model {model_id}. The sizes of the input datatimes and the corresponding observations do not match {y.size} != {dlen(ds)}!')

    if y.size != forecast_period:
        log_message(f'Forecast: client {params.client_id} model {model_id}. Unexpected imput data size: {forecast_input_data.Hour.size} for defined forecast period: {forecast_period}!', MessageLevel.Critical)
        raise HTTPException(status_code=400, detail=f'Unexpected imput data size: {forecast_input_data.size} for defined forecast period: {forecast_period}!')

    # preprocess imput forecast data
    forecast_input_data = y.reshape(1,-1)

    # load model
    mlp_model = await load_model(params.client_id, model_id, settings.MODELS_DIR + client_id + '\\' + model_storage_name)

    # forecast
    forecast = mlp_model.predict(forecast_input_data)

    # postprocess forecast
    forecast = forecast.flatten().tolist()
    forecast = [0 if x < 0 else x for x in forecast]
	
    format = "%Y-%m-%d %H:%M:%S"
    ini_datetime = datetime.strptime(ds[len(ds) - 1], format)
    ds = []

    i = 1
    while i <= forecast_period:
        ds += [str(ini_datetime + timedelta(hours = i))]
        i += 1

    return ForecastOut(ds=ds, forecast=forecast)

async def hgbr_forecast(model_id: int, params: ForecastIn, model_storage_name: str, forecast_period: int) -> ForecastOut:
    """
    Gets a forecast using a stored HistGradientBoostingRegressor model.
    """

    client_id = params.client_id

    # load input forecast data
    input_data = pd.DataFrame(params.model_input_data)
    input_data['ds'] = pd.to_datetime(input_data['ds'])

    month = input_data['ds'].apply(lambda x: x.month)
    weekday = input_data['ds'].apply(lambda x: x.weekday())
    hour = input_data['ds'].apply(lambda x: x.hour)

    forecast_input_data = pd.DataFrame.from_dict({'Month': month, 'Day': weekday, 'Hour': hour})

    if len(params.model_input_data['ds']) != forecast_period:
        log_message(f'Forecast: client {params.client_id} model {model_id}. Unexpected imput data size: {len(params.model_input_data["ds"])} for defined forecast period: {forecast_period}!', MessageLevel.Critical)
        raise HTTPException(status_code=400, detail=f'Unexpected imput data size: {len(params.model_input_data["ds"])} for defined forecast period: {forecast_period}!')

    # load model
    mlp_model = await load_model(params.client_id, model_id, settings.MODELS_DIR + client_id + '\\' + model_storage_name)

    # forecast
    forecast = mlp_model.predict(forecast_input_data)

    # postprocess forecast
    forecast = forecast.flatten().tolist()
    forecast = [0 if x < 0 else x for x in forecast]
    
    return ForecastOut(ds=params.model_input_data['ds'], forecast=forecast)

# --------------------------------------------------------------------
# auxiliary functions
# --------------------------------------------------------------------

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

async def get_model_forecast(model_type: str, model_id: int, params: ForecastIn, model_storage_name: str, forecast_period: int) -> Union[ForecastOut, None]:
    """
	Gets a forecast according to the model type.
	"""

    try:
        if ForecastModels.MLPRegressor.value == model_type:
            return await mlp_forecast(model_id, params, model_storage_name, forecast_period)
        elif ForecastModels.HistGradientBoostingRegressor.value == model_type:
            return await hgbr_forecast(model_id, params, model_storage_name, forecast_period)
        elif ForecastModels.Prophet.value == model_type:
            return await prophet_forecast(model_id, params, model_storage_name)
    except Exception as e:
        error_report = f'Forecast: client {params.client_id} model {model_id}. An error occurred: {e}!'
        log_message(error_report, MessageLevel.Critical)
        raise HTTPException(status_code=500, detail=f'Forecast: client {params.client_id} model {model_id}. An error occurred!')
    finally:
        pass

    log_message(f'Forecast: client {params.client_id} model {model_id}. Model {model_type} not supported!', MessageLevel.Critical)
    return None

# --------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------

@router.post('', response_model=ForecastOut)
async def get_forecast(model_id: int, params: ForecastIn, request: Request, db: Session = Depends(get_db)) -> ForecastOut:
    """
	Forecast.
	"""

    # log info about the forecast request.
    log_message(f'Forecast: client {params.client_id} model {model_id}.', MessageLevel.Info)

    # validate forecast request.
    client_pkey = crud_client.get_client_pkey(db, params.client_id)

    if not client_pkey:
        log_message(f'Forecast: client {params.client_id} model {model_id}. Client {params.client_id} not found!', MessageLevel.Critical)
        raise HTTPException(status_code=404, detail=f'Client {params.client_id} not found!')

    model = crud_model.get_model_details(db, model_id, client_pkey)

    if not model:
        log_message(f'Forecast: client {params.client_id} model {model_id}. Model {model_id} not found for client {params.client_id}!', MessageLevel.Critical)
        raise HTTPException(status_code=404, detail=f'Model {model_id} not found for client {params.client_id}!')

    if model.type != ForecastModels.Prophet:
        if model.forecast_period != params.forecast_period:
            log_message(f'Forecast: client {params.client_id} model {model_id}. Unexpected forecast period: received {params.forecast_period} expecting {model.forecast_period}!', MessageLevel.Critical)
            raise HTTPException(status_code=400, detail=f'Unexpected forecast period: received {params.forecast_period} expecting {model.forecast_period}!')

    # get forecast.
    log_message(f'Getting forecast: client {params.client_id} model {model_id}.', MessageLevel.Info)

    forecast = await get_model_forecast(model.type, model_id, params, model.storage_name, model.forecast_period)

    # no forecast.
    if forecast == None:
        log_message(f'Forecast: client {params.client_id} model {model_id}. Unexpected error occured!', MessageLevel.Critical)
        raise HTTPException(status_code=500, detail='An unexpected error occured!')

    # send forecast.
    log_message(f'Sending forecast: client {params.client_id} model {model_id}.', MessageLevel.Info)
    return forecast
