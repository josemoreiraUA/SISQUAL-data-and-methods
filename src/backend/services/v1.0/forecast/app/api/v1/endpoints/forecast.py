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

from joblib import load
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from fastapi import APIRouter, HTTPException, Header, Depends
from typing import Union

from sqlalchemy.orm import Session

# custom imports

from app.models.forecast import ForecastIn, ForecastOut, ForecastModels
from app.core.config import settings#, global_file_logger#, global_console_logger
from app.crud import crud_client, crud_model
from app.api.dependencies.db import get_db

router = APIRouter()

async def load_model(model_location_on_disk: str) -> Union[MLPRegressor, HistGradientBoostingRegressor]:
    """
	Loads a stored model.
	"""

    mlp_model = load(model_location_on_disk)
    return mlp_model

async def mlp_forecast(params: ForecastIn, model_storage_name: str, forecast_period: int) -> ForecastOut:
    """
	Gets a forecast using a stored MLPRegressor model.
	"""

    client_id = params.client_id

    # load input forecast data
    forecast_input_data = np.asarray(params.model_input_data)

    if forecast_input_data.size != forecast_period:
        raise HTTPException(
		        status_code=400, 
				detail=f'Unexpected imput data size: {forecast_input_data.size} for defined forecast period: {forecast_period}!'
			)

    # preprocess imput forecast data
    forecast_input_data = forecast_input_data.reshape(1,-1)

    # load model
    mlp_model = await load_model(settings.MODELS_DIR + client_id + '\\' + model_storage_name)

    # forecast
    forecast = mlp_model.predict(forecast_input_data)

    # postprocess forecast
    forecast = forecast.flatten().tolist()
    forecast = [0 if x < 0 else x for x in forecast]
	
    #return ForecastOut(forecast=forecast, outparams=[len(forecast)])
    return ForecastOut(forecast=forecast)

async def hgbr_forecast(params: ForecastIn, model_storage_name: str, forecast_period: int) -> ForecastOut:
    """
    Gets a forecast using a stored MLPRegressor model.
    """

    client_id = params.client_id

    # load input forecast data
    forecast_input_data = pd.DataFrame(params.model_input_data)

    if forecast_input_data.Hour.size != forecast_period:
        raise HTTPException(
                status_code=400, 
                detail=f'Unexpected imput data size: {forecast_input_data.Hour.size} for defined forecast period: {forecast_period}!'
            )

    # load model
    mlp_model = await load_model(settings.MODELS_DIR + client_id + '\\' + model_storage_name)

    # forecast
    forecast = mlp_model.predict(forecast_input_data)

    # postprocess forecast
    forecast = forecast.flatten().tolist()
    forecast = [0 if x < 0 else x for x in forecast]
    
    #return ForecastOut(forecast=forecast, outparams=[len(forecast)])
    return ForecastOut(forecast=forecast)

async def get_model_forecast(model_type: str, params: ForecastIn, model_storage_name: str, forecast_period: int) -> Union[ForecastOut, None]:
    """
	Gets a forecast according to the model type.
	"""

    if ForecastModels.MLPRegressor.value == model_type:
        return await mlp_forecast(params, model_storage_name, forecast_period)
    elif ForecastModels.HistGradientBoostingRegressor.value == model_type:
        return await hgbr_forecast(params, model_storage_name, forecast_period)

    settings.file_logger.critical(f'Model {model_type} not supported!')
    return None

@router.post('', response_model=ForecastOut)
async def get_forecast(model_id: int, params: ForecastIn, db: Session = Depends(get_db)) -> ForecastOut:
    """
	Forecast.
	"""

    # log info about the forecast.

    settings.file_logger.info(f'Forecast for client {params.client_id} model {model_id}.')

    # validate forecast request.

    client_pkey = crud_client.get_client_pkey(db, params.client_id)

    if not client_pkey:
        settings.file_logger.error(f'Client {params.client_id} not found!')
        raise HTTPException(
		        status_code=404, 
				detail=f'Client {params.client_id} not found!'
			)

    model = crud_model.get_model_details(db, model_id, client_pkey)

    if not model:
        settings.file_logger.critical(f'Model {model_id} not found for client {params.client_id}!')
        raise HTTPException(
		        status_code=404, 
				detail=f'Model {model_id} not found for client {params.client_id}!'
			)

    if model.forecast_period != params.forecast_period:
        settings.file_logger.error(f'Unexpected forecast period: {params.forecast_period}!')
        raise HTTPException(
		        status_code=400, 
				detail=f'Unexpected forecast period: {params.forecast_period}!'
			)

    # get forecast.

    forecast = await get_model_forecast(model.type, params, model.storage_name, model.forecast_period)

    # no forecast.

    if forecast == None:
        raise HTTPException(
				status_code=500, 
				detail='An unexpected error occured!'
			)

    return forecast
