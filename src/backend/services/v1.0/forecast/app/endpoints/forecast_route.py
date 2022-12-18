""" 
	WFM Forecast integration web service.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022

	Services:
		provides forecasts using different forecast models.
	
	APIs:
		/api/v1/app/models/{model_id}/forecast/

	Design Options:
	
		DO1. The model is loaded from disk in every call.
		Preloading the models in memory is another option but might not scale well.

		DO2. Models are identified using an id.

	Designed Issues:
		
		A possible weakness of this API design is that if we are using 
		a data model for in/out we have to make sure that it works for 
		every model! That is to say that somehow, every model has the 
		same in/out parameters.
"""

#import pathlib
import numpy as np

from joblib import load
from sklearn.neural_network import MLPRegressor
from fastapi import APIRouter, HTTPException, Header, Depends
from typing import Union

from sqlalchemy.orm import Session

# custom imports

from models.models import ForecastIn, ForecastOut, ForecastModels2
from core.config import settings
from db import crud#, models, schemas
from dependencies.db import get_db

#models_path = '../train_app/models/ml/'
#current_file_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent
#models_path = str(current_file_path) + '\\train_model\\app\\models\\ml\\'

router = APIRouter()

async def load_mlp_model(model_location_on_disk: str) -> MLPRegressor:
    """
	Loads a stored MLPRegressor model.
	"""

    mlp_model = load(model_location_on_disk)
    return mlp_model

async def mlp_forecast(params: ForecastIn, model_storage_name: str, forecast_period: int) -> ForecastOut:
    """
	Gets a forecast using a stored MLPRegressor model.
	"""

    client_id = params.client_id
    #forecast_period = params.forecast_period    # TODO: should come from the db instead.

    # load imput forecast data
    forecast_imput_data = np.asarray(params.model_imput_data)

    if forecast_imput_data.size != forecast_period:
        raise HTTPException(
		        status_code=400, 
				detail=f'Unexpected imput data size: {forecast_imput_data.size} for defined forecast period: {forecast_period}!'
			)

    # preprocess imput forecast data
    forecast_imput_data = forecast_imput_data.reshape(1,-1)

    # load model
    mlp_model = await load_mlp_model(settings.MODELS_DIR + client_id + '\\' + model_storage_name)

    # forecast
    forecast = mlp_model.predict(forecast_imput_data)

    # postprocess forecast
    forecast = forecast.flatten().tolist()
    forecast = [0 if x < 0 else x for x in forecast]
	
    # return forecast
    # TODO: define what are the values/parameters in outparams.
    return ForecastOut(forecast=forecast, outparams=[len(forecast), 0, 0, 0])

async def get_model_forecast(model_type: str, params: ForecastIn, model_storage_name: str, forecast_period: int) -> Union[ForecastOut, None]:
    """
	Gets a forecast according to the model type.
	"""

    if ForecastModels2.MLPRegressor.value == model_type:
        return await mlp_forecast(params, model_storage_name, forecast_period)

    return None

@router.post('')
async def get_forecast(model_id: int, params: ForecastIn, db: Session = Depends(get_db)):
    """
	Forecast.
	"""

    client_pkey = crud.get_client_pkey(db, params.client_id)

    if not client_pkey:
        raise HTTPException(
		        status_code=404, 
				detail=f'Client {params.client_id} not found!'
			)

    model = crud.get_model_details(db, model_id, client_pkey)

    if not model:
        raise HTTPException(
		        status_code=404, 
				detail=f'Model {model_id} not found for client {params.client_id}!'
			)

    if model.forecast_period != params.forecast_period:
        raise HTTPException(
		        status_code=400, 
				detail=f'Unexpected forecast period: {params.forecast_period}!'
			)

    forecast = await get_model_forecast(model.type, params, model.storage_name, model.forecast_period)

    if forecast == None:
        raise HTTPException(
				status_code=500, 
				detail='An unexpected error occured!'
			)

    return forecast
