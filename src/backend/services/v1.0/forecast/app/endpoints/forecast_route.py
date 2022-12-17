""" 
	WFM integration web service.

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

import pathlib
import numpy as np

from joblib import load
from sklearn.neural_network import MLPRegressor
from fastapi import APIRouter, HTTPException, Header, Depends

from sqlalchemy.orm import Session
from api import deps
from db import crud, models, schemas

# custom imports
from models.models import ForecastIn, ForecastOut, ForecastModels2

#models_path = '../train_app/models/ml/'
current_file_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent

models_path = str(current_file_path) + '\\train_model\\app\\models\\ml\\'
#print(models_path)

router = APIRouter()

async def load_mlp_model(model_name_on_disk : str):
    """
	Loads a stored MLPRegressor model.
	"""

    mlp_model = load(model_name_on_disk)
    return mlp_model

async def mlp_forecast(params : ForecastIn, model_storage_name: str) -> ForecastOut:
    """
	Provides a forecast using a stored MLPRegressor model.
	"""

    client_id = params.client_id
    forecast_period = params.forecast_period

    # load imput forecast data
    forecast_imput_data = np.asarray(params.model_imput_data)

    # preprocess imput forecast data
    forecast_imput_data = forecast_imput_data.reshape(1,-1)

    # load model
    mlp_model = await load_mlp_model(models_path + client_id + '\\' + model_storage_name)

    # forecast
    forecast = mlp_model.predict(forecast_imput_data)

    # postprocess forecast
    forecast = forecast.flatten().tolist()
    forecast = [0 if x < 0 else x for x in forecast]
	
    # return forecast
    return ForecastOut(forecast=forecast, outparams=[len(forecast), 0, 0, 0])

async def switch(id : int, params : ForecastIn, model_storage_name: str) -> {ForecastOut, None}:
    """
	Chooses which type of model to be used for each request.
	"""

    if ForecastModels2.MLPRegressor.value:
        return await mlp_forecast(params, model_storage_name)

    return None

@router.post('')
async def get_forecast(model_id: int, params: ForecastIn, db: Session = Depends(deps.get_db)):
    """
	Forecast.
	"""

    model = crud.get_model(db, model_id)

    if not model:
        raise HTTPException(
		        status_code=404, detail=f'Model {model_id} not found!'
			)

    client = crud.get_client(db, params.client_id)

    if not client:
        raise HTTPException(
		        status_code=404, detail=f'Client {params.client_id} not found!'
			)	

    forecast = await switch(model_id, params, model.storage_name)

    if forecast == None:
        raise HTTPException(status_code=404, detail='An error occured!')

    return forecast
