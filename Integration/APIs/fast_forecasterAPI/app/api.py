from typing import Union

import pandas as pd
import requests
from celery.result import AsyncResult
from fast_forecaster import __version__ as model_version
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from loguru import logger

from app import __version__, schemas
from app.config import settings
from app.tasks import predict_task

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


@api_router.post("/predict/", response_model=schemas.Task, status_code=202)
async def predict(store_id: int, n_output: int) -> Union[JSONResponse, dict]:
    """
    ###Make Time Series predictions with the fast_forecaster XGBoost package!
    """

    # GET the data from the PreProcesser
    payload = {"rosterCode": store_id}
    url = "http://localhost:8001/api/v1/GetHistory"
    response = requests.post(url, params=payload).json()
    input_df = pd.read_json(response["history"], convert_dates=False)
    isEmpty = bool(response["isEmpty"])
    # Check if DataFrame is empty.
    if(not isEmpty):
        # Get the number of samples in a day from the response.
        dataset_day = int(response["nSamplesDay"])  

        logger.info(f"Making prediction on inputs: {input_df}")
        task_id = predict_task.delay(
            input_df.to_json(date_format='iso'), dataset_day, n_output
        )
        task_response = schemas.Task(task_id=str(task_id), status="Processing")
        return task_response.dict()
    return JSONResponse(
        status_code=404,
        content={
            "msg": "DataFrame is empty."
            f" Does the store with ID({store_id}) possess any records from the past 4 years?"
        },
    )


@api_router.get(
    "/predict/result/{task_id}", response_model=schemas.Prediction, status_code=200
)
async def forecasting_results(task_id: str) -> Union[JSONResponse, dict]:
    """Fetch result for given task_id"""
    task = AsyncResult(task_id)
    if not task.ready():
        return JSONResponse(
            status_code=202, content={"task_id": task_id, "status": str(task.status)}
        )
    result = task.get()
    prediction_response = schemas.Prediction(
        task_id=task_id, status="Success", forecast=str(result)
    )
    return prediction_response.dict()
