""" 
	Models integration API.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022

	Services:

        Get the list of available/supported models for training.
        Get the details of a model that has been previously trained.
	
	APIs:

        /models
        /models/{model_id}

"""

from fastapi import APIRouter, HTTPException, Depends

from sqlalchemy.orm import Session

from app.dependencies.db import get_db
from app.db.crud import get_model
from app.core.config import settings

router = APIRouter()

@router.get("/sizing")
def get_available_sizing_forecasting_models():
    """
    List of sizing forecasting models available for training.
    """

    return {'models': settings.AVAILABLE_SIZING_FORECASTING_MODELS}

@router.get("/customer/flow/and/sizing")
def get_available_customer_flow_and_sizing_forecasting_models():
    """
    List of customer flow and sizing forecasting models available for training.
    """

    return {'models': settings.AVAILABLE_CUSTOMER_FLOW_AND_SIZING_MODELS}

@router.get("/customer/flow")
def get_available_customer_flow_forecasting_models():
    """
    List of customer flow forecasting models available for training.
    """

    return {'models': settings.AVAILABLE_CUSTOMER_FLOW_FORECASTING_MODELS}

@router.get("/{model_id}")
def get_model_details(model_id: int, db: Session = Depends(get_db)):
    """
    Model details.
    """

    model = get_model(db, model_id)

    if not model:
        raise HTTPException(status_code=404, detail=f'Model {model_id} not found!')

    return model