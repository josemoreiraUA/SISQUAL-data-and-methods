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

from fastapi import APIRouter, HTTPException, Depends#, BackgroundTasks, Request

from sqlalchemy.orm import Session

from app.dependencies.db import get_db
from app.db.crud import get_model
from app.core.config import settings

router = APIRouter()

@router.get("")
def get_available_models():
    """
    List of models available for training.
    """

    return {'models': settings.AVAILABLE_MODELS}

@router.get("/{model_id}")
def get_model_details(model_id: int, db: Session = Depends(get_db)):
    """
    Model details.
    """

    model = get_model(db, model_id)

    if not model:
        raise HTTPException(status_code=404, detail=f'Model {model_id} not found!')

    return model