""" 
	Tasks integration API.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022

	Services:

        Task details.
        Task state.
	
	APIs:

        /{task_id}
        /{task_id}/state

	Designed options:

		...

    TODOS:


"""

#from typing import Union
from fastapi import APIRouter, HTTPException, Depends#, BackgroundTasks, Request

from sqlalchemy.orm import Session

#from app.models.models import TrainIn, ForecastModels, TrainTaskOut, TaskState, ForecastModels
#import app.tools.auxiliary_functions as myfuncs
from app.dependencies.db import get_db
#from app.db.crud import get_model#, schemas#, models
#from app.core.config import settings
from app.db import crud

router = APIRouter()

@router.get("/{task_id}")
def get_task_details(task_id: int, db: Session = Depends(get_db)):
    task = crud.get_task(db, task_id)

    if not task:
        raise HTTPException(
                status_code=404, 
                detail=f'Task {task_id} not found!'
            )

    return task

@router.get("/{task_id}/state")
def get_task_state(task_id: int, db: Session = Depends(get_db)) -> str:
    task = crud.get_task_state(db, task_id)

    if not task:
        raise HTTPException(
                status_code=404, 
                detail=f'Task {task_id} not found!'
            )

    return {'state': task.state}