""" WFM integration web service.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022

	Services:
		insert, update, get and delete the parameters associated with a client.
	
	APIs:
		/parameters/{id}

"""

from fastapi import APIRouter, HTTPException, Header, Depends#, status, Request
from typing import Union

# custom imports
from models.models import ParametersIn, ParametersOut
from security.auth import JWTBearer

router = APIRouter()

@router.post('/{id}', dependencies=[Depends(JWTBearer())])

async def insert_parameters(id: int, payload: ParametersIn) -> str:
    """Inserts the parameters associated with a client."""

    return {'message' : 'Insert not implemented yet!'}

@router.get('/{id}', response_model=ParametersOut, dependencies=[Depends(JWTBearer())])
async def get_parameters(id: int) -> ParametersOut:
    """Gets the parameters associated with a client."""

    return ParametersOut(culture='pt', forecast_period=720)

@router.put('/{id}', dependencies=[Depends(JWTBearer())])
async def update_parameters(id: int, payload: ParametersIn) -> str:
    """Updates the parameters associated with a client."""

    return {'message' : 'Update not implemented yet!'}

@router.delete('/{id}', dependencies=[Depends(JWTBearer())])
async def delete_parameters(id: int) -> str:
    """Deletes the parameters associated with a client."""

    return {'message' : 'Delete not implemented yet!'}
