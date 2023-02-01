""" 
	WFM integration web service.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022

	Services:
		insert, update, get and delete the parameters associated with a client.
	
	APIs:
		/{client_id}/parameters
"""

from fastapi import APIRouter, HTTPException, Header, Depends
from typing import Union

from models.models import ParametersIn, ParametersOut

router = APIRouter()

@router.put('/{client_id}/parameters')
async def insert_parameters(client_id: str, payload: ParametersIn) -> str:
    """
	Inserts the parameters associated with a client.
    """

    return {'detail' : 'Not implemented yet!'}

@router.get('/{client_id}/parameters', response_model=ParametersOut)
async def get_parameters(client_id: str) -> ParametersOut:
    """
	Gets the parameters associated with a client.
    """

    return {'detail' : 'Not implemented yet!'}

@router.post('/{client_id}/parameters')
async def update_parameters(client_id: str, payload: ParametersIn) -> str:
    """
	Updates the parameters associated with a client.
    """

    return {'detail' : 'Not implemented yet!'}

@router.delete('/{client_id}/parameters')
async def delete_parameters(client_id: str) -> str:
    """
	Deletes the parameters associated with a client.
    """

    return {'detail' : 'Not implemented yet!'}
