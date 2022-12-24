""" 
	Forecast parameters web service integration.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022

	Services:

		insert, update and get the parameters associated with a client.
	
	APIs:
	
		/clients/{client_id}/parameters
"""

from fastapi import APIRouter, HTTPException, Header, Depends#, status, Request
from typing import Union

# custom imports
from models.models import ParametersIn, ParametersOut

router = APIRouter()

@router.post('/clients/{client_id}/parameters')
async def insert_parameters(client_id: int, payload: ParametersIn) -> str:
    """
	Inserts the parameters associated with a client.
	"""

    return {'detail': 'Not implemented yet!'}

@router.get('/clients/{client_id/parameters}', response_model=ParametersOut)
async def get_parameters(client_id: int) -> ParametersOut:
    """
	Gets the parameters associated with a client.
	"""

    return ParametersOut(culture='ac-AC', forecast_period=0)

@router.put('/clients/{client_id}/parameters')
async def update_parameters(client_id: int, payload: ParametersIn) -> str:
    """
	Updates the parameters associated with a client.
	"""

    return {'detail': 'Not implemented yet!'}
