""" 
    JWT Token Authorization dependency.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022
"""

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status

from app.core.config import settings

async def has_authorization(authorization: HTTPAuthorizationCredentials=Depends(HTTPBearer())):
    """
	Checks the requests' JWT Bearer authorization.
	"""

    jwt_token = authorization.credentials

    if jwt_token != settings.JWT_TOKEN:
        raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail='Invalid authorization credentials!'
                )
