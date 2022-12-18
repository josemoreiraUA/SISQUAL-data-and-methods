from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status

async def has_authorization(credentials: HTTPAuthorizationCredentials=Depends(HTTPBearer())):
    """ 
	Authorization function.
    """

    jwt_token = credentials.credentials

    if jwt_token != 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYmYiOjE2NjY2MzMxMjYsImV4cCI6MTg1NjAyMTkyNiwiaXNzIjoiU0lTUVVBTFdGTSM3MDAxIiwiYXVkIjoiRm9yZWNhc3RNYW5hZ2VyIn0.qii3Q-Ocp1YDs9ASTiZNEzFNiDu7Ia3ZxOCUNRQgJ_o':
        raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail='Invalid authorization credentials.'
                )
