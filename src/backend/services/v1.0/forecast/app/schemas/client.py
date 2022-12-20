from typing import Optional
from pydantic import BaseModel

class ClientParametersIn(BaseModel):
    culture: str
    is_active: Optional[bool] = False

class ClientBase(BaseModel):
    id: str
    culture: str
    is_active: bool = True

class ClientCreate(ClientBase):
    pass

class Client(ClientBase):
    client_pkey: int

    class Config:
        orm_mode = True
