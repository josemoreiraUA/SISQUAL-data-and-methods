from sqlalchemy.orm import Session
from typing import Union

from app.db import models

def get_client_pkey(db: Session, client_id: str) -> Union[int, None]:

    client = db.query(models.Client.pkey).filter(models.Client.id == client_id).first()

    if not client:
        return None

    return client.pkey
