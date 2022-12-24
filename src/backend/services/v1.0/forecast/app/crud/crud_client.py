""" 
    Database clients table operations.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022
"""

from sqlalchemy.orm import Session
from typing import Union

from app.db import models

def get_client_pkey(db: Session, client_id: str) -> Union[int, None]:

    client = db.query(models.Client.pkey).filter(models.Client.id == client_id).first()

    if not client:
        return None

    return client.pkey
