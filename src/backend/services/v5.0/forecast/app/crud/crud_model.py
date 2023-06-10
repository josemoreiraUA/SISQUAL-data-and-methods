""" 
    Database models table operations.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022
"""

from sqlalchemy.orm import Session
from typing import Any

from app.db import models

def get_model_details(db: Session, model_id: int, client_pkey: int) -> Any:
    return db.query(
                      models.Model.type.label('type') \
                    , models.Model.storage_name.label('storage_name') \
                    , models.Model.type_forecast.label('type_forecast') \
                    , models.Model.forecast_period.label('forecast_period')
            ).filter(
                    models.Model.id == model_id, 
                    models.Model.client_pkey == client_pkey
            ).first()
