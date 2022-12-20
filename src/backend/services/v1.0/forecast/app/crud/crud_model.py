from sqlalchemy.orm import Session
from typing import Any

from app.db import models

def get_model_details(db: Session, model_id: int, client_pkey: int) -> Any:
    return db.query(
                      models.Model.type.label('type') \
                    , models.Model.storage_name.label('storage_name') \
                    , models.Model.forecast_period.label('forecast_period')
            ).filter(
                    models.Model.id == model_id, 
                    models.Model.client_pkey == client_pkey
            ).first()
