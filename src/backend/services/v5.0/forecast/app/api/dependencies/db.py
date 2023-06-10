""" 
    Database session connection dependency.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022
"""

from app.db.database import SessionLocal
from typing import Generator

def get_db() -> Generator:
    db_conn = SessionLocal()
    try:
        yield db_conn
    finally:
        db_conn.close()
