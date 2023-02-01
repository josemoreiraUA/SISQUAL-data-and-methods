from app.db.database import SessionLocal
from typing import Generator

def get_db() -> Generator:
    db_conn = SessionLocal()
    try:
        yield db_conn
    finally:
        db_conn.close()
