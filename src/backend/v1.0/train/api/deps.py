from db.database import SessionLocal

def get_db():
    db_conn = SessionLocal()
    try:
        yield db_conn
    finally:
        db_conn.close()
