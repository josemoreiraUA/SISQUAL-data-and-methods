from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

connection_uri = settings.db.SQLALCHEMY_DATABASE_URI
if connection_uri.startswith("postgres://"):
    connection_uri = connection_uri.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    connection_uri,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
