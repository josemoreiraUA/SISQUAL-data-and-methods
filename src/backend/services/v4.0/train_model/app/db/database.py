from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

db_engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI, 
	connect_args={"check_same_thread": False}
)

"""
db_engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI, 
	connect_args={"check_same_thread": False},
	echo=True
)
"""

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

Base = declarative_base()
