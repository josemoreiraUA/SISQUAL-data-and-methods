from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

#from db import crud, models, schemas
from core.config import settings

#("check_same_thread": False) => (sqlite only)
db_engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI, 
	connect_args={"check_same_thread": False},
	echo=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

Base = declarative_base()

#models.Base.metadata.create_all(bind=db_engine)