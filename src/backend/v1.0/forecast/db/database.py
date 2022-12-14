from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///../train_app/sql_app.db"
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

#("check_same_thread": False) => (sqlite only)
db_engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
	connect_args={"check_same_thread": False},
	echo=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

Base = declarative_base()
