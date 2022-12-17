from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import pathlib

#PATH = pathlib.Path(__file__).resolve().parent.parent
current_file_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent
#print(PATH)
#print(PATH2)

#SQLALCHEMY_DATABASE_URL = "sqlite:///.../train_model/app/sql_app.db"
SQLALCHEMY_DATABASE_URL = 'sqlite:///' + str(current_file_path) + '\\train_model\\app\\sql_app.db'
#print(SQLALCHEMY_DATABASE_URL)
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

#("check_same_thread": False) => (sqlite only)
db_engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
	connect_args={"check_same_thread": False},
	echo=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

Base = declarative_base()

#models.Base.metadata.create_all(bind=db_engine)