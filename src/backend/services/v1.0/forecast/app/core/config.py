""" 
    Service global configuration (settings) class.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
    authors: jd
    version: 1.0
    date:    29/11/2022
"""

import pathlib
import sys
import logging
import datetime

from pydantic import AnyHttpUrl, BaseSettings, EmailStr, validator
from typing import List, Optional, Union
import loguru
from loguru import logger

# global configuration (settings)
class Settings(BaseSettings):
    # endpoints versioning prefix
    API_V1_STR: str = "/api/v1"

    # hard drive models storage location
    ROOT_DIR: str = str(pathlib.Path(__file__).resolve().parent.parent.parent.parent)
    MODELS_DIR: str = ROOT_DIR + '\\train_model\\app\\models\\ml\\'

    # database information
    DATABASE_NAME: str = 'sql_app.db'
    SQLALCHEMY_DATABASE_URI: str = 'sqlite:///' + ROOT_DIR + '\\train_model\\app\\' + DATABASE_NAME

    # authorization token
    JWT_TOKEN: str = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYmYiOjE2NjY2MzMxMjYsImV4cCI6MTg1NjAyMTkyNiwiaXNzIjoiU0lTUVVBTFdGTSM3MDAxIiwiYXVkIjoiRm9yZWNhc3RNYW5hZ2VyIn0.qii3Q-Ocp1YDs9ASTiZNEzFNiDu7Ia3ZxOCUNRQgJ_o'

    # allowed external connections
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        'http://localhost:3000', 
		'http://127.0.0.1:3000'
    ]

    BACKEND_CORS_ORIGIN_REGEX: Optional[str] = None

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    current_time = datetime.datetime.now()
    #print(type(current_time))

    LOG_FILENAME: str = 'forecast_API_Log_' + str(current_time.day) + '_' + str(current_time.month) + '_' + str(current_time.year) + '.log'
    LOG_FILE_DIR: str = 'logs/'

    # file logger
    file_logger: loguru._logger.Logger = None
    # console logger
    console_logger: loguru._logger.Logger = None

    class Config:
        case_sensitive = True

# logging settings
def setup_app_logging(config: Settings) -> None:
    #'format':      '{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name: ^15}:{function: ^15}:{line: >3} | {message}', 
    log_config = {
        'handlers': [
            {'sink': sys.stdout, 
             'filter':      lambda record: record["extra"].get("name") == "clogger", 
             'format':      '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level: <8} | <blue>{name}:{function}:{line}</blue> | {message}', 
             'level':       logging.INFO
            },
            {'sink': config.LOG_FILE_DIR + config.LOG_FILENAME, 
             'filter':      lambda record: record["extra"].get("name") == "flogger", 
             'format':      '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}', 
             'level':       logging.INFO, 
             'rotation':    '1 MB', 
             'enqueue':     True
            },
        ]
    }

    logger.configure(**log_config)

    config.file_logger = logger.bind(name="flogger")
    config.console_logger = logger.bind(name="clogger")

settings = Settings()
