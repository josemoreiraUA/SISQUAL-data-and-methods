import pathlib
import sys
import logging

from pydantic import AnyHttpUrl, BaseSettings, EmailStr, validator
from typing import List, Optional, Union
from loguru import logger

"""
class LoggingSettings(BaseSettings):
    LOGGING_LEVEL: int = logging.INFO  # logging levels are ints
"""

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    ROOT_DIR: str = str(pathlib.Path(__file__).resolve().parent.parent.parent.parent)
    DATABASE_NAME: str = 'sql_app.db'
    MODELS_DIR: str = ROOT_DIR + '\\train_model\\app\\models\\ml\\'
    SQLALCHEMY_DATABASE_URI: str = 'sqlite:///' + ROOT_DIR + '\\train_model\\app\\' + DATABASE_NAME
    JWT_TOKEN: str = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYmYiOjE2NjY2MzMxMjYsImV4cCI6MTg1NjAyMTkyNiwiaXNzIjoiU0lTUVVBTFdGTSM3MDAxIiwiYXVkIjoiRm9yZWNhc3RNYW5hZ2VyIn0.qii3Q-Ocp1YDs9ASTiZNEzFNiDu7Ia3ZxOCUNRQgJ_o'

    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        'http://localhost:3000', 
		'http://127.0.0.1:3000'
    ]

    # origins that match this regex or are in the above list are allowed
    BACKEND_CORS_ORIGIN_REGEX: Optional[str] = "https.*\.(sisqual.app|sisqual.com)"

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    """
    logging: LoggingSettings = LoggingSettings()

    log_config = {
        'handlers': [
            {'sink': sys.stdout, 
                 'filter':      lambda record: 'gclog' in record['extra'], 
                 'format':      '{time} - {message}', 
                 'level':       logging.LOGGING_LEVEL
                 },
            {'sink': 'forecast.log', 
                 'filter':      lambda record: 'gflog' in record['extra'], 
                 'format':      '{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}', 
                 'level':       logging.LOGGING_LEVEL, 
                 'rotation':    '1 MB', 
                 'enqueue':     True
            },
        ]
    }

    logger.configure(**log_config)

    global_file_logger = logger.bind(gflog=True)
    global_console_logger = logger.bind(gclog=True)

    #global_file_logger: Optional[logger] = None
    #global_console_logger: Optional[logger] = None
    """

    class Config:
        case_sensitive = True

"""
def setup_app_logging(config: Settings) -> None:
    log_config = {
        'handlers': [
            {'sink': sys.stdout, 
                 'filter':      lambda record: 'gclog' in record['extra'], 
                 'format':      '{time} - {message}', 
                 'level':       config.logging.LOGGING_LEVEL
                 },
            {'sink': 'forecast.log', 
                 'filter':      lambda record: 'gflog' in record['extra'], 
                 'format':      '{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}', 
                 'level':       config.logging.LOGGING_LEVEL, 
                 'rotation':    '1 MB', 
                 'enqueue':     True
            },
        ],
        'extra': {}
    }

    logger.configure(**log_config)

    config.global_file_logger = logger.bind(gflog=True)
    config.global_console_logger = logger.bind(gclog=True)

    #logger.info('forecast service startup event.')
    #logger.add("forecast.log", filter=lambda record: "glog" in record["extra"], format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", rotation="1 MB", enqueue=True)
    #logger.add("special.log", filter=lambda record: "special" in record["extra"], enqueue=True)
    #logger.debug("This message is not logged to the file")
    #logger.bind(special=True).info("This message, though, is logged to the file!")
    #logger.add("somefile.log", enqueue=True)
    #logger.bind('This message, though, is logged to the file!')
    #global_logger = logger.bind(glog=True)
    #global_logger.info('forecast service startup event.')

"""

#'format':      '{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name: ^15}:{function: ^15}:{line: >3} | {message}', 

log_config = {
    'handlers': [
        {'sink': sys.stdout, 
             'filter':      lambda record: record["extra"].get("name") == "gclog", 
             'format':      '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level: <8} | <blue>{name}:{function}:{line}</blue> | {message}', 
             'level':       logging.INFO
        },
        {'sink': 'forecast.log', 
             'filter':      lambda record: record["extra"].get("name") == "gflog", 
             'format':      '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}', 
             'level':       logging.INFO, 
             'rotation':    '1 MB', 
             'enqueue':     True
        },
    ]
}

logger.configure(**log_config)

global_file_logger = logger.bind(name="gflog")
global_console_logger = logger.bind(name="gclog")

settings = Settings()
