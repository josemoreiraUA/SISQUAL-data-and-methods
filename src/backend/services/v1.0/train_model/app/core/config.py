import pathlib

from pydantic import AnyHttpUrl, BaseSettings, EmailStr, validator
from typing import List, Optional, Union

ROOT = pathlib.Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"

    # list of allowed origins
    # e.g: '["http://localhost", "http://localhost:4200", "http://localhost:3000", \
    # "http://localhost:8080", "http://local.dockertoolbox.tiangolo.com"]'
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:8001",
    ]

    # Origins that match this regex or are in the above list are allowed
    BACKEND_CORS_ORIGIN_REGEX: Optional[
        str
    ] = "https.*\.(sisqual.app|sisqual.com)"

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    DATABASE_NAME = 'sql_app.db'
    SQLALCHEMY_DB_URI = 'sqlite:///'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///./' + DATABASE_NAME
    #SQLALCHEMY_DATABASE_URI = "postgresql://user:password@postgresserver/db"
    JWT_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYmYiOjE2NjY2MzMxMjYsImV4cCI6MTg1NjAyMTkyNiwiaXNzIjoiU0lTUVVBTFdGTSM3MDAxIiwiYXVkIjoiRm9yZWNhc3RNYW5hZ2VyIn0.qii3Q-Ocp1YDs9ASTiZNEzFNiDu7Ia3ZxOCUNRQgJ_o'
    MODELS_STORAGE_DIR = 'models/ml/'

    class Config:
        case_sensitive = True

settings = Settings()
