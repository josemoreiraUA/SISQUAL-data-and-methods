import pathlib

from pydantic import AnyHttpUrl, BaseSettings, EmailStr, validator
from typing import List, Optional, Union

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent.parent)
    DATABASE_NAME = 'sql_app.db'
    MODELS_DIR = ROOT_DIR + '\\train_model\\app\\models\\ml\\'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + ROOT_DIR + '\\train_model\\app\\' + DATABASE_NAME
    JWT_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYmYiOjE2NjY2MzMxMjYsImV4cCI6MTg1NjAyMTkyNiwiaXNzIjoiU0lTUVVBTFdGTSM3MDAxIiwiYXVkIjoiRm9yZWNhc3RNYW5hZ2VyIn0.qii3Q-Ocp1YDs9ASTiZNEzFNiDu7Ia3ZxOCUNRQgJ_o'

    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        'http://localhost:3000', 
		'http://127.0.0.1:3000'
    ]

    # origins that match this regex or are in the above list are allowed
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

    class Config:
        case_sensitive = True

settings = Settings()
