from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, func, DateTime
from sqlalchemy.orm import relationship

from app.db.database import Base

class Client(Base):
    __tablename__ = "clients"

    pkey = Column(Integer, primary_key=True, index=True)
    id = Column(String, unique=True, index=True)
    culture = Column(String)
    is_active = Column(Boolean, default=True)

    models = relationship("Model", back_populates="client")

class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String)
    model_name = Column(String)
    storage_name = Column(String)
    time_trained = Column(DateTime(timezone=True), server_default=func.now())
    metrics = Column(String)
    forecast_period = Column(Integer)
    n_lags = Column(Integer)
    train_params = Column(String)
    type_forecast = Column(String)
    html_report = Column(String)
    client_pkey = Column(Integer, ForeignKey("clients.pkey"))

    client = relationship("Client", back_populates="models")

class TrainTask(Base):
    __tablename__ = "traintasks"

    id = Column(Integer, primary_key=True, index=True)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_started = Column(DateTime(timezone=True))
    time_finished = Column(DateTime(timezone=True))
    client_pkey = Column(Integer, ForeignKey("clients.pkey"))
    model_type = Column(String(100))
    model_id = Column(Integer)
    state = Column(String(25))
    error_report = Column(String)    
