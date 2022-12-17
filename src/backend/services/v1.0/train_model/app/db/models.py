from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, func, DateTime
from sqlalchemy.orm import relationship

from db.database import Base

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
    train_params = Column(String)
    client_pkey = Column(Integer, ForeignKey("clients.pkey"))

    client = relationship("Client", back_populates="models")

class TrainTask(Base):
    __tablename__ = "traintasks"

    id = Column(Integer, primary_key=True, index=True)
    #time_started = Column(DateTime(timezone=True), server_default=func.now())
    #time_finished = Column(DateTime(timezone=True))
    #client_pkey = Column(Integer, ForeignKey("clients.pkey"))
    #model_id = Column(Integer, ForeignKey("models.id"))
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_started = Column(DateTime(timezone=True))
    time_finished = Column(DateTime(timezone=True))
    client_pkey = Column(Integer, ForeignKey("clients.pkey"))
    model_type = Column(String(100))	
    state = Column(String(25))

#    owner = relationship("Client", back_populates="models")

"""
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)

    items = relationship("Item", back_populates="owner")


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="items")
"""