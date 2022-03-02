from pydantic import BaseModel


class History(BaseModel):
    store_id: str
    history: str
    startDate: str
    endDate: str
    isEmpty: bool
    nSamplesDay: int
