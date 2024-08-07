import fastapi
from pydantic import BaseModel, ValidationError,validator
from typing import Any, Dict, List, Union
import logging
import pandas as pd
from pathlib import Path
import os
import sys

from challenge.model import DelayModel
logger = logging.getLogger(__name__)

# List of allowed values
ALLOWED_AIRLINES = [
    "Aerolineas Argentinas", "Aeromexico", "Air Canada", "Air France",
    "Alitalia", "American Airlines", "Austral", "Avianca",
    "British Airways", "Copa Air", "Delta Air", "Gol Trans",
    "Grupo LATAM", "Iberia", "JetSmart SPA", "K.L.M.",
    "Lacsa", "Latin American Wings", "Oceanair Linhas Aereas",
    "Plus Ultra Lineas Aereas", "Qantas Airways", "Sky Airline", "United Airlines"
]

ALLOWED_FLIGHT_TYPES = ["I", "N"]

ALLOWED_MONTHS = [str(i) for i in range(1, 13)]

class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: str

    @validator('OPERA')
    def validate_opera(cls, v):
        if v not in ALLOWED_AIRLINES:
            raise ValueError(f'Invalid airline: {v}')
        return v

    @validator('TIPOVUELO')
    def validate_tipovuelo(cls, v):
        if v not in ALLOWED_FLIGHT_TYPES:
            raise ValueError(f'Invalid flight type: {v}')
        return v

    @validator('MES')
    def validate_mes(cls, v):
        if v not in ALLOWED_MONTHS:
            raise ValueError(f'Invalid month: {v}')
        return v

class Flights(BaseModel):
    flights: List[Flight]


app = fastapi.FastAPI()


def is_payload_valid(data_payload: Dict[str, Any]) -> bool:
    """Check if the body request comply with the allowed
    values 
    Args:
        data_payload (Dict[str, Any]): Data payload received from request API.

    Returns:
        bool: True if the payload is valid 
    """
    is_valid = False
    try:
        Flights.parse_obj(data_payload)
        is_valid = True
    except ValidationError as err:
        logger.error(err)
    return is_valid

async def fit_model() -> None:
    """ API to fit and create the DelayModel
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '../data/data.csv')
    data = pd.read_csv(filepath_or_buffer=data_path)
    delay_model = DelayModel()
    features_train, target = delay_model.preprocess(data, "delay")
    delay_model.fit(features_train, target)


@app.on_event("startup")
async def startup_event():
    """FastAPI Startup """
    await fit_model()
    logger.info("Model Fitted!")
    logger.info("started app")


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data_payload: Dict[str, List[Dict[str, str]]]) -> Dict[str, Union[str, List[int]]]:
    if not is_payload_valid(data_payload):
        return fastapi.responses.JSONResponse(
            {"error": "unknown column received"}, status_code=400
        )
    delay_model = DelayModel()
    request_df = pd.DataFrame(data_payload["flights"])
    features = delay_model.preprocess(request_df)
    return {"predict": delay_model.predict(features)}