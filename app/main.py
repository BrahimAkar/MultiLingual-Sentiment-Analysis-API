from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from .model import predict_sentiment

import os

app = FastAPI()


API_KEY =  os.getenv("SECRET_KEY")  
API_KEY_NAME = "X-Internal-Token"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Not authorized :(")
    return api_key



class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Sentiment API is running"}


@app.post("/analyze", dependencies=[Security(verify_api_key)])
def analyze(data: TextInput):
    try:
        if not data.text.strip() or len(data.text.strip()) < 5:
            raise HTTPException(status_code=400, detail="Text too short to analyze.")

        result = predict_sentiment(data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
