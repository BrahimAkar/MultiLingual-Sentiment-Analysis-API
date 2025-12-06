from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import predict_sentiment

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Sentiment API is running"}


@app.post("/analyze")
def analyze(data: TextInput):
    try:
        if not data.text.strip() or len(data.text.strip()) < 5:
            raise HTTPException(status_code=400, detail="Text too short to analyze.")

        result = predict_sentiment(data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
