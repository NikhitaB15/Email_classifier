from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import spacy
import re
from pii_utils import mask_pii, demask_email

from model_utils import load_model, predict_category
import json

app = FastAPI()

# Load models at startup
nlp = spacy.load("en_core_web_sm")
model, tokenizer, label_encoder, max_length = load_model()

class EmailRequest(BaseModel):
    email_body: str

class MaskedEntity(BaseModel):
    position: List[int]
    classification: str
    entity: str

class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[MaskedEntity]
    masked_email: str
    category_of_the_email: str

@app.post("/classify", response_model=EmailResponse)
async def classify_email(request: EmailRequest):
    # Mask PII
    masked_result = mask_pii(request.email_body)
    
    # Classify
    category, confidence = predict_category(
        masked_result['masked_text'], 
        model, tokenizer, label_encoder, max_length
    )
    
    # Prepare response
    response = {
        "input_email_body": request.email_body,
        "list_of_masked_entities": masked_result['entities'],
        "masked_email": masked_result['masked_text'],
        "category_of_the_email": category
    }
    
    return response

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
