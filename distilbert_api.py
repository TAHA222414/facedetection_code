from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# ----------- MODEL LOADING -----------
model_dir = r"D:\fyp\distilbert\distilbert_3label_priority_20250915_224504"
tok = DistilBertTokenizerFast.from_pretrained(model_dir)
model = DistilBertForSequenceClassification.from_pretrained(model_dir)
model.eval()

# ----------- API SETUP -----------
app = FastAPI(title="DistilBERT Priority API")

class NoticeInput(BaseModel):
    text: str

@app.post("/predict")
async def predict_priority(item: NoticeInput):
    inputs = tok(item.text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
    priority = model.config.id2label[predicted_class_id]
    return {"priority": priority}
