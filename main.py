from datetime import datetime

from transformers import AutoTokenizer, RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class Comment(BaseModel):
    comment_id: int
    user_id: int
    post_id: int
    content: str
    created_at: datetime


def load_model():
    model = RobertaForSequenceClassification.from_pretrained("./models/")
    tokenizer = AutoTokenizer.from_pretrained("./models/")
    return model, tokenizer


def inference(
    model: RobertaForSequenceClassification, tokenizer: AutoTokenizer, text: str
) -> str:
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs: SequenceClassifierOutput = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(dim=-1)
        label = model.config.id2label[predicted_class_idx.item()]
    return label


models, tokenizer = load_model()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
async def predict(comment: Comment):
    label = inference(models, tokenizer, comment.content)
    return {**comment.model_dump(), "label": label}
