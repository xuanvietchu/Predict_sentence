from fastapi import FastAPI
from pydantic import BaseModel
from infer_bartpho import main as predict, filter_answer, tokenize_input

app = FastAPI()

class InputSentence(BaseModel):
    sent: str

@app.post("/Predict")
def para(item: str):
    text = tokenize_input(item)
    predicts = predict(text)
    for i, _ in enumerate(predicts):
        predicts[i] = predicts[i].replace('_', ' ')

    res = filter_answer(item, predicts)

    return res