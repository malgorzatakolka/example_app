# -*- coding: utf-8 -*-

import pandas as pd
import os
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

#Setup environment credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'my-turning-69ded1e770c0.json'
PROJECT = "my-turning-college-project"
REGION = "europ-west1"

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("model")

# Create input/output pydantic models
input_model = create_model("reject_accept_api_input", **{'amount': 8000.0, 'risk_score': 710.0, 'dti': 25.440000534057617, 'state': 'OH', 'emp_length': 12.0})
output_model = create_model("reject_accept_api_output", prediction=1)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
