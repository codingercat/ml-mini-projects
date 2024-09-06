from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Create an instance of FastAPI
app = FastAPI()

# Load the trained model and scaler
with open('models/diabetes_model.pkl', 'rb') as f:
    diabetes_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the input schema for the API
class DiabetesInput(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: float

# API endpoint to get prediction
@app.post("/predict")
def predict_diabetes(data: DiabetesInput):
    try:
        # Prepare input data for prediction
        input_data = np.array([[data.pregnancies, data.glucose, data.blood_pressure, data.skin_thickness, data.insulin, data.bmi, data.diabetes_pedigree_function, data.age]])
        scaled_data = scaler.transform(input_data)

        # Make prediction
        prediction = diabetes_model.predict(scaled_data)

        # Return the result (0: Non-diabetic, 1: Diabetic)
        result = "Diabetic" if prediction[0] == 1 else "Non-diabetic"
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Run Uvicorn for local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
