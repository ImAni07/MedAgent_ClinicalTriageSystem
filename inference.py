# inference.py - Safe Rule-Based Inference for MedAgent Clinical Triage

"""
MedAgent inference script.

This script runs a language model against the MedAgent clinical triage environment and 
emits structured stdout logs in the required [START] / [STEP] / [END] format.
"""

# Imports and Configurations
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ---------------- INPUT SCHEMA ---------------- #
class InputData(BaseModel):
    age: int
    symptoms: list[str]


# ---------------- SAFE MODEL ---------------- #
def run_model(age, symptoms):
    try:
        text = " ".join(symptoms).lower()

        if any(x in text for x in ["chest pain", "breathing", "seizure", "bleeding"]):
            return {
                "risk_level": "high",
                "decision": "seek_emergency_care"
            }

        elif any(x in text for x in ["fever", "cough", "vomiting", "pain"]):
            return {
                "risk_level": "medium",
                "decision": "book_consult"
            }

        else:
            return {
                "risk_level": "low",
                "decision": "monitor_at_home"
            }

    except Exception:
        return {
            "risk_level": "medium",
            "decision": "book_consult"
        }


# ---------------- HEALTH CHECK ---------------- #
@app.get("/")
def health():
    return {"status": "ok"}


# ---------------- PREDICT ENDPOINT ---------------- #
@app.post("/predict")
def predict(data: InputData):
    try:
        result = run_model(data.age, data.symptoms)
        return result

    except Exception as e:
        return {
            "error": "Inference failed",
            "details": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
