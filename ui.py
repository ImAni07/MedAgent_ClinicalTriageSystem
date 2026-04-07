# ui.py - Gradio UI for MedAgent Clinical Triage System

# Import Requirements
import gradio as gr
import requests

# Works inside Docker / HuggingFace
BASE_URL = "http://localhost:8000"

def run_triage(age, symptoms):
    
    try:
        
        # Convert symptoms input → list
        symptoms_list = [s.strip() for s in symptoms.split(",") if s.strip()]

        # Step 1: Reset environment
        reset_res = requests.post(f"{BASE_URL}/reset").json()

        # Step 2: Send action (you can later replace with LLM)
        action = {
            "risk_level": "medium",
            "decision": "book_consult"
        }

        step_res = requests.post(f"{BASE_URL}/step", json=action).json()

        # OpenEnv response structure → step_res["data"]
        obs = step_res.get("data", {})

        predicted = obs.get("predicted_disease", "N/A")
        feedback = obs.get("feedback", "N/A")
        reward = obs.get("reward", "N/A")

        return f"""
🧾 Symptoms: {symptoms_list}

🧠 Predicted Disease: {predicted}

💬 Feedback: {feedback}

🏆 Reward: {reward}
"""

    except Exception as e:
        return f"❌ Error: {str(e)}"


# UI Design
demo = gr.Interface(
    fn=run_triage,
    inputs=[
        gr.Number(label="Age"),
        gr.Textbox(label="Symptoms (comma-separated)")
    ],
    outputs=gr.Textbox(label="Output"),
    title="🧠 MedAgent Clinical Triage System",
    description="Enter patient details to get AI-powered triage decision"
)

# Run UI
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)