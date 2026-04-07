"""
MedAgent inference script.

This script runs a language model against the MedAgent clinical triage environment and 
emits structured stdout logs in the required [START] / [STEP] / [END] format.
"""

# Import Requirements
from __future__ import annotations
import asyncio
import json
import os
import textwrap
from typing import List, Optional
from openai import OpenAI
from client import MedAgentAction, MedAgentEnv

# Configuration: environment URL, model, and runtime parameters
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MEDAGENT_TASK", "clinical-triage")
BENCHMARK = os.getenv("MEDAGENT_BENCHMARK", "medagent")
EPISODE_SEED = int(os.getenv("MEDAGENT_SEED", "42"))
MAX_STEPS = int(os.getenv("MEDAGENT_MAX_STEPS", "1"))
TEMPERATURE = float(os.getenv("MEDAGENT_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MEDAGENT_MAX_TOKENS", "200"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("MEDAGENT_SUCCESS_THRESHOLD", "0.7"))

VALID_RISK_LEVELS = {"low", "medium", "high"}

VALID_DECISIONS = {
    "monitor_at_home",
    "book_consult",
    "seek_emergency_care",
}

# System prompt guiding the LLM to output structured triage decisions
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a clinical triage agent operating in a hackathon evaluation environment.
    You will receive a patient's age and a list of symptoms.
    Return exactly one JSON object with two keys:
    {
        "risk_level": "low" | "medium" | "high",
        "decision": "monitor_at_home" | "book_consult" | "seek_emergency_care"
    }
    Use higher urgency for red-flag symptoms such as chest pain, breathing difficulty,
    seizures, confusion, paralysis, or major bleeding.
    Output JSON only. No markdown, no explanation.
    """
).strip()

# Logging utilities to output structured evaluation logs

# Log the start of an evaluation episode with task, environment, and model info
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

# Log each step with action, reward, done status, and any errors
def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    
    error_val = error if error else "null"
    
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )

# Log the end of an episode with success status, total steps, final score, and reward trajectory
def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# Prompt builder
def build_user_prompt(age: int, symptoms: List[str], history: List[str]) -> str:
    
    history_block = "\n".join(history[-3:]) if history else "None"
    symptom_text = ", ".join(symptoms)
    
    return textwrap.dedent(
        f"""
        Patient age: {age}
        Symptoms: {symptom_text}
        Previous trajectory:
        {history_block}

        Choose the best triage action and respond with JSON only.
        """
    ).strip()

# JSON Extraction
def _extract_json_object(text: str) -> dict[str, str]:
    
    text = text.strip()
    
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        data = json.loads(snippet)
        
        if isinstance(data, dict):
            return data

    raise ValueError("Model output did not contain a valid JSON object.")

# Validate and clean model output to ensure valid action format
def _sanitize_action_fields(data: dict[str, str]) -> MedAgentAction:
    
    risk_level = str(data.get("risk_level", "")).strip().lower()
    decision = str(data.get("decision", "")).strip().lower()

    if risk_level not in VALID_RISK_LEVELS:
        risk_level = "medium"
    
    if decision not in VALID_DECISIONS:
        decision = "book_consult"

    return MedAgentAction(risk_level=risk_level, decision=decision)

# Rule-based fallback if model fails to ensure robustness
def _fallback_action(symptoms: List[str]) -> MedAgentAction:
    
    symptom_text = " ".join(symptoms).lower()
    
    high_markers = (
        "chest pain",
        "shortness of breath",
        "difficulty breathing",
        "seizure",
        "seizures",
        "confusion",
        "paralysis",
        "bleeding",
    )
    
    medium_markers = (
        "fever",
        "abdominal pain",
        "vomiting",
        "cough",
        "dizziness",
        "weakness",
        "rash",
    )

    if any(marker in symptom_text for marker in high_markers):
        
        return MedAgentAction(
            risk_level="high",
            decision="seek_emergency_care",
        )
    
    if any(marker in symptom_text for marker in medium_markers):
        
        return MedAgentAction(
            risk_level="medium",
            decision="book_consult",
        )
    
    return MedAgentAction(
        risk_level="low",
        decision="monitor_at_home",
    )

# Generate action using LLM, fallback to rule-based logic if needed
def get_model_action(
    client: OpenAI,
    age: int,
    symptoms: List[str],
    history: List[str],
) -> MedAgentAction:
    
    user_prompt = build_user_prompt(age=age, symptoms=symptoms, history=history)
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = (completion.choices[0].message.content or "").strip()
        return _sanitize_action_fields(_extract_json_object(response_text))
    
    except Exception as exc:
        print(f"[DEBUG] Model action generation failed: {exc}", flush=True)
        return _fallback_action(symptoms)

# Environment Creation
async def _create_env() -> MedAgentEnv:
    
    if IMAGE_NAME:
        return await MedAgentEnv.from_docker_image(IMAGE_NAME)

    env = MedAgentEnv(base_url=ENV_BASE_URL)
    await env.connect()
    
    return env

# Main Function
async def main() -> None:
    
    if not API_KEY:
        
        raise RuntimeError(
            "Missing API key. Set HF_TOKEN, OPENAI_API_KEY, or API_KEY before running inference."
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await _create_env()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(seed=EPISODE_SEED)
        last_observation = result.observation

        for step in range(1, MAX_STEPS + 1):
            
            if result.done:
                break

            action = get_model_action(
                client=client,
                age=last_observation.age,
                symptoms=last_observation.symptoms,
                history=history,
            )

            result = await env.step(action)
            observation = result.observation
            reward = float(result.reward or 0.0)
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_observation = observation

            action_str = json.dumps(
                {
                    "risk_level": action.risk_level,
                    "decision": action.decision,
                },
                separators=(",", ":"),
            )
            
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_str} -> reward {reward:.2f}, feedback={observation.feedback}"
            )

            if done:
                break

        score = min(max(sum(rewards), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        
        try:
            await env.close()
        
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)
        
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# Run the script
if __name__ == "__main__":
    asyncio.run(main())