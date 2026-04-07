# inference.py - Safe Rule-Based Inference for MedAgent Clinical Triage

"""
MedAgent inference script.

This script runs a language model against the MedAgent clinical triage environment and 
emits structured stdout logs in the required [START] / [STEP] / [END] format.

Validator-safe inference script for the MedAgent clinical triage environment.
"""

# Import Required Libraries and Modules
from __future__ import annotations
import asyncio
import json
import os
import re
import time
from typing import Any
from openai import OpenAI
from client import MedAgentAction, MedAgentEnv

TASK_NAME = "clinical-triage"
BENCHMARK = "medagent"
EPISODE_SEED = int(os.getenv("MEDAGENT_SEED", "42"))
MAX_STEPS = max(1, int(os.getenv("MEDAGENT_MAX_STEPS", "1")))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("MEDAGENT_SUCCESS_SCORE_THRESHOLD", "0.7"))

API_BASE_URL = os.getenv("API_BASE_URL", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "").strip()

API_KEY = (
    os.getenv("HF_TOKEN", "").strip()
    or os.getenv("OPENAI_API_KEY", "").strip()
    or os.getenv("API_KEY", "").strip()
)

IMAGE_NAME = os.getenv("IMAGE_NAME", "").strip() or os.getenv("LOCAL_IMAGE_NAME", "").strip()
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "").strip()
PORT = os.getenv("PORT", "7860").strip() or "7860"

VALID_RISK_LEVELS = {"low", "medium", "high"}

VALID_DECISIONS = {
    "monitor_at_home",
    "book_consult",
    "seek_emergency_care",
}

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    
    error_value = error if error else "null"
    
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

def safe_error(error: Exception | str) -> str:
    
    text = str(error).strip() if isinstance(error, Exception) else str(error).strip()
    
    if not text:
        text = error.__class__.__name__ if isinstance(error, Exception) else "unknown_error"
    
    text = re.sub(r"\s+", "_", text)
    
    return text[:160]

def candidate_base_urls() -> list[str]:
    
    candidates = [
        ENV_BASE_URL,
        f"http://127.0.0.1:{PORT}",
        "http://127.0.0.1:7860",
        "http://127.0.0.1:8000",
    ]
    
    unique: list[str] = []
    seen: set[str] = set()
    
    for candidate in candidates:
        
        if candidate and candidate not in seen:
            
            seen.add(candidate)
            unique.append(candidate)
    
    return unique

def build_prompt(age: int, symptoms: list[str]) -> str:
    
    symptom_text = ", ".join(symptoms) if symptoms else "none reported"
    
    return (
        "You are a clinical triage assistant for a hackathon environment.\n"
        "Return JSON only with exactly two keys: risk_level and decision.\n"
        'Valid risk_level values: "low", "medium", "high".\n'
        'Valid decision values: "monitor_at_home", "book_consult", "seek_emergency_care".\n'
        "Be conservative about urgent symptoms.\n\n"
        f"Patient age: {age}\n"
        f"Symptoms: {symptom_text}\n"
    )

def fallback_action(age: int, symptoms: list[str]) -> MedAgentAction:
    
    text = " ".join(symptoms).lower()
    
    high_markers = (
        "chest pain",
        "shortness of breath",
        "difficulty breathing",
        "breathing",
        "seizure",
        "seizures",
        "bleeding",
        "confusion",
        "paralysis",
    )
    
    medium_markers = (
        "fever",
        "cough",
        "vomiting",
        "abdominal pain",
        "pain",
        "dizziness",
        "weakness",
        "rash",
    )

    if any(marker in text for marker in high_markers):
        risk_level = "high"
    
    elif any(marker in text for marker in medium_markers):
        risk_level = "medium"
    
    else:
        risk_level = "low"

    if age >= 70 and risk_level == "medium":
        risk_level = "high"
    
    elif age <= 12 and risk_level == "low" and any(marker in text for marker in ("fever", "vomiting", "rash")):
        risk_level = "medium"

    decision = {
        "low": "monitor_at_home",
        "medium": "book_consult",
        "high": "seek_emergency_care",
    }[risk_level]

    return MedAgentAction(risk_level=risk_level, decision=decision)

def extract_json_object(text: str) -> dict[str, Any] | None:
    
    text = text.strip()
    
    if not text:
        return None

    try:
        
        parsed = json.loads(text)
        
        return parsed if isinstance(parsed, dict) else None
    
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    
    if not match:
        return None

    try:
        
        parsed = json.loads(match.group(0))
        
        return parsed if isinstance(parsed, dict) else None
    
    except Exception:
        return None


def sanitize_action(data: dict[str, Any] | None, fallback: MedAgentAction) -> MedAgentAction:
    
    payload = data or {}
    risk_level = str(payload.get("risk_level", fallback.risk_level)).strip().lower()
    decision = str(payload.get("decision", fallback.decision)).strip().lower()

    if risk_level not in VALID_RISK_LEVELS:
        risk_level = fallback.risk_level
    
    if decision not in VALID_DECISIONS:
        decision = fallback.decision

    return MedAgentAction(risk_level=risk_level, decision=decision)

def llm_action(age: int, symptoms: list[str]) -> MedAgentAction:
    
    fallback = fallback_action(age, symptoms)

    if not (API_BASE_URL and MODEL_NAME and API_KEY):
        return fallback

    try:
        
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY, timeout=20.0)
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Respond with valid JSON only.",
                },
                {
                    "role": "user",
                    "content": build_prompt(age, symptoms),
                },
            ],
            temperature=0.2,
            max_tokens=120,
        )
        content = completion.choices[0].message.content or ""
        parsed = extract_json_object(content)
        return sanitize_action(parsed, fallback)
    
    except Exception:
        return fallback


def create_env() -> Any:
    
    if IMAGE_NAME:
        return asyncio.run(MedAgentEnv.from_docker_image(IMAGE_NAME)).sync()

    errors: list[str] = []
    deadline = time.time() + 45.0

    while time.time() < deadline:
        
        for base_url in candidate_base_urls():
            
            try:
                
                env = MedAgentEnv(
                    base_url=base_url,
                    connect_timeout_s=10.0,
                    message_timeout_s=60.0,
                ).sync()
                env.connect()
                return env
            
            except Exception as exc:
                errors.append(f"{base_url}:{safe_error(exc)}")
        
        time.sleep(2.0)

    raise RuntimeError("env_connect_failed_" + "|".join(errors[-3:]))

def action_to_json(action: MedAgentAction) -> str:
    
    return json.dumps(
        {
            "risk_level": action.risk_level,
            "decision": action.decision,
        },
        separators=(",", ":"),
    )

def main() -> int:
    
    model_label = MODEL_NAME or "rule-based-fallback"
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env: Any = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=model_label)

    try:
        
        env = create_env()
        reset_result = env.reset(seed=EPISODE_SEED)
        observation = reset_result.observation

        for step in range(1, MAX_STEPS + 1):
            
            if reset_result.done and step == 1:
                break

            action = llm_action(observation.age, observation.symptoms)
            step_result = env.step(action)
            reward = round(float(step_result.reward or 0.0), 2)
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_to_json(action),
                reward=reward,
                done=bool(step_result.done),
                error=None,
            )

            observation = step_result.observation
            
            if step_result.done:
                break

        score = min(max(sum(rewards), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        
        error_text = safe_error(exc)
        fallback = fallback_action(40, ["fever"])
        
        if steps_taken == 0:
            steps_taken = 1
            log_step(
                step=1,
                action=action_to_json(fallback),
                reward=0.0,
                done=True,
                error=error_text,
            )

    finally:
        
        if env is not None:
            
            try:
                env.close()
            
            except Exception:
                pass
        
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return 0

if __name__ == "__main__":
    
    try:
        raise SystemExit(main())
    
    except SystemExit:
        raise
    
    except Exception:
        
        log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME or "rule-based-fallback")
        fallback = fallback_action(40, ["fever"])
        log_step(
            step=1,
            action=action_to_json(fallback),
            reward=0.0,
            done=True,
            error=safe_error("unexpected_top_level_failure"),
        )
        log_end(success=False, steps=1, score=0.0, rewards=[])
        raise SystemExit(0)
