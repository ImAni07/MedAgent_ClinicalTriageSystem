# inference.py - Safe Rule-Based Inference for MedAgent Clinical Triage

"""
MedAgent inference script.

This script runs a language model against the MedAgent clinical triage environment and 
emits structured stdout logs in the required [START] / [STEP] / [END] format.
"""

# Imports and Configurations
from __future__ import annotations
import asyncio
import json
import os
from typing import List, Optional
from client import MedAgentAction, MedAgentEnv

# CONFIG
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")

TASK_NAME = "clinical-triage"
BENCHMARK = "medagent"
EPISODE_SEED = 42
MAX_STEPS = 1
SUCCESS_SCORE_THRESHOLD = 0.7

# LOGGING
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    
    error_val = error if error else "null"
    
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# SAFE RULE-BASED MODEL
def run_model(symptoms: List[str]) -> MedAgentAction:
    
    """
    This replaces LLM → ALWAYS SAFE
    """
    
    try:
        text = " ".join(symptoms).lower()

        if any(x in text for x in ["chest pain", "breathing", "seizure", "bleeding"]):
            return MedAgentAction("high", "seek_emergency_care")

        elif any(x in text for x in ["fever", "cough", "vomiting", "pain"]):
            return MedAgentAction("medium", "book_consult")

        else:
            return MedAgentAction("low", "monitor_at_home")

    except Exception:
        return MedAgentAction("medium", "book_consult")

# ENV
async def _create_env() -> MedAgentEnv:
    
    try:
        
        if IMAGE_NAME:
            return await MedAgentEnv.from_docker_image(IMAGE_NAME)

        env = MedAgentEnv(base_url=ENV_BASE_URL)
        await env.connect()
        
        return env

    except Exception as e:
        
        print(f"[ERROR] Env creation failed: {e}", flush=True)
        
        raise

# MAIN
async def main() -> None:
    
    env = await _create_env()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model="rule-based")

    try:
        
        result = await env.reset(seed=EPISODE_SEED)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):

            if result.done:
                break

            # SAFE MODEL CALL
            action = run_model(obs.symptoms)

            result = await env.step(action)

            reward = float(result.reward or 0.0)
            done = result.done

            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps({
                "risk_level": action.risk_level,
                "decision": action.decision,
            })

            log_step(step, action_str, reward, done, None)

            if done:
                break

        score = min(max(sum(rewards), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[FATAL ERROR] {e}", flush=True)

    finally:
        
        try:
            await env.close()
        
        except Exception:
            pass

        log_end(success, steps_taken, score, rewards)

# ENTRY
if __name__ == "__main__":
    
    try:
        asyncio.run(main())
    
    except Exception as e:
        print(f"[CRASH] {e}", flush=True)
