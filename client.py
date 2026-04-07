# client.py

# Client for the MedAgent clinical triage environment.

# Import Requirements
from __future__ import annotations
from typing import Any

# Import OpenEnv client base class and response type
from openenv.core import EnvClient
from openenv.core.client_types import StepResult

# Import OpenEnv client base class and response type
from models import MedAgentAction, MedAgentObservation, MedAgentState

# Client to interact with MedAgent environment via API
class MedAgentEnv(EnvClient[MedAgentAction, MedAgentObservation, MedAgentState]):
    
    """
    Typed client for the MedAgent environment.
    This client talks to the OpenEnv HTTP/WebSocket server exposed by `server.app:app` and converts raw payloads into the project's typed Pydantic models.
    """

    # Convert action object into JSON
    def _step_payload(self, action: MedAgentAction) -> dict[str, Any]:
        
        """
        Convert a typed action into the JSON payload expected by `/step`.
        """
        
        return {
            "risk_level": action.risk_level,
            "decision": action.decision,
            "metadata": action.metadata,
        }

    # Convert API response into Python object
    def _parse_result(self, payload: dict[str, Any]) -> StepResult[MedAgentObservation]:
        
        """
        Parse an OpenEnv step/reset payload into a typed step result.
        """
        
        obs_data = payload.get("observation", {})
        observation = MedAgentObservation(
            age=obs_data.get("age", 0),
            symptoms=obs_data.get("symptoms", []),
            predicted_disease=obs_data.get("predicted_disease"),
            feedback=obs_data.get("feedback", ""),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    # Convert state response into Python object
    def _parse_state(self, payload: dict[str, Any]) -> MedAgentState:
        
        """
        Parse `/state` payloads into the environment's typed state.
        """
        
        return MedAgentState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            age=payload.get("age"),
            symptoms=payload.get("symptoms", []),
            task_id=payload.get("task_id"),
            task_description=payload.get("task_description"),
            expected_disease=payload.get("expected_disease"),
            target_risk=payload.get("target_risk"),
            target_decision=payload.get("target_decision"),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )

# Define public exports
__all__ = ["MedAgentEnv", "MedAgentAction", "MedAgentObservation", "MedAgentState"]
