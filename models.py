# models.py

# Models for the MedAgent clinical triage environment, including action, observation, and state definitions.

# Import Requirements

from __future__ import annotations
from typing import Literal
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

# Define allowed values for risk level and decision type
RiskLevel = Literal["low", "medium", "high"]

DecisionType = Literal[
    "monitor_at_home",
    "book_consult",
    "seek_emergency_care",
]

# Action submitted by the agent: predicts risk level and decision
class MedAgentAction(Action):
    
    """
    Triage action submitted by the agent.
    """

    risk_level: RiskLevel = Field(
        ..., description="Estimated patient risk severity."
    )
    
    decision: DecisionType = Field(
        ..., description="Recommended next clinical action."
    )

# Observation returned to the agent: includes patient info and feedback
class MedAgentObservation(Observation):
    """
    Patient snapshot plus evaluation feedback from the environment.
    """

    age: int = Field(..., ge=0, description="Patient age in years.")
    
    symptoms: list[str] = Field(
        default_factory=list,
        description="Observed patient symptoms for this episode.",
    )
    
    predicted_disease: str | None = Field(
        default=None,
        description="Reference disease revealed by the environment after triage.",
    )
    
    feedback: str = Field(
        default="",
        description="Short textual feedback about the submitted triage.",
    )

# Internal state used for evaluation and debugging
class MedAgentState(State):
    
    """
    Internal state used for grading and debugging.
    """

    age: int | None = Field(default=None, ge=0)
    symptoms: list[str] = Field(default_factory=list)
    task_id: str | None = Field(default=None)
    task_description: str | None = Field(default=None)
    expected_disease: str | None = Field(default=None)
    target_risk: RiskLevel | None = Field(default=None)
    target_decision: DecisionType | None = Field(default=None)
    cumulative_reward: float = Field(default=0.0, ge=0.0)
