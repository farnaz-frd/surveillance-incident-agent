from __future__ import annotations
from pydantic import BaseModel
from schemas.detection_schema import RawDetection


class RetrievedSOP(BaseModel):
    sop_id: str
    title: str
    text: str
    similarity_score: float


class AgentStep(BaseModel):
    step_number: int
    thought: str
    action: str | None = None
    action_input: dict | None = None
    observation: str | None = None


class Incident(BaseModel):
    incident_id: str
    detection: RawDetection
    frame_count: int = 1
    persistence_duration_s: float = 0.0
    retrieved_sops: list[RetrievedSOP] = []
    agent_trace: list[AgentStep] = []
    final_conclusion: str | None = None
    escalation_level: int | None = None
    recommended_actions: list[str] = []
    review_flags: list[str] = []