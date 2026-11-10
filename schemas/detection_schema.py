"""
Data contracts for the Ingestion & Detection stage.

Design note: every stage of this pipeline communicates through a Pydantic
model, never through raw strings or dicts. This is the single most important
production-readiness decision in an LLM pipeline: it forces every prompt to
produce validated, typed output, and it means downstream stages (vector
store, agent loop, report generator) can trust the shape of what they receive
instead of defensively parsing free text.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AnomalyType(str, Enum):
    TRESPASSING = "trespassing"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    HAZARD = "hazard"
    LOITERING = "loitering"
    ABANDONED_OBJECT = "abandoned_object"
    NONE = "none"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class DetectedEntity(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox | None = None


class RawDetection(BaseModel):
    """
    The structured output we force the VLM to produce for a single
    sampled frame (or small batch of frames) from a video buffer.
    """

    camera_id: str
    source_frame_path: str
    timestamp: datetime
    anomaly_type: AnomalyType
    anomaly_confidence: float = Field(ge=0.0, le=1.0)
    scene_description: str
    entities: list[DetectedEntity] = Field(default_factory=list)
    severity: Severity

    def is_actionable(self, confidence_threshold: float = 0.55) -> bool:
        """A detection only proceeds to the agent stage if it clears both
        gates: it's not a NONE classification, and confidence is high enough
        to be worth an SOP lookup + agent investigation."""
        return (
            self.anomaly_type != AnomalyType.NONE
            and self.anomaly_confidence >= confidence_threshold
        )