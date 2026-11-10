from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SOP_DIR = DATA_DIR / "sops"
FRAME_CACHE_DIR = PROJECT_ROOT / ".frame_cache"


@dataclass
class VLMConfig:
    ollama_base_url: str = "http://localhost:11434"
    ollama_vision_model: str = "moondream:1.8b"
    request_timeout_s: int = 180


@dataclass
class ReasoningConfig:
    ollama_base_url: str = "http://localhost:11434"
    ollama_reasoning_model: str = "qwen2.5:3b"
    # Bumped from 120s: on this hardware (CPU-bound, older GPU), individual
    # ReAct steps have been observed routinely taking 2-4 minutes even
    # before this timeout was hit, and the prompt has grown substantially
    # across fixes (persistence checks, unit-math guardrails, escalation
    # level instructions, full SOP text for top_k retrieved documents) --
    # more tokens in, more time to generate. 120s was cutting it close even
    # in earlier runs; it's now genuinely too tight.
    request_timeout_s: int = 300
    max_react_steps: int = 6


@dataclass
class VectorStoreConfig:
    persist_dir: str = str(PROJECT_ROOT / ".chroma_store")
    collection_name: str = "security_sops"
    embedding_model: str = "nomic-embed-text"
    top_k: int = 3


@dataclass
class IngestionConfig:
    frame_sample_rate_s: float = 2.0
    max_frames_per_clip: int = 10
    anomaly_confidence_threshold: float = 0.55


CAMERA_ZONE_MAP: dict[str, str] = {
    "CAM_01": "ZONE_MAIN_PLAZA",
    "CAM_02": "ZONE_ELEC_ROOM",
    "CAM_03": "ZONE_LOADING_DOCK",
    "CAM_04": "ZONE_LOBBY",
    "CAM_05": "ZONE_SERVER_RM",
}
DEFAULT_ZONE_ID = "ZONE_UNKNOWN"


@dataclass
class SystemConfig:
    vlm: VLMConfig = field(default_factory=VLMConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)


CONFIG = SystemConfig()