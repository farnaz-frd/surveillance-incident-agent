"""
Multimodal detection, TWO-STAGE design:

  Stage 1 (perception): moondream:1.8b looks at the image and produces a
    plain-language scene description. Small VLMs are reasonably good at
    this — free-form description is their natural output mode.

  Stage 2 (structuring): qwen2.5:3b (a text model, better at strict
    instruction-following) takes that description and classifies it into
    our RawDetection schema as JSON.

Why not one stage: we initially asked moondream to look at the image AND
produce complex nested JSON in one call. On real hardware this failed —
the model echoed placeholder text from the prompt's schema example instead
of generating real content, because a 1.8B model asked to do visual
reasoning and strict structured-output-following simultaneously does
neither well. Splitting the job so each model does the one thing it's
actually capable of is the fix, and is a legitimate production pattern for
resource-constrained multimodal pipelines, not just a workaround.

The tolerant JSON extraction + bounded retry loop from the original
single-stage version is preserved here, now applied to Stage 2 only —
Stage 1's plain-text output has no parsing to fail.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path

import requests
from pydantic import ValidationError

from config import CONFIG
from schemas.detection_schema import RawDetection

logger = logging.getLogger(__name__)


class DetectionError(Exception):
    """Raised when the detection pipeline fails after retries."""


# --- Stage 1: perception prompt (kept deliberately simple) ---

DESCRIPTION_PROMPT_TEMPLATE = """You are looking at a still frame from a security camera (camera "{camera_id}", captured at {timestamp}).

Describe what you see in 2-3 plain sentences. Mention any people, vehicles, or objects, what they are doing, and anything that looks unusual, out of place, or potentially concerning (e.g. someone climbing a fence, an unattended bag, smoke, a person in a restricted-looking area). If the scene looks completely normal and empty, say so."""


# --- Stage 2: structuring prompt (text-only, no image) ---

CLASSIFICATION_PROMPT_TEMPLATE = """You are a security analyst. Based ONLY on the following scene description from a security camera, classify the incident.

Scene description: "{scene_description}"

Respond with ONLY a single JSON object, no markdown fences, no extra text, matching exactly this shape:
{{
  "anomaly_type": "trespassing" | "unauthorized_access" | "hazard" | "loitering" | "abandoned_object" | "none",
  "anomaly_confidence": <float 0.0 to 1.0>,
  "severity": "low" | "medium" | "high" | "critical",
  "entities": [
    {{"label": "person or vehicle or object type mentioned", "confidence": <float 0.0 to 1.0>}}
  ]
}}

If the scene description mentions nothing unusual, use "none" for anomaly_type, a low anomaly_confidence, and "low" severity."""


def _extract_json_object(raw_text: str) -> dict:
    """
    Finds and parses the first complete JSON object in the text, ignoring
    anything before or after it. Using json.JSONDecoder.raw_decode rather
    than a greedy regex means trailing commentary after a valid object
    doesn't cause a confusing "Extra data" failure.
    """
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start_idx = text.find("{")
    if start_idx == -1:
        raise DetectionError(f"No JSON object found in model output: {raw_text[:200]!r}")

    try:
        obj, _end_idx = json.JSONDecoder().raw_decode(text, start_idx)
    except json.JSONDecodeError as e:
        raise DetectionError(
            f"Could not parse a valid JSON object starting at position {start_idx}: {e}. "
            f"Raw text: {raw_text[:200]!r}"
        ) from e

    return obj


class OllamaVisionClient:
    """Stage 1 client — image + text prompt, plain text response."""

    def __init__(self, base_url: str, model: str, timeout_s: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def describe(self, prompt: str, image_path: Path) -> str:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0.3},  # slightly higher: free-text description benefits from a little more fluency than strict-JSON generation does
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()["response"].strip()


class OllamaTextClient:
    """Stage 2 client — text-only prompt, used for structured classification."""

    def __init__(self, base_url: str, model: str, timeout_s: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},  # low temp: we want consistent JSON, not creativity
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()["response"]


class MultimodalDetector:
    def __init__(
        self,
        vision_client: OllamaVisionClient | None = None,
        reasoning_client: OllamaTextClient | None = None,
        max_retries: int = 2,
    ):
        self.vision_client = vision_client or OllamaVisionClient(
            base_url=CONFIG.vlm.ollama_base_url,
            model=CONFIG.vlm.ollama_vision_model,
            timeout_s=CONFIG.vlm.request_timeout_s,
        )
        self.reasoning_client = reasoning_client or OllamaTextClient(
            base_url=CONFIG.reasoning.ollama_base_url,
            model=CONFIG.reasoning.ollama_reasoning_model,
            timeout_s=CONFIG.reasoning.request_timeout_s,
        )
        self.max_retries = max_retries

    def detect(self, frame_path: Path, camera_id: str, timestamp) -> RawDetection:
        # --- Stage 1: perception (no retry needed — plain text can't "fail to parse") ---
        description_prompt = DESCRIPTION_PROMPT_TEMPLATE.format(camera_id=camera_id, timestamp=timestamp)
        try:
            scene_description = self.vision_client.describe(description_prompt, frame_path)
        except requests.RequestException as e:
            raise DetectionError(
                f"Could not reach vision model at {self.vision_client.base_url}. "
                f"Is `ollama serve` running and is '{self.vision_client.model}' pulled? Original error: {e}"
            ) from e

        if not scene_description:
            raise DetectionError(f"Vision model returned an empty description for {frame_path}")

        # --- Stage 2: structuring, with the same tolerant retry loop as before ---
        prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(scene_description=scene_description)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 2):
            try:
                raw_text = self.reasoning_client.generate(prompt)
                parsed = _extract_json_object(raw_text)

                detection = RawDetection(
                    camera_id=camera_id,
                    source_frame_path=str(frame_path),
                    timestamp=timestamp,
                    scene_description=scene_description,  # from Stage 1, not re-generated
                    **parsed,
                )

                # Semantic sanity check, not just schema validation: a
                # non-"none" anomaly_type paired with near-zero confidence
                # is schema-valid but almost always means the model echoed
                # the prompt's placeholder example (e.g. "confidence: 0.0")
                # rather than making a real judgment. Treat it as a soft
                # failure and retry instead of silently returning a
                # nonsensical detection.
                if detection.anomaly_type.value != "none" and detection.anomaly_confidence < 0.1:
                    raise DetectionError(
                        f"Suspicious combination: anomaly_type='{detection.anomaly_type.value}' "
                        f"with anomaly_confidence={detection.anomaly_confidence} — likely echoed "
                        f"placeholder text rather than a genuine classification."
                    )

                return detection

            except ValidationError as e:
                last_error = e
                clean_errors = "; ".join(
                    f"'{'.'.join(str(loc) for loc in err['loc'])}' is {err['msg'].lower()}"
                    for err in e.errors()
                )
                logger.warning(
                    "Classification validation failed on attempt %d/%d for %s: %s",
                    attempt, self.max_retries + 1, frame_path.name, clean_errors,
                )
                prompt = (
                    CLASSIFICATION_PROMPT_TEMPLATE.format(scene_description=scene_description)
                    + f"\n\nYour previous response was missing or had invalid fields: {clean_errors}. "
                      f"Respond with ONLY the complete raw JSON object including every required field, nothing else."
                )
            except (DetectionError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning(
                    "Classification failed on attempt %d/%d for %s: %s",
                    attempt, self.max_retries + 1, frame_path.name, e,
                )
                prompt = (
                    CLASSIFICATION_PROMPT_TEMPLATE.format(scene_description=scene_description)
                    + f"\n\nYour previous response was invalid: {e}. "
                      f"Respond with ONLY the raw JSON object, nothing else — no commentary, no markdown fences. "
                      f"Make sure anomaly_confidence genuinely reflects your assessment of the scene description above."
                )
            except requests.RequestException as e:
                raise DetectionError(
                    f"Could not reach reasoning model at {self.reasoning_client.base_url}. "
                    f"Is `ollama serve` running and is '{self.reasoning_client.model}' pulled? Original error: {e}"
                ) from e

        raise DetectionError(
            f"Failed to classify detection for {frame_path} after {self.max_retries + 1} attempts: {last_error}. "
            f"Scene description was: {scene_description!r}"
        )