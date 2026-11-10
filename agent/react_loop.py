from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass

import requests

from agent.prompts import build_system_prompt, format_sop_context
from agent.tools import TOOL_REGISTRY
from agent.escalation_guard import check_escalation_ceiling, requires_zone_check_before_concluding
from config import CONFIG, CAMERA_ZONE_MAP, DEFAULT_ZONE_ID
from schemas.detection_schema import RawDetection
from schemas.incident_schema import AgentStep, Incident, RetrievedSOP

logger = logging.getLogger(__name__)


class OllamaReasoningClient:
    def __init__(self, base_url, model, timeout_s=120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def generate(self, prompt, stop=None):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.2, "stop": stop or []}},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()["response"]


@dataclass
class ParsedStep:
    thought: str
    is_final: bool
    action: str | None = None
    action_input: dict | None = None
    final_answer: str | None = None
    escalation_level: int | None = None
    recommended_actions: list[str] | None = None
    parse_error: str | None = None


def _extract_json_block(text: str) -> dict:
    start_idx = text.find("{")
    if start_idx == -1:
        raise ValueError("No JSON object found in Action Input")
    try:
        obj, _end_idx = json.JSONDecoder().raw_decode(text, start_idx)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse Action Input JSON: {e}") from e
    return obj


def _parse_escalation_level(step_text: str) -> int | None:
    """
    Looks for an explicit "Escalation Level: X" field first (the format we
    instruct the model to use). Falls back to searching for a bare "Level X"
    mention anywhere in the text if the explicit field is missing -- small
    models have been observed skipping required labels before (see the
    Final Answer: / Recommended Actions: history in this file's sibling
    fixes), so a tolerant fallback is worth having rather than returning
    None whenever the model phrases it slightly differently.
    """
    explicit_match = re.search(r"Escalation Level:\s*([123])", step_text)
    if explicit_match:
        return int(explicit_match.group(1))

    fallback_match = re.search(r"\bLevel\s*([123])\b", step_text, re.IGNORECASE)
    if fallback_match:
        return int(fallback_match.group(1))

    return None


def parse_react_step(step_text: str) -> ParsedStep:
    thought_match = re.search(
        r"Thought:\s*(.*?)(?=Action:|Final Answer:|Escalation Level:|Recommended Actions:|$)",
        step_text, re.DOTALL,
    )
    thought = thought_match.group(1).strip() if thought_match else ""

    has_final_marker = "Final Answer:" in step_text
    has_recommended_actions = "Recommended Actions:" in step_text

    if has_final_marker or has_recommended_actions:
        if has_final_marker:
            final_match = re.search(
                r"Final Answer:\s*(.*?)(?=Escalation Level:|Recommended Actions:|$)", step_text, re.DOTALL
            )
            final_answer = final_match.group(1).strip() if final_match else ""
        else:
            final_answer = thought

        escalation_level = _parse_escalation_level(step_text)

        actions_match = re.search(
            r"Recommended Actions:\s*(.*?)(?=\nThought:|\nFinal Answer:|$)", step_text, re.DOTALL
        )
        recommended_actions = []
        if actions_match:
            for line in actions_match.group(1).strip().splitlines():
                line = line.strip().lstrip("-").strip()
                if line and not line.startswith(("Thought:", "Final Answer:", "Action:", "Action Input:", "Escalation Level:")):
                    recommended_actions.append(line)

        return ParsedStep(
            thought=thought, is_final=True, final_answer=final_answer,
            escalation_level=escalation_level, recommended_actions=recommended_actions,
        )

    action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", step_text)
    action_input_match = re.search(r"Action Input:\s*(.*)", step_text, re.DOTALL)

    if not action_match or not action_input_match:
        return ParsedStep(
            thought=thought, is_final=False,
            parse_error="Response did not contain a valid Action/Action Input pair, nor a Final Answer.",
        )

    action_name = action_match.group(1).strip()
    try:
        action_input = _extract_json_block(action_input_match.group(1))
    except (ValueError, json.JSONDecodeError) as e:
        return ParsedStep(thought=thought, is_final=False, action=action_name, parse_error=f"Action Input was not valid JSON: {e}")

    return ParsedStep(thought=thought, is_final=False, action=action_name, action_input=action_input)


class ReactAgent:
    def __init__(self, client=None, max_steps=None):
        self.client = client or OllamaReasoningClient(
            base_url=CONFIG.reasoning.ollama_base_url,
            model=CONFIG.reasoning.ollama_reasoning_model,
            timeout_s=CONFIG.reasoning.request_timeout_s,
        )
        self.max_steps = max_steps or CONFIG.reasoning.max_react_steps

    def run(self, incident_id: str, detection: RawDetection, retrieved_sops: list[RetrievedSOP],
            frame_count: int = 1, persistence_duration_s: float = 0.0) -> Incident:
        zone_id = CAMERA_ZONE_MAP.get(detection.camera_id, DEFAULT_ZONE_ID)
        if zone_id == DEFAULT_ZONE_ID:
            logger.warning("No zone mapping found for camera '%s' -- using '%s'.", detection.camera_id, DEFAULT_ZONE_ID)

        system_prompt = build_system_prompt(
            camera_id=detection.camera_id, zone_id=zone_id, timestamp=str(detection.timestamp),
            anomaly_type=detection.anomaly_type.value, anomaly_confidence=detection.anomaly_confidence,
            scene_description=detection.scene_description, severity=detection.severity.value,
            sop_context=format_sop_context(retrieved_sops), max_steps=self.max_steps,
            frame_count=frame_count, persistence_duration_s=persistence_duration_s,
        )

        transcript = ""
        steps = []
        final_conclusion = None
        escalation_level = None
        recommended_actions: list[str] = []
        called_tools: set[str] = set()

        for step_num in range(1, self.max_steps + 1):
            prompt = f"{system_prompt}\n{transcript}\nThought:"
            raw_response = self.client.generate(prompt, stop=["Observation:"])
            full_step_text = "Thought:" + raw_response
            parsed = parse_react_step(full_step_text)

            if parsed.parse_error:
                logger.warning("Step %d parse error: %s", step_num, parsed.parse_error)
                observation = f"ERROR: {parsed.parse_error} Follow the exact Thought/Action/Action Input format."
                steps.append(AgentStep(step_number=step_num, thought=parsed.thought or "(unparseable)",
                                        action=parsed.action, action_input=None, observation=observation))
                transcript += (f"Thought: {parsed.thought}\n"
                               f"{f'Action: {parsed.action}' if parsed.action else ''}\n"
                               f"Observation: {observation}\n")
                continue

            if parsed.is_final:
                if requires_zone_check_before_concluding(called_tools, step_num, self.max_steps):
                    correction = ("You must call get_zone_metadata for this zone before reaching a Final "
                                  "Answer -- escalation severity depends on the zone's actual tier and hazard "
                                  "classification. Call get_zone_metadata now, then reach your Final Answer.")
                    logger.info("Step %d: agent tried to conclude without checking zone metadata -- forcing a corrective step.", step_num)
                    steps.append(AgentStep(step_number=step_num, thought=parsed.thought, action=None, observation=f"REJECTED: {correction}"))
                    transcript += f"Thought: {parsed.thought}\nObservation: REJECTED: {correction}\n"
                    continue

                steps.append(AgentStep(step_number=step_num, thought=parsed.thought, action=None, observation=None))
                final_conclusion = parsed.final_answer
                escalation_level = parsed.escalation_level
                recommended_actions = parsed.recommended_actions or []
                break

            tool_name = parsed.action
            action_input = parsed.action_input or {}
            if tool_name not in TOOL_REGISTRY:
                observation = f"ERROR: Unknown tool '{tool_name}'. Available tools: {list(TOOL_REGISTRY.keys())}"
            else:
                try:
                    observation = TOOL_REGISTRY[tool_name].func(**action_input)
                    called_tools.add(tool_name)
                except TypeError as e:
                    observation = f"ERROR calling tool '{tool_name}' with input {action_input}: {e}"

            steps.append(AgentStep(step_number=step_num, thought=parsed.thought, action=tool_name,
                                    action_input=action_input, observation=observation))
            transcript += (f"Thought: {parsed.thought}\nAction: {tool_name}\n"
                           f"Action Input: {json.dumps(action_input)}\nObservation: {observation}\n")
        else:
            logger.warning("ReAct loop exhausted %d steps without a Final Answer", self.max_steps)
            final_conclusion = "Agent did not reach a conclusion within the step limit. Manual review recommended."

        review_flags = check_escalation_ceiling(
            frame_count=frame_count, final_conclusion=final_conclusion,
            recommended_actions=recommended_actions, called_tools=called_tools,
        )
        for flag in review_flags:
            logger.warning("REVIEW FLAG for %s: %s", incident_id, flag)

        return Incident(
            incident_id=incident_id, detection=detection, frame_count=frame_count,
            persistence_duration_s=persistence_duration_s, retrieved_sops=retrieved_sops,
            agent_trace=steps, final_conclusion=final_conclusion, escalation_level=escalation_level,
            recommended_actions=recommended_actions, review_flags=review_flags,
        )