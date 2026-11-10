from __future__ import annotations
from datetime import datetime
from pathlib import Path

from schemas.incident_schema import Incident

SEVERITY_LABELS = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "critical": "CRITICAL"}
ESCALATION_LEVEL_LABELS = {1: "Level 1 (Low)", 2: "Level 2 (Medium)", 3: "Level 3 (High)"}


def _format_entities(entities):
    if not entities:
        return "None detected."
    return "\n".join(f"- {e.label} (confidence: {e.confidence:.0%})" for e in entities)


def _format_sops(retrieved_sops):
    if not retrieved_sops:
        return "_No relevant SOPs were retrieved for this incident._"
    return "\n\n".join(f"**{s.sop_id} -- {s.title}** (relevance: {s.similarity_score:.2f})" for s in retrieved_sops)


def _format_agent_trace(agent_trace):
    if not agent_trace:
        return "_No agent investigation was performed for this incident._"
    lines = []
    for step in agent_trace:
        lines.append(f"**Step {step.step_number}**")
        lines.append(f"- *Reasoning:* {step.thought}")
        if step.action:
            action_input_str = ", ".join(f"{k}={v}" for k, v in (step.action_input or {}).items())
            lines.append(f"- *Action:* `{step.action}({action_input_str})`")
        if step.observation:
            lines.append(f"- *Observation:* {chr(10).join(step.observation.split(chr(10)))}")
        lines.append("")
    return "\n".join(lines)


def _format_review_banner(review_flags):
    if not review_flags:
        return ""
    flag_bullet_lines = []
    for flag_text in review_flags:
        flag_bullet_lines.append("- " + flag_text)
    joined_flags = "\n".join(flag_bullet_lines)
    banner = "\n> **[AUTOMATED REVIEW REQUIRED]** Do not act on this report without human verification:\n>\n"
    banner += joined_flags
    banner += "\n"
    return banner


def generate_report(incident: Incident) -> str:
    d = incident.detection
    initial_severity_label = SEVERITY_LABELS.get(d.severity.value, d.severity.value.upper())

    # Two SEPARATE severity-like signals, shown side by side rather than
    # collapsed into one field: the detection stage's per-frame estimate
    # (made before any investigation) and the agent's own post-investigation
    # SOP escalation level (made after gathering evidence). They can
    # legitimately disagree -- e.g. a HIGH initial read that the
    # investigation walks back to Level 1 once context rules it out. Showing
    # both, clearly labeled, is more honest than picking one.
    escalation_display = ESCALATION_LEVEL_LABELS.get(incident.escalation_level, "Not stated by agent")

    review_banner = _format_review_banner(incident.review_flags)

    persistence_line = f"observed across {incident.frame_count} sampled frame(s)"
    if incident.frame_count > 1:
        persistence_line += f", spanning {incident.persistence_duration_s:.0f} seconds"

    report = "# Security Incident Report\n\n"
    report += f"**Incident ID:** {incident.incident_id}\n"
    report += f"**Camera:** {d.camera_id}\n"
    report += f"**Timestamp:** {d.timestamp.isoformat(timespec='seconds')}\n"
    report += f"**Classification:** {d.anomaly_type.value.replace('_', ' ').title()}\n"
    report += f"**Detection Confidence:** {d.anomaly_confidence:.0%}\n"
    report += f"**Initial Detection Severity:** {initial_severity_label} _(per-frame estimate, before investigation)_\n"
    report += f"**Agent Escalation Level:** {escalation_display} _(post-investigation, evidence-based)_\n"
    report += f"**Persistence:** {persistence_line}\n"
    report += review_banner
    report += "\n---\n\n"
    report += "## Scene Description\n\n"
    report += f"{d.scene_description}\n\n"
    report += "**Entities Detected:**\n"
    report += _format_entities(d.entities)
    report += "\n\n---\n\n"
    report += "## Relevant Standard Operating Procedures\n\n"
    report += _format_sops(incident.retrieved_sops)
    report += "\n\n---\n\n"
    report += "## Investigation Trace\n\n"
    report += _format_agent_trace(incident.agent_trace)
    report += "\n---\n\n"
    report += "## Conclusion\n\n"
    report += incident.final_conclusion or "_No conclusion reached, investigation may be incomplete or still in progress._"
    report += "\n\n## Recommended Actions\n\n"
    report += ("\n".join(f"- {a}" for a in incident.recommended_actions) if incident.recommended_actions else "_None specified._")
    report += "\n\n---\n\n"
    report += f"*Report generated {datetime.now().isoformat(timespec='seconds')} by Autonomous Video Surveillance & Incident Report Agent.*\n"
    return report


def save_report(incident: Incident, output_dir="reports") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_text = generate_report(incident)
    file_path = output_path / f"{incident.incident_id}.md"
    file_path.write_text(report_text, encoding="utf-8")
    return file_path