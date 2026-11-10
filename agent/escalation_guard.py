"""
Deterministic, code-enforced safety checks applied to the agent's output.
See prior version's docstring for the full rationale (soft prompt
instructions were observed being ignored on real hardware).

This version fixes a specific false-positive risk found in review: the
original marker check was a bare substring match, so a conclusion saying
"...without immediate evacuation" (a CAUTIOUS, correct statement) would
match the same as "...requires immediate evacuation" (a genuine high-
urgency statement) -- both contain "evacuat". Negation-awareness below
checks a short window of text immediately BEFORE each marker match for a
negation cue ("no", "not", "without", etc.) and skips that occurrence if
found. This only handles PRECEDING negation ("without evacuation"), not
trailing negation ("evacuation was ruled out") -- a known, documented
scope limit, not a silent gap.
"""

from __future__ import annotations
import re

MIN_PERSISTENCE_FRAMES = 3

HIGH_URGENCY_MARKERS = [
    "level 3", "immediately", "evacuat", "bomb-threat", "bomb threat",
    "area clearance", "critical escalation",
]

NEGATION_WINDOW_CHARS = 30
NEGATION_CUES = ["no ", "not ", "without ", "n't ", "avoid ", "unnecessary ", "does not require ", "doesn't require "]


def _has_unnegated_marker(text: str, markers: list[str]) -> bool:
    """True if at least one marker appears in text WITHOUT an immediately
    preceding negation cue. A marker with a negation cue right before it
    (within NEGATION_WINDOW_CHARS) is treated as a cautious, non-urgent
    statement and skipped."""
    for marker in markers:
        for match in re.finditer(re.escape(marker), text):
            window_start = max(0, match.start() - NEGATION_WINDOW_CHARS)
            preceding_text = text[window_start:match.start()]
            if any(cue in preceding_text for cue in NEGATION_CUES):
                continue
            return True
    return False


def check_escalation_ceiling(frame_count, final_conclusion, recommended_actions, called_tools):
    flags: list[str] = []
    if not final_conclusion:
        return flags

    combined_text = (final_conclusion + " " + " ".join(recommended_actions)).lower()
    has_high_urgency_language = _has_unnegated_marker(combined_text, HIGH_URGENCY_MARKERS)

    if frame_count < MIN_PERSISTENCE_FRAMES and has_high_urgency_language:
        flags.append(
            f"Agent recommended high-urgency escalation language despite only {frame_count} "
            f"observed frame(s) -- below the {MIN_PERSISTENCE_FRAMES}-frame minimum this SOP category "
            f"requires before treating the anomaly as confirmed rather than a possible false positive. "
            f"This escalation should be treated as UNCONFIRMED pending human review."
        )

    if "get_zone_metadata" not in called_tools and has_high_urgency_language:
        flags.append(
            "Agent recommended high-urgency escalation language without ever calling "
            "get_zone_metadata to verify the zone's actual tier/hazard classification -- any zone "
            "context in the conclusion was inferred, not looked up, and should not be trusted "
            "without verification."
        )

    return flags


def requires_zone_check_before_concluding(called_tools, step_num, max_steps):
    steps_remaining = max_steps - step_num
    return "get_zone_metadata" not in called_tools and steps_remaining > 0