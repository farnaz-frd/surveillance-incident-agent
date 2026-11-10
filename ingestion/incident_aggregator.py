"""
Groups per-frame RawDetections into incident-level clusters before they go
to SOP retrieval and the agent. Without this, a person walking through 5
sampled frames of the same event would generate 5 separate incident
reports for what is actually one event — noisy and operationally useless.

Grouping rule: consecutive (by timestamp) actionable detections on the same
camera with the same anomaly_type, where the gap between them doesn't
exceed `merge_window_s`, are merged into one cluster. This also happens to
directly implement the persistence requirement several SOPs already state
explicitly — e.g. SOP-004 (loitering) requires a minimum 3-minute dwell
time across multiple frames before treating a detection as actionable, and
SOP-005 (abandoned object) requires 3 consecutive frames before an object
is treated as genuinely unattended rather than a momentary false positive.
This module is what makes that persistence check possible upstream of the
agent, rather than something the agent has to reason about after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas.detection_schema import RawDetection


@dataclass
class IncidentCluster:
    detections: list[RawDetection]

    @property
    def representative(self) -> RawDetection:
        """The single detection used for SOP retrieval and as the
        incident's headline record — the highest-confidence one in the
        cluster, since that's the model's best single read on the event."""
        return max(self.detections, key=lambda d: d.anomaly_confidence)

    @property
    def frame_count(self) -> int:
        return len(self.detections)

    @property
    def persistence_duration_s(self) -> float:
        if len(self.detections) < 2:
            return 0.0
        timestamps = sorted(d.timestamp for d in self.detections)
        return (timestamps[-1] - timestamps[0]).total_seconds()


def group_detections_into_incidents(
    detections: list[RawDetection],
    merge_window_s: float = 6.0,
) -> list[IncidentCluster]:
    """
    Filters to actionable detections only, sorts by timestamp, and merges
    consecutive same-camera/same-anomaly-type detections within
    merge_window_s of each other into a single IncidentCluster.

    merge_window_s should generally be set to a small multiple of your
    frame sampling interval (config.py's IngestionConfig.frame_sample_rate_s)
    — e.g. 2-3x — so consecutive samples of one ongoing event merge, but
    genuinely separate events (even of the same type, same camera, minutes
    apart) don't get incorrectly combined.
    """
    actionable = sorted(
        (d for d in detections if d.is_actionable()),
        key=lambda d: d.timestamp,
    )

    clusters: list[IncidentCluster] = []
    current: list[RawDetection] = []

    for detection in actionable:
        if current:
            last = current[-1]
            same_event = (
                detection.camera_id == last.camera_id
                and detection.anomaly_type == last.anomaly_type
                and (detection.timestamp - last.timestamp).total_seconds() <= merge_window_s
            )
            if same_event:
                current.append(detection)
            else:
                clusters.append(IncidentCluster(detections=current))
                current = [detection]
        else:
            current = [detection]

    if current:
        clusters.append(IncidentCluster(detections=current))

    return clusters