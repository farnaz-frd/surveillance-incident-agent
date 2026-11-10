"""
End-to-end pipeline orchestration:

  video file
    -> sample frames (ingestion.video_processor)
    -> per-frame detection, 2-stage VLM + reasoning model (ingestion.multimodal_detector)
    -> filter to actionable detections + merge same-event frames (ingestion.incident_aggregator)
    -> per incident: SOP retrieval (knowledge.vector_store)
    -> per incident: ReAct agent investigation (agent.react_loop)
    -> per incident: Markdown report (reporting.report_generator)

Requires (on the machine actually running this, not the dev sandbox):
  - `ollama serve` running with moondream:1.8b, qwen2.5:3b, nomic-embed-text pulled
  - `python data/seed_db.py` already run once to create data/facility_ops.db
  - `python -m knowledge.sop_ingest` already run once to index the SOPs

End-to-end pipeline orchestration:
  video file -> sample frames -> per-frame detection -> merge into incidents
  -> per incident: SOP retrieval -> agent investigation -> Markdown report

Run: python main.py <path_to_video> [--camera-id CAM_01]
     python main.py data/sample_videos/test.mp4 --camera-id CAM_01
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from config import CONFIG, FRAME_CACHE_DIR
from ingestion.incident_aggregator import group_detections_into_incidents
from ingestion.multimodal_detector import DetectionError, MultimodalDetector
from ingestion.video_processor import VideoProcessingError, VideoProcessor
from knowledge.embedder import OllamaEmbeddingProvider
from knowledge.vector_store import VectorStore
from agent.react_loop import ReactAgent
from reporting.report_generator import save_report
from schemas.detection_schema import RawDetection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(video_path, camera_id, detector=None, vector_store=None, agent=None):
    logger.info("=== Stage 1/5: Sampling frames from %s ===", video_path)
    video_processor = VideoProcessor(
        output_dir=FRAME_CACHE_DIR,
        sample_rate_s=CONFIG.ingestion.frame_sample_rate_s,
        max_frames=CONFIG.ingestion.max_frames_per_clip,
    )
    try:
        sampled_frames = video_processor.sample_frames(video_path, camera_id=camera_id)
    except VideoProcessingError as e:
        logger.error("Video processing failed: %s", e)
        sys.exit(1)
    logger.info("Sampled %d frames", len(sampled_frames))

    logger.info("=== Stage 2/5: Running multimodal detection on each frame ===")
    detector = detector or MultimodalDetector()
    pipeline_start = datetime.now()

    detections: list[RawDetection] = []
    for frame in sampled_frames:
        frame_timestamp = pipeline_start + timedelta(seconds=frame.timestamp_s)
        try:
            detection = detector.detect(frame_path=frame.frame_path, camera_id=camera_id, timestamp=frame_timestamp)
            logger.info("  %s -> anomaly_type=%s confidence=%.2f", frame.frame_path.name, detection.anomaly_type.value, detection.anomaly_confidence)
            detections.append(detection)
        except DetectionError as e:
            logger.warning("  %s -> detection failed, skipping this frame: %s", frame.frame_path.name, e)
            continue

    if not detections:
        logger.warning("No successful detections from any frame. Nothing further to do.")
        return []

    logger.info("=== Stage 3/5: Grouping detections into incidents ===")
    merge_window_s = CONFIG.ingestion.frame_sample_rate_s * 3
    clusters = group_detections_into_incidents(detections, merge_window_s=merge_window_s)

    if not clusters:
        logger.info("No actionable anomalies detected in this clip. No incidents to report.")
        return []
    logger.info("%d incident(s) identified from %d actionable detection(s)", len(clusters), sum(c.frame_count for c in clusters))

    logger.info("=== Stage 4/5: SOP retrieval + agent investigation per incident ===")
    if vector_store is None:
        embedding_provider = OllamaEmbeddingProvider(base_url=CONFIG.vlm.ollama_base_url, model=CONFIG.vector_store.embedding_model)
        vector_store = VectorStore(embedding_provider=embedding_provider, persist_dir=CONFIG.vector_store.persist_dir, collection_name=CONFIG.vector_store.collection_name)
    if vector_store.count() == 0:
        logger.warning("Vector store is empty -- run `python -m knowledge.sop_ingest` first. Proceeding without SOP context.")

    agent = agent or ReactAgent()
    report_paths: list[Path] = []

    for i, cluster in enumerate(clusters, start=1):
        rep = cluster.representative
        incident_id = f"INC-{camera_id}-{rep.timestamp.strftime('%Y%m%dT%H%M%S')}-{i:02d}"
        logger.info("  [%s] %s (confidence=%.2f, %d frame(s) over %.0fs)", incident_id, rep.anomaly_type.value, rep.anomaly_confidence, cluster.frame_count, cluster.persistence_duration_s)

        # Each incident is processed independently and defensively: a
        # failure investigating ONE incident (a slow/timed-out model call,
        # a tool error, anything) must not prevent the remaining incidents
        # in this batch from being investigated and reported. Before this
        # fix, an exception anywhere in this block crashed run_pipeline
        # entirely, silently discarding every incident after the failed
        # one -- including ones whose SOP retrieval had already succeeded.
        try:
            retrieved_sops = []
            if vector_store.count() > 0:
                query_text = (f"{rep.scene_description} Anomaly type: {rep.anomaly_type.value}. "
                              f"Observed across {cluster.frame_count} frame(s) over {cluster.persistence_duration_s:.0f} seconds.")
                retrieved_sops = vector_store.query(query_text, top_k=CONFIG.vector_store.top_k)
                for sop in retrieved_sops:
                    logger.info("    retrieved: %s (score=%.3f)", sop.sop_id, sop.similarity_score)

            incident = agent.run(
                incident_id=incident_id, detection=rep, retrieved_sops=retrieved_sops,
                frame_count=cluster.frame_count, persistence_duration_s=cluster.persistence_duration_s,
            )

            report_path = save_report(incident, output_dir="reports")
            logger.info("    report written: %s", report_path)
            report_paths.append(report_path)

        except Exception as e:
            logger.error(
                "    Investigation FAILED for %s (%s) -- skipping this incident and continuing with the rest. "
                "Error: %s: %s",
                incident_id, rep.anomaly_type.value, type(e).__name__, e,
            )
            continue

    logger.info("=== Stage 5/5: Done. %d report(s) generated in ./reports/ (out of %d incident(s) identified) ===", len(report_paths), len(clusters))
    return report_paths


def main():
    parser = argparse.ArgumentParser(description="Autonomous Video Surveillance & Incident Report Agent")
    parser.add_argument("video_path", help="Path to a video file to analyze")
    parser.add_argument("--camera-id", default="CAM_01", help="Camera identifier to attach to detections")
    args = parser.parse_args()

    if not Path(args.video_path).exists():
        logger.error("Video file not found: %s", args.video_path)
        sys.exit(1)

    run_pipeline(args.video_path, camera_id=args.camera_id)


if __name__ == "__main__":
    main()