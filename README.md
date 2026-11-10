# Surveillance Incident Agent

An autonomous video-surveillance agent that turns a video clip into an SOP-grounded incident report. It combines computer vision, retrieval over standard operating procedures (SOPs), and a hand-written ReAct agent that investigates each incident with tools before writing a Markdown report.

It runs **fully local** with [Ollama](https://ollama.com), with no cloud API and no API keys. The agent loop, parsing, guardrails and vector retrieval are written in plain Python. **No LangChain and no LlamaIndex.**

> **Status:** working end to end on real clips, still being evaluated. See [Limitations](#limitations-and-honest-notes) before drawing conclusions about accuracy.

## What it does

```
video.mp4
   |
   v
[1] Frame sampling (OpenCV, fixed interval)
   |
   v
[2] Two-stage detection per frame
      moondream:1.8b  -> plain-language scene description
      qwen2.5:3b      -> structured JSON classification (Pydantic-validated)
   |
   v
[3] Incident aggregation (same camera + time proximity -> one incident)
   |
   v
[4] For each incident:
      SOP retrieval (ChromaDB, cosine similarity, nomic-embed-text)
      ReAct agent investigation (tools: badge log, maintenance schedule, zone metadata)
   |
   v
[5] Executive Markdown incident report
```

### Design points

- **Framework-free ReAct loop.** Thought / Action / Action Input / Observation, with a stop sequence, a step cap, a tolerant parser and a tool registry rendered into the prompt.
- **Ground truth is injected, not inferred.** The zone comes from a camera-to-zone map. Frame count and persistence are measured, not guessed by the model. A precomputed verdict says whether the SOP's minimum persistence is met.
- **Guardrails enforced in code**, because small models ignore soft prompt instructions:
  - the agent cannot conclude before it has looked up the zone metadata;
  - a post-hoc review flags high-urgency conclusions based on fewer than 3 frames, using negation-aware matching so "no immediate threat" is not flagged;
  - the report shows the per-frame *Initial Detection Severity* and the post-investigation *Agent Escalation Level* separately.
- **Robust structured output.** JSON is extracted with `json.JSONDecoder.raw_decode`. Failed parses are retried with clean error feedback, and a sanity check rejects contradictory output.
- **Pydantic schemas** are the contract between stages. Providers sit behind small interfaces and are dependency-injected, so the pipeline is testable with mock clients.

## Requirements

- Python 3.10+ (developed and run on 3.12)
- [Ollama](https://ollama.com) running locally
- About 8 GB RAM (CPU-only is enough, but slow, see below)

Pull the models:

```bash
ollama pull moondream:1.8b
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

python data/seed_db.py            # creates the mock facility database (SQLite)
python -m knowledge.sop_ingest    # embeds and indexes the SOPs in data/sops/
```

## Usage

```bash
python main.py path/to/clip.mp4 --camera-id CAM_04
```

The camera ID decides which zone the footage belongs to (see `CAMERA_ZONE_MAP` in `config.py`). Reports are written as Markdown files.

Tests use mock clients, so they do not need Ollama. Run a test module directly:

```bash
python tests/test_react_loop.py
python tests/test_tools.py
```

An example report is in [docs/example_report.md](docs/example_report.md).

## Project structure

```
config.py                    models, timeouts, camera -> zone map
main.py                      pipeline entry point
diag_ollama_timing.py        optional: measures Ollama load / prompt / generation speed
schemas/                     Pydantic models (detection, incident, report)
ingestion/
  video_processor.py         OpenCV frame sampling
  multimodal_detector.py     two-stage detection (describe, then classify)
  incident_aggregator.py     merge per-frame detections into incidents
knowledge/
  embedder.py                Ollama embeddings
  vector_store.py            ChromaDB, cosine similarity
  sop_ingest.py              parse and index SOP files
data/
  seed_db.py                 builds the mock facility database (SQLite)
  mock_badge_logs.json       mock badge-scan data
  mock_maintenance.json      mock maintenance schedule
  sops/                      five example SOPs (written for this project)
  sample_videos/             put your own test clips here (not tracked by git)
agent/
  tools.py                   badge log, maintenance schedule, zone metadata
  tool_schemas.py            tool definitions
  prompts.py                 system prompt and SOP formatting
  react_loop.py              ReAct agent
  escalation_guard.py        code-enforced guardrails
reporting/
  report_generator.py        Markdown report
reports/                     generated reports (not tracked by git)
tests/                       unit tests (ReAct loop, tools) using mock clients
docs/                        example output
```

## Example results

Results from runs on real clips, on CPU only. These are single runs, not a benchmark.

| Scenario | Camera | What happened |
|---|---|---|
| Abandoned bag (3 consecutive frames) | CAM_01 | Classified as abandoned object (95%). The agent checked badge and maintenance logs, then looked up the zone. Report recommended facilities inspection. |
| Tailgating through a door | CAM_04 | Detected as unauthorized access in some runs, but also as trespassing or loitering in others. Two adjacent detections were merged into one incident. The zone-check gate rejected an early conclusion. |

<!-- Add the remaining test scenarios here as you run them. -->

## Limitations and honest notes

- **Speed.** On an 8 GB CPU-only laptop, measured on this machine: about 7 tokens/s reading prompts and about 3 tokens/s generating. A short clip takes on the order of 15 minutes. Prompt reading is the main cost, not model loading.
- **Classification varies between runs.** The vision model is 1.8B parameters and the reasoning model is 3B. The same clip can receive different labels on different runs.
- **Frames are judged independently.** Events that unfold over time, such as tailgating, are hard to recognise from single frames. Persistence is measured, but motion is not.
- **Demo timing is accelerated.** Frames are sampled seconds apart, while the SOPs talk in minutes. The prompt tells the model not to convert between them, and the persistence rule is expressed in frames.
- **The aggregator can over-merge.** Detections from the same camera within the time window are merged, even if they are unrelated events.
- **Escalation negation handling** only looks at words before a marker phrase.
- **Mock data.** The badge log, maintenance schedule and zone metadata are a seeded SQLite database, and the SOPs are examples written for this project. Nothing comes from a real facility.
- **Not evaluated at scale.** No accuracy figures are claimed.

Larger local models, a GPU, or a hosted model should improve label consistency and speed. The provider interfaces are meant to make that swap straightforward, but only Ollama is implemented and tested.

## Possible next steps

- Wider test coverage (aggregator, detector, report generator, full pipeline)
- Provider adapters for hosted models behind the existing interfaces
- Multi-frame (temporal) input to the detector
- A labelled evaluation set and measured detection accuracy
- Per-call token and latency logging

## Author

Farnaz Farhand

## License

MIT, see [LICENSE](LICENSE).