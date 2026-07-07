"""FrameFlow local inference worker.

Runs an open-source text-to-video diffusion model entirely on YOUR OWN
hardware. After the one-time model download, video generation makes no
external network calls — no cloud AI APIs, no third-party services.

Usage:
    pip install -r requirements.txt
    uvicorn main:app --host 127.0.0.1 --port 8001

Environment variables:
    T2V_MODEL     Model to load (default: damo-vilab/text-to-video-ms-1.7b).
                  Any diffusers text-to-video pipeline works.
    OUTPUT_DIR    Where generated MP4s are written (default: ./outputs).
    WORKER_TOKEN  Optional shared secret. When set, every request must send
                  "Authorization: Bearer <token>". Set the same value as
                  LOCAL_WORKER_TOKEN in the Next.js app.
    FORCE_CPU     Set to "1" to ignore an available GPU.

Then point the FrameFlow app at this worker:
    VIDEO_PROVIDER="local"
    LOCAL_WORKER_URL="http://127.0.0.1:8001"
"""

from __future__ import annotations

import os
import queue
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("T2V_MODEL", "damo-vilab/text-to-video-ms-1.7b")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))
WORKER_TOKEN = os.environ.get("WORKER_TOKEN")
FORCE_CPU = os.environ.get("FORCE_CPU") == "1"
MAX_QUEUED_JOBS = int(os.environ.get("MAX_QUEUED_JOBS", "10"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FrameFlow local T2V worker", version="1.0")

# ---------------------------------------------------------------------------
# Job store (in-memory; jobs are transient — the Next.js app owns durability)
# ---------------------------------------------------------------------------

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
job_queue: "queue.Queue[str]" = queue.Queue()

_pipeline = None
_pipeline_error: Optional[str] = None
_pipeline_lock = threading.Lock()


def _device() -> str:
    if FORCE_CPU:
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_pipeline():
    """Load the diffusion pipeline once (first job or explicit warmup)."""
    global _pipeline, _pipeline_error
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        try:
            import torch
            from diffusers import DiffusionPipeline

            device = _device()
            dtype = torch.float16 if device == "cuda" else torch.float32
            pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
            pipe = pipe.to(device)
            if device == "cpu":
                # Reduce peak memory on CPU boxes.
                try:
                    pipe.enable_attention_slicing()
                except Exception:
                    pass
            _pipeline = pipe
            _pipeline_error = None
            return _pipeline
        except Exception as exc:  # noqa: BLE001 — surfaced to the caller
            _pipeline_error = f"{type(exc).__name__}: {exc}"
            raise


def _run_job(job_id: str) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return
        job["status"] = "processing"
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        params = dict(job["params"])

    try:
        import torch
        from diffusers.utils import export_to_video

        pipe = _load_pipeline()

        generator = None
        if params.get("seed") is not None:
            generator = torch.Generator(device=_device()).manual_seed(int(params["seed"]))

        result = pipe(
            prompt=params["prompt"],
            negative_prompt=params.get("negative_prompt") or None,
            num_frames=params["num_frames"],
            num_inference_steps=params["steps"],
            width=params["width"],
            height=params["height"],
            generator=generator,
        )
        frames = result.frames[0]

        video_path = OUTPUT_DIR / f"{job_id}.mp4"
        export_to_video(frames, str(video_path), fps=params["fps"])

        with jobs_lock:
            job["status"] = "completed"
            job["video_file"] = video_path.name
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:  # noqa: BLE001 — job errors must not kill the worker
        traceback.print_exc()
        message = f"{type(exc).__name__}: {exc}"
        if _pipeline is None:
            message = (
                f"Model could not be loaded ({message}). The first run downloads "
                f"'{MODEL_ID}' from Hugging Face — check internet access, disk "
                f"space, and the T2V_MODEL setting. Generation itself runs fully "
                f"offline once the weights are cached."
            )
        with jobs_lock:
            job["status"] = "failed"
            job["error"] = message[:1000]
            job["finished_at"] = datetime.now(timezone.utc).isoformat()


def _worker_loop() -> None:
    """Single sequential consumer — diffusion saturates one GPU/CPU anyway."""
    while True:
        job_id = job_queue.get()
        try:
            _run_job(job_id)
        finally:
            job_queue.task_done()


threading.Thread(target=_worker_loop, daemon=True, name="t2v-worker").start()

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def _check_auth(authorization: Optional[str]) -> None:
    if WORKER_TOKEN is None:
        return
    expected = f"Bearer {WORKER_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing worker token.")


class JobRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)
    negative_prompt: Optional[str] = Field(default=None, max_length=2000)
    num_frames: int = Field(default=16, ge=8, le=64)
    fps: int = Field(default=8, ge=4, le=24)
    width: int = Field(default=256, ge=128, le=1024, multiple_of=8)
    height: int = Field(default=256, ge=128, le=1024, multiple_of=8)
    steps: int = Field(default=25, ge=5, le=60)
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_ID,
        "device": _device(),
        "model_loaded": _pipeline is not None,
        "model_error": _pipeline_error,
        "queue_depth": job_queue.qsize(),
    }


@app.post("/jobs", status_code=201)
def create_job(body: JobRequest, authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)
    if job_queue.qsize() >= MAX_QUEUED_JOBS:
        raise HTTPException(status_code=429, detail="Worker queue is full — try again shortly.")

    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "params": body.model_dump(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "error": None,
            "video_file": None,
        }
    job_queue.put(job_id)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job.")
        return {
            "job_id": job_id,
            "status": job["status"],
            "error": job["error"],
            "video_url": f"/videos/{job['video_file']}" if job["video_file"] else None,
            "created_at": job["created_at"],
        }


@app.get("/videos/{filename}")
def get_video(filename: str):
    # Videos are served unauthenticated so the browser <video> tag can play
    # them; filenames are unguessable 32-char job IDs.
    safe = Path(filename).name
    if not safe.endswith(".mp4"):
        raise HTTPException(status_code=404, detail="Not found.")
    path = OUTPUT_DIR / safe
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(path, media_type="video/mp4")
