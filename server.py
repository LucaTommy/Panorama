"""
FastAPI backend for the video-to-panorama pipeline.

Endpoints:
  POST /api/jobs              -> upload video, returns {job_id}
  GET  /api/jobs/{id}         -> job status + log tail
  GET  /api/jobs/{id}/result  -> download panorama_output.jpg
  GET  /healthz               -> liveness probe

Long videos: handled by streaming uploads to disk + running the pipeline in a
ThreadPoolExecutor so the HTTP request returns immediately. Client polls.

Deploy on a real server (Render, Railway, Fly.io, a VPS). NOT on Netlify
Functions -- they cap at 10-26s and 6 MB request bodies.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import panorama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JOBS_ROOT = Path(os.environ.get("JOBS_ROOT", "./jobs")).resolve()
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

# Hard cap on uploaded video size (bytes). Default 1 GB; tune to host limits.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024)))

# How many panoramas to render concurrently. Stitching is CPU/RAM heavy; keep low.
WORKER_CONCURRENCY = int(os.environ.get("WORKER_CONCURRENCY", "1"))

# Comma-separated list of allowed origins for CORS (your Netlify URL).
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "*").split(",")
    if o.strip()
]

ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}

# Strict pattern for job IDs — prevents path traversal.
_JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# How long to keep finished jobs on disk (seconds). Cleared lazily on access.
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", str(60 * 60)))

# Thread-local storage for per-job log capture.
_thread_local = threading.local()

# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

logger = logging.getLogger("panorama.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | error
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    log_lines: list = field(default_factory=list)
    result: Optional[dict] = None
    error: Optional[str] = None
    video_path: Optional[Path] = None
    output_dir: Optional[Path] = None


JOBS: dict[str, Job] = {}
JOBS_LOCK = Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=WORKER_CONCURRENCY)


def _append_log(job: Job, line: str) -> None:
    job.log_lines.append(line)
    # keep last 500 lines only
    if len(job.log_lines) > 500:
        job.log_lines = job.log_lines[-500:]
    job.updated_at = time.time()


def _set_status(job: Job, status: str) -> None:
    job.status = status
    job.updated_at = time.time()


def _gc_old_jobs() -> None:
    """Remove finished jobs older than JOB_TTL_SECONDS."""
    cutoff = time.time() - JOB_TTL_SECONDS
    with JOBS_LOCK:
        stale = [
            jid for jid, j in JOBS.items()
            if j.status in ("done", "error") and j.updated_at < cutoff
        ]
        for jid in stale:
            j = JOBS.pop(jid)
            if j.output_dir and j.output_dir.exists():
                shutil.rmtree(j.output_dir, ignore_errors=True)


def _process_job(job_id: str) -> None:
    """Worker: run the full panorama pipeline for one job."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return

    _set_status(job, "running")
    _append_log(job, "Worker picked up job")

    # Capture panorama.log() output by monkey-patching for this thread's calls.
    original_log = panorama.log

    def capture(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        _append_log(job, f"[{ts}] {msg}")
        original_log(msg)

    panorama.log = capture
    try:
        frames_dir = job.output_dir / "frames"
        result = panorama.run_pipeline(
            video_path=job.video_path,
            output_dir=job.output_dir,
            frames_dir=frames_dir,
        )
        job.result = result
        _set_status(job, "done")
        _append_log(job, f"Job finished: {result}")
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        job.error = str(e)
        _set_status(job, "error")
        _append_log(job, f"FATAL: {e}")
    finally:
        panorama.log = original_log
        # delete the uploaded video file -- result image is what we keep
        if job.video_path and job.video_path.exists():
            try:
                job.video_path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Video-to-Panorama API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _validate_job_id(job_id: str) -> None:
    """Raise 400 if job_id doesn't match expected format (path traversal guard)."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(400, "Invalid job id format")


@app.get("/healthz")
def healthz():
    return {"ok": True, "active_jobs": sum(1 for j in JOBS.values() if j.status == "running")}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)):
    """Accept an uploaded video and enqueue a panorama job."""
    _gc_old_jobs()

    ext = Path(file.filename or 'video').suffix.lower()
    if not ext or ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported file type: {ext or 'unknown'}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / f"input{ext}"

    # Stream upload to disk to handle multi-GB videos without buffering in RAM.
    bytes_written = 0
    try:
        with video_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    out.close()
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(
                        413,
                        f"Upload exceeds {MAX_UPLOAD_BYTES} byte limit",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, f"Failed to save upload: {e}") from e

    job = Job(id=job_id, video_path=video_path, output_dir=job_dir)
    with JOBS_LOCK:
        JOBS[job_id] = job
    _append_log(job, f"Received {bytes_written} bytes, queued for processing")

    EXECUTOR.submit(_process_job, job_id)
    return {"job_id": job_id, "status": job.status, "size_bytes": bytes_written}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    _validate_job_id(job_id)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    return {
        "job_id": job.id,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "log_tail": job.log_lines[-50:],
        "result": job.result,
        "error": job.error,
    }


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str):
    _validate_job_id(job_id)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    if job.status != "done" or not job.result:
        raise HTTPException(409, f"Job not finished (status={job.status})")
    jpg_path = Path(job.result["jpg_path"])
    if not jpg_path.exists():
        raise HTTPException(410, "Result file no longer available")
    return FileResponse(
        jpg_path,
        media_type="image/jpeg",
        filename=f"panorama_{job_id}.jpg",
    )


# Serve the static frontend in dev so you can run everything with one command.
# In production on Netlify, the frontend is hosted there and points to this API.
_WEB_DIR = Path(__file__).parent / "web"
if _WEB_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
