# Video → Panorama

Local CLI + web UI that turns a horizontal panning video into a single stitched panoramic image. Filters out blur, motion artifacts and near-duplicate frames before stitching.

## Files

- `panorama.py` — pipeline (also a CLI: `python panorama.py video.mp4`)
- `server.py` — FastAPI backend (upload + job polling + result download)
- `web/index.html` — drag-and-drop frontend (deployable to Netlify)
- `requirements.txt`, `Dockerfile`, `netlify.toml`

## Run locally

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn server:app --reload
```

Then open http://localhost:8000 — the backend serves the frontend in dev mode. The "API base URL" field in the UI defaults to the same origin.

## Long videos (5–10 min and beyond)

Yes — supported. The pipeline samples frames adaptively rather than extracting every single one:

- `MAX_CANDIDATE_FRAMES = 3000` in `panorama.py` caps how many raw frames are written to disk. A 10‑min @ 30 fps video (18,000 frames) is stride‑sampled down to 3000 candidates.
- Filters (blur / motion / dedup) then narrow that to ≤ `MAX_FRAMES_FOR_STITCH = 300` for OpenCV's stitcher.
- Uploads stream to disk in 1 MB chunks, so RAM stays low even for multi‑GB files.
- The HTTP request returns immediately with a `job_id`; the frontend polls `/api/jobs/{id}` every 1.5s for status + log tail.

Practical notes for long clips:
- Stitching 300 frames at full resolution is CPU/RAM heavy. Budget 1–4 GB RAM and several minutes of CPU per job. Set `WORKER_CONCURRENCY=1` unless you have a beefy host.
- Increase `MAX_UPLOAD_BYTES` env var if you need files larger than the default 1 GB.
- For very long pans (15 min+), drop `MAX_FRAMES_FOR_STITCH` to ~200 if you hit OOMs.

## Deploying

### About Netlify

**Netlify cannot run the pipeline itself.** Netlify Functions cap at 10s (26s on Pro) and a 6 MB request body — useless for multi‑minute video processing. Use Netlify only for the frontend.

### Recommended split

1. **Backend** on a real host that supports long-running Python + multi‑GB uploads:
   - Render, Railway, Fly.io, a $5 Hetzner/DO VPS, etc.
   - Build with the included `Dockerfile`.
   - Set env vars: `CORS_ORIGINS=https://your-site.netlify.app`, optionally `MAX_UPLOAD_BYTES`, `WORKER_CONCURRENCY`, `JOB_TTL_SECONDS`.
   - Mount a persistent volume at `/data` if your host is ephemeral.

2. **Frontend** on Netlify:
   - Edit `netlify.toml` and replace `YOUR-BACKEND-HOST.example.com` with your backend URL.
   - Drag‑and‑drop the `web/` folder into the Netlify UI, or push this repo and set "Publish directory" to `web`.
   - The `/api/*` redirect proxies through Netlify so the browser never hits CORS.

### Single-host alternative

Skip Netlify entirely and just run the Docker image on Render/Fly/VPS — the backend serves the frontend on `/`. Simpler, fewer moving parts.

## Tunable thresholds (top of `panorama.py`)

| Constant | Default | Purpose |
| --- | --- | --- |
| `BLUR_RATIO` | 0.5 | Drop frame if Laplacian variance < mean * ratio |
| `MOTION_FLOW_THRESHOLD` | 8.0 | Max mean optical flow magnitude (px/frame) |
| `PHASH_MIN_DISTANCE` | 8 | Min Hamming distance vs last kept frame |
| `MAX_CANDIDATE_FRAMES` | 3000 | Cap on raw frames extracted from disk |
| `MAX_FRAMES_FOR_STITCH` | 300 | Cap on frames fed to OpenCV stitcher |
