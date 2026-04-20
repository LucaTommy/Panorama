"""
Video-to-Panorama Pipeline
Usage: python panorama.py <video_path>
"""

import os
import sys
import shutil
import time
import atexit
import tempfile
import subprocess
from pathlib import Path

# =============================================================================
# CONFIGURABLE THRESHOLDS
# =============================================================================

FRAMES_DIR = Path(tempfile.gettempdir()) / "frames"  # temp dir for extracted frames

# 3a. Blur detection: dynamic threshold = mean(variances) * BLUR_RATIO
BLUR_RATIO = 0.5  # frames with Laplacian variance below mean*ratio are dropped

# 3b. Motion blur (optical flow): drop frame if mean flow magnitude exceeds this
MOTION_FLOW_THRESHOLD = 8.0  # pixels/frame
OPTICAL_FLOW_DOWNSCALE = 0.25  # downscale for speed when computing flow

# 3c. Perceptual dedup: drop frame if pHash Hamming distance to last kept < this
PHASH_MIN_DISTANCE = 8

# 3d. Temporal subsampling cap (prevents stitcher OOM)
MAX_FRAMES_FOR_STITCH = 300

# Long-video support: cap how many raw frames we ever extract from disk.
# For a 10-min @ 30fps video that's 18,000 frames; we only need ~10x the
# stitcher cap as candidates, so the filter has headroom to reject blurs/dups.
MAX_CANDIDATE_FRAMES = 3000

# Output files
OUTPUT_JPG = "panorama_output.jpg"
OUTPUT_PNG = "panorama_output.png"
JPG_QUALITY = 95
PNG_MAX_PIXELS = 50_000_000  # only write PNG if image has fewer than this many pixels

# Stitcher memory guard: if frames exceed this pixel count, downscale for stitching.
MAX_FRAME_PIXELS = 2_000_000  # ~1920x1080; frames above this get scaled down

# Self-verification
MIN_MEAN_PIXEL_VALUE = 20  # final image must be brighter than this on average


# =============================================================================
# UTILITIES
# =============================================================================

def log(msg: str) -> None:
    """Print a timestamped progress message to stdout."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def cleanup_frames_dir() -> None:
    """Remove the temporary frames directory."""
    try:
        if FRAMES_DIR.exists():
            shutil.rmtree(FRAMES_DIR, ignore_errors=True)
            log(f"Cleaned up {FRAMES_DIR}")
    except Exception as e:
        log(f"WARN: failed cleaning frames dir: {e}")


# =============================================================================
# STEP 1 — ENVIRONMENT SETUP
# =============================================================================

REQUIRED_PACKAGES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "PIL": "Pillow",
    "imagehash": "ImageHash",
    "scipy": "scipy",
}


def ensure_dependencies() -> None:
    """Verify (and install if missing) required packages."""
    log("STEP 1: Checking dependencies...")
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        log(f"Installing missing packages: {missing}")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", *missing]
            )
        except subprocess.CalledProcessError as e:
            log(f"ERROR: pip install failed: {e}")
            sys.exit(1)

    # Print versions
    import cv2
    import numpy as np
    import PIL
    import imagehash
    import scipy

    log(f"Python   : {sys.version.split()[0]}")
    log(f"OpenCV   : {cv2.__version__}")
    log(f"NumPy    : {np.__version__}")
    log(f"Pillow   : {PIL.__version__}")
    log(f"imagehash: {imagehash.__version__}")
    log(f"SciPy    : {scipy.__version__}")


# =============================================================================
# STEP 2 — FRAME EXTRACTION
# =============================================================================

def extract_frames(video_path: Path, frames_dir: Path = FRAMES_DIR) -> list:
    """Extract frames into ``frames_dir`` using an adaptive stride.

    For long videos we don't need every single frame -- just enough candidates
    for the filter stage to pick MAX_FRAMES_FOR_STITCH good ones. We compute a
    stride so we end up with at most ``MAX_CANDIDATE_FRAMES`` evenly spaced
    samples. Falls back to reading every frame for short videos.
    """
    import cv2

    log(f"STEP 2: Extracting frames from {video_path}")
    if frames_dir.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = (total / fps) if fps > 0 else 0.0

    stride = 1
    if total > MAX_CANDIDATE_FRAMES:
        stride = max(1, total // MAX_CANDIDATE_FRAMES)
    log(
        f"  video info: total_frames={total}, fps={fps:.2f}, "
        f"duration={duration:.1f}s, stride={stride}"
    )

    frame_paths = []
    idx = 0
    saved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                out_path = frames_dir / f"frame_{saved:06d}.jpg"
                try:
                    cv2.imwrite(str(out_path), frame)
                    frame_paths.append(out_path)
                    saved += 1
                except Exception as e:
                    log(f"WARN: failed to write frame {idx}: {e}")
            idx += 1
    finally:
        cap.release()

    log(f"Extracted {len(frame_paths)} candidate frames (read {idx} total)")
    if not frame_paths:
        raise RuntimeError("No frames could be read from the video.")
    return frame_paths


# =============================================================================
# STEP 3 — SMART FRAME FILTERING
# =============================================================================

def filter_frames(frame_paths: list) -> list:
    """Apply blur, motion, dedup, and subsample filters. Returns kept paths."""
    import cv2
    import numpy as np
    from PIL import Image
    import imagehash

    log("STEP 3: Filtering frames...")

    discarded = {"blurry": 0, "motion": 0, "duplicate": 0, "subsampled": 0}

    # ---- 3a: Blur detection (Laplacian variance) ----
    log("  3a: Computing Laplacian variance for blur detection...")
    variances = []
    grays = []  # cache grayscales for optical flow
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            variances.append(0.0)
            grays.append(None)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        variances.append(float(var))
        grays.append(gray)

    mean_var = float(np.mean([v for v in variances if v > 0])) if variances else 0.0
    blur_threshold = mean_var * BLUR_RATIO
    log(f"     mean variance={mean_var:.2f}, threshold={blur_threshold:.2f}")

    sharp_indices = []
    for i, v in enumerate(variances):
        if v >= blur_threshold and grays[i] is not None:
            sharp_indices.append(i)
        else:
            discarded["blurry"] += 1
    log(f"     blur passed: {len(sharp_indices)}/{len(frame_paths)}")

    # ---- 3b: Motion blur via optical flow ----
    log("  3b: Computing optical flow magnitudes...")
    motion_passed = []
    prev_small = None
    for i in sharp_indices:
        gray = grays[i]
        small = cv2.resize(
            gray,
            None,
            fx=OPTICAL_FLOW_DOWNSCALE,
            fy=OPTICAL_FLOW_DOWNSCALE,
            interpolation=cv2.INTER_AREA,
        )
        if prev_small is None:
            motion_passed.append(i)
            prev_small = small
            continue
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_small, small, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mean_mag = float(mag.mean()) / OPTICAL_FLOW_DOWNSCALE  # rescale to original
        except cv2.error as e:
            log(f"WARN: optical flow failed at frame {i}: {e}")
            mean_mag = 0.0

        if mean_mag <= MOTION_FLOW_THRESHOLD:
            motion_passed.append(i)
            prev_small = small
        else:
            discarded["motion"] += 1
            # do not update prev_small: compare next frame to last good one
    log(f"     motion passed: {len(motion_passed)}/{len(sharp_indices)}")

    # Free grayscale cache; not needed anymore
    grays = None

    # ---- 3c: Perceptual dedup via pHash ----
    log("  3c: Perceptual dedup via pHash...")
    dedup_passed = []
    last_hash = None
    for i in motion_passed:
        try:
            with Image.open(frame_paths[i]) as im:
                h = imagehash.phash(im)
        except Exception as e:
            log(f"WARN: phash failed at frame {i}: {e}")
            continue
        if last_hash is None or (h - last_hash) >= PHASH_MIN_DISTANCE:
            dedup_passed.append(i)
            last_hash = h
        else:
            discarded["duplicate"] += 1
    log(f"     dedup passed: {len(dedup_passed)}/{len(motion_passed)}")

    # ---- 3d: Temporal subsampling cap ----
    if len(dedup_passed) > MAX_FRAMES_FOR_STITCH:
        log(f"  3d: Subsampling {len(dedup_passed)} -> {MAX_FRAMES_FOR_STITCH}")
        idxs = np.linspace(0, len(dedup_passed) - 1, MAX_FRAMES_FOR_STITCH).astype(int)
        idxs = sorted(set(idxs.tolist()))
        subsampled = [dedup_passed[k] for k in idxs]
        discarded["subsampled"] = len(dedup_passed) - len(subsampled)
        final_indices = subsampled
    else:
        final_indices = dedup_passed

    kept = [frame_paths[i] for i in final_indices]
    total_discarded = sum(discarded.values())
    log(
        f"FILTER RESULT: kept={len(kept)} discarded={total_discarded} "
        f"(blurry={discarded['blurry']}, motion={discarded['motion']}, "
        f"duplicate={discarded['duplicate']}, subsampled={discarded['subsampled']})"
    )

    if len(kept) < 2:
        raise RuntimeError(
            f"Too few frames remain after filtering ({len(kept)}). "
            "Loosen thresholds or supply a slower/sharper video."
        )

    return kept


# =============================================================================
# STEP 4 — PANORAMIC STITCHING
# =============================================================================

def _interpret_status(status_code: int) -> str:
    import cv2
    mapping = {
        cv2.Stitcher_OK: "OK",
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "ERR_NEED_MORE_IMGS",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "ERR_HOMOGRAPHY_EST_FAIL",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "ERR_CAMERA_PARAMS_ADJUST_ERR",
    }
    return mapping.get(status_code, f"UNKNOWN({status_code})")


def stitch_frames(frame_paths: list):
    """Run cv2.Stitcher on the kept frames. Returns the stitched image (BGR ndarray)."""
    import cv2

    log(f"STEP 4: Stitching {len(frame_paths)} frames (SCANS mode)...")

    images = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        # Downscale huge frames to prevent stitcher OOM
        h_f, w_f = img.shape[:2]
        if h_f * w_f > MAX_FRAME_PIXELS:
            scale = (MAX_FRAME_PIXELS / (h_f * w_f)) ** 0.5
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            log(f"  downscaled {w_f}x{h_f} -> {img.shape[1]}x{img.shape[0]} for stitching")
        images.append(img)
    if len(images) < 2:
        raise RuntimeError("Not enough readable frames to stitch.")

    def _try(mode, label):
        log(f"  Trying stitcher mode={label}")
        try:
            stitcher = cv2.Stitcher_create(mode=mode)
        except AttributeError:
            stitcher = cv2.Stitcher.create(mode)
        try:
            status, pano = stitcher.stitch(images)
        except cv2.error as e:
            log(f"  cv2.error during stitching: {e}")
            return None, -1
        return pano, status

    pano, status = _try(cv2.Stitcher_SCANS, "SCANS")
    label = _interpret_status(status) if status >= 0 else "EXCEPTION"
    log(f"  Stitcher status: {label}")

    if status == cv2.Stitcher_OK and pano is not None:
        return pano

    # Diagnose
    if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        log("  Diagnosis: too few overlapping frames. Loosen filter thresholds.")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        log("  Diagnosis: homography estimation failed. Use a slower, smoother pan.")
    elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        log("  Diagnosis: camera params adjust failed. Trying PANORAMA mode...")

    # Auto-retry once with PANORAMA mode
    pano2, status2 = _try(cv2.Stitcher_PANORAMA, "PANORAMA")
    label2 = _interpret_status(status2) if status2 >= 0 else "EXCEPTION"
    log(f"  Stitcher status (retry): {label2}")
    if status2 == cv2.Stitcher_OK and pano2 is not None:
        return pano2

    raise RuntimeError(
        f"Stitching failed in both SCANS ({label}) and PANORAMA ({label2}) modes."
    )


# =============================================================================
# STEP 5 — POST-PROCESSING & EXPORT
# =============================================================================

def crop_black_borders(image):
    """Crop black borders using contour-based largest-bounding-rect."""
    import cv2
    import numpy as np

    log("STEP 5: Cropping black borders...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold non-black pixels (use 5 not 1 to ignore JPEG compression noise)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        log("  WARN: no contours found, returning uncropped image")
        return image
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cropped = image[y:y + h, x:x + w]
    log(f"  cropped to {cropped.shape[1]}x{cropped.shape[0]}")
    return cropped


def export_image(image, output_dir: Path = Path(".")) -> Path:
    """Save panorama as JPG (always) and PNG (if small enough). Returns JPG path."""
    import cv2
    from PIL import Image

    log("STEP 5: Exporting image...")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    jpg_path = output_dir / OUTPUT_JPG
    pil_img.save(jpg_path, "JPEG", quality=JPG_QUALITY)

    h, w = image.shape[:2]
    total_pixels = h * w
    if total_pixels < PNG_MAX_PIXELS:
        png_path = output_dir / OUTPUT_PNG
        pil_img.save(png_path, "PNG")
        log(f"  saved PNG: {png_path}")
    else:
        log(f"  skipped PNG (image has {total_pixels} pixels >= {PNG_MAX_PIXELS})")

    size_mb = jpg_path.stat().st_size / (1024 * 1024)
    log(f"  final dimensions: {w}x{h}, JPG size: {size_mb:.2f} MB")
    return jpg_path


# =============================================================================
# STEP 6 — SELF-VERIFICATION
# =============================================================================

def verify(image, jpg_path: Path, original_frame_width: int) -> bool:
    """Run sanity checks on the output. Returns True if all pass."""
    import numpy as np

    log("STEP 6: Self-verification...")
    ok = True

    h, w = image.shape[:2]
    if w <= original_frame_width:
        log(
            f"  FAIL: output width ({w}) <= original frame width ({original_frame_width}). "
            "Stitcher may not have merged frames horizontally."
        )
        ok = False
    else:
        log(f"  PASS: output width ({w}) > original frame width ({original_frame_width})")

    mean_val = float(np.mean(image))
    if mean_val <= MIN_MEAN_PIXEL_VALUE:
        log(f"  FAIL: image is mostly black (mean={mean_val:.2f}).")
        ok = False
    else:
        log(f"  PASS: mean pixel value={mean_val:.2f}")

    if not jpg_path.exists() or jpg_path.stat().st_size == 0:
        log(f"  FAIL: output file missing or empty: {jpg_path}")
        ok = False
    else:
        log(f"  PASS: file exists ({jpg_path.stat().st_size} bytes)")

    return ok


# =============================================================================
# PROGRAMMATIC ENTRYPOINT (used by the web server)
# =============================================================================

def run_pipeline(video_path: Path, output_dir: Path, frames_dir: Path) -> dict:
    """Run the full pipeline against ``video_path`` writing artifacts to
    ``output_dir`` and using ``frames_dir`` as scratch space.

    Returns a dict with keys: jpg_path, width, height, size_mb, verified.
    Raises on fatal errors so callers can surface them.
    """
    import cv2

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    frames_dir = Path(frames_dir)

    cropped = None
    jpg_path = None
    verified = False
    try:
        frame_paths = extract_frames(video_path, frames_dir=frames_dir)
        first = cv2.imread(str(frame_paths[0]))
        original_width = first.shape[1] if first is not None else 0
        kept = filter_frames(frame_paths)
        pano = stitch_frames(kept)
        cropped = crop_black_borders(pano)
        jpg_path = export_image(cropped, output_dir=output_dir)
        verified = verify(cropped, jpg_path, original_width)
    finally:
        # always purge scratch frames for this job
        shutil.rmtree(frames_dir, ignore_errors=True)

    if cropped is None or jpg_path is None:
        raise RuntimeError("Pipeline did not produce an output image.")

    h, w = cropped.shape[:2]
    size_mb = jpg_path.stat().st_size / (1024 * 1024)
    return {
        "jpg_path": str(jpg_path),
        "width": int(w),
        "height": int(h),
        "size_mb": round(size_mb, 2),
        "verified": bool(verified),
    }


# =============================================================================
# MAIN (CLI)
# =============================================================================

def main() -> int:
    atexit.register(cleanup_frames_dir)

    if len(sys.argv) < 2:
        print("Usage: python panorama.py <video_path>")
        return 2
    video_path = Path(sys.argv[1]).expanduser().resolve()
    if not video_path.exists():
        print(f"ERROR: video file does not exist: {video_path}")
        return 2

    try:
        ensure_dependencies()
    except Exception as e:
        log(f"FATAL: dependency setup failed: {e}")
        return 1

    try:
        result = run_pipeline(
            video_path, output_dir=Path("."), frames_dir=FRAMES_DIR
        )
    except Exception as e:
        log(f"FATAL: pipeline failed: {e}")
        return 1

    if not result["verified"]:
        log("Output failed self-verification. Consider a slower pan or looser filters.")
        return 1

    print(
        f"✅ panorama_output.jpg — {result['width']}x{result['height']}px "
        f"— {result['size_mb']:.2f}MB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
