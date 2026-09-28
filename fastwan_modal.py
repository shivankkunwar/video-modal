"""
FastWan 2.2 TI2V-5B (Wan 2.2 5B, DMD-distilled to 3 steps, Apache 2.0) on Modal.

Same shape as qwen-image-modal: a GPU class plus a small CPU `api` in front.

Plain diffusers can't run this checkpoint (its model_index names FastVideo's
WanDMDPipeline), so the GPU image uses the `fastvideo` package. It gets its own
app because fastvideo pins torch==2.12.0 and many other packages.

Text-to-video only: in fastvideo 0.2.1 the 3-step DMD denoising stage never
applies the start-image mask (only the full-step stage does), so a start image
would be silently ignored. Use LTX-2.5 for image-to-video.

One-time setup:
    modal run fastwan_modal.py::download_weights    # ~24 GB -> Volume, CPU only

CLI (MP4 files land in fastwan_outputs/, no audio):
    modal run fastwan_modal.py --prompt "a neon-lit Tokyo alley in heavy rain at night"
    modal run fastwan_modal.py --prompts-file prompts.txt

Deploy the API:
    modal deploy fastwan_modal.py
"""

import base64
import random
import time
from typing import Literal

import modal
from pydantic import BaseModel, Field, model_validator

APP_NAME = "fastwan-22-video"
MODEL_ID = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
GPU = "L40S"  # 48 GB; the 5B model peaks at ~23 GB on a 4090 with offload
CACHE_DIR = "/models"
FPS = 24  # the model's training frame rate

AspectRatio = Literal["16:9", "9:16", "1:1"]
# Trained at 1280x704; other sizes work but quality may drop (model card).
SIZE_PRESETS = {"16:9": (1280, 704), "9:16": (704, 1280), "1:1": (960, 960)}
MAX_FRAMES = 121  # 5 s at 24 fps, the training length
MAX_VIDEOS_PER_REQUEST = 4
SECONDS_PER_CLIP = 41  # measured 2026-09-28, 1280x704x121 on L40S (for the UI's estimate)


class VideoRequest(BaseModel):
    """Everything this checkpoint can do, in one request. Only `prompt` is required."""

    prompt: str = Field(min_length=1, max_length=4000)
    aspect_ratio: AspectRatio | None = Field(None, description="Preset shape. Default 16:9 (1280x704).")
    width: int | None = Field(None, ge=256, le=1280, multiple_of=32, description="Exact width; overrides aspect_ratio.")
    height: int | None = Field(None, ge=256, le=1280, multiple_of=32, description="Exact height. Set with width.")
    duration: float = Field(5.0, ge=1.0, le=5.0, description="Seconds (the model is trained on 5 s clips).")
    num_videos: int = Field(1, ge=1, le=MAX_VIDEOS_PER_REQUEST, description="Variations; video k uses seed + k.")
    seed: int | None = Field(None, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def _check(self):
        if (self.width is None) != (self.height is None):
            raise ValueError("set both width and height, or neither")
        if self.width and self.aspect_ratio:
            raise ValueError("use either width/height or aspect_ratio, not both")
        return self

    def num_frames(self) -> int:
        """Wan needs 4k+1 frames."""
        return min(round(self.duration * FPS / 4) * 4 + 1, MAX_FRAMES)

    def target_size(self) -> tuple[int, int]:
        if self.width:
            return self.width, self.height
        return SIZE_PRESETS[self.aspect_ratio or "16:9"]


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")  # imageio-ffmpeg / opencv runtime libs
    .uv_pip_install("fastvideo==0.2.1", "pydantic>=2", "huggingface_hub[hf_xet]")
    .env({
        "HF_HUB_CACHE": CACHE_DIR,
        "HF_XET_HIGH_PERFORMANCE": "1",
        # Full-attention checkpoint; torch SDPA avoids building flash-attn.
        "FASTVIDEO_ATTENTION_BACKEND": "TORCH_SDPA",
    })
)
api_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("fastapi[standard]", "pydantic>=2")

app = modal.App(APP_NAME, image=image)
weights = modal.Volume.from_name("fastwan-22-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
progress = modal.Dict.from_name("fastwan-22-progress", create_if_missing=True)


@app.function(volumes={CACHE_DIR: weights}, secrets=[hf_secret], cpu=4, memory=8192, timeout=60 * 60)
def download_weights():
    from huggingface_hub import snapshot_download

    t0 = time.time()
    # Full repo: FastVideo rejects a cached snapshot with any file missing, even the README images.
    path = snapshot_download(MODEL_ID)
    weights.commit()
    print(f"Downloaded {MODEL_ID} to {path} in {time.time() - t0:.0f}s")


@app.cls(
    gpu=GPU,
    volumes={CACHE_DIR: weights},
    secrets=[hf_secret],
    scaledown_window=60,   # idle GPU time was 79% of the Qwen bill; batch prompts instead
    max_containers=1,      # hard cap on burn rate
    timeout=20 * 60,
)
class FastWan:
    @modal.enter()
    def load(self):
        import os

        from fastvideo import VideoGenerator

        os.environ["HF_HUB_OFFLINE"] = "1"  # weights come from the Volume, never from HF at request time
        t0 = time.time()
        # Pass the repo id (not a local path): FastVideo picks the DMD config from it.
        self.gen = VideoGenerator.from_pretrained(MODEL_ID, num_gpus=1)
        print(f"Model loaded in {time.time() - t0:.1f}s")

    @modal.method()
    def generate(self, req: dict) -> dict:
        import glob

        from fastvideo.api import GenerationRequest, OutputConfig, SamplingConfig

        r = VideoRequest(**req)
        width, height = r.target_size()
        frames = r.num_frames()
        base_seed = r.seed if r.seed is not None else random.randint(0, 2**32 - 1)
        job_id = modal.current_function_call_id()

        t_start = time.time()
        videos = []
        for k in range(r.num_videos):
            seed = (base_seed + k) % 2**32
            if job_id:
                # FastVideo runs the model in a worker process, so there is no per-step
                # callback; the UI shows time against SECONDS_PER_CLIP instead.
                try:
                    progress.put(job_id, {"video": k + 1, "videos": r.num_videos, "stage": "denoising",
                                          "started_at": time.time(), "expected_seconds": SECONDS_PER_CLIP})
                except Exception:
                    pass
            out_dir = f"/tmp/out_{seed}"
            t0 = time.time()
            result = self.gen.generate(GenerationRequest(
                prompt=r.prompt,
                sampling=SamplingConfig(
                    seed=seed, width=width, height=height, num_frames=frames, fps=FPS,
                    num_inference_steps=3,  # DMD timesteps 1000,757,522 come from the model's config
                ),
                output=OutputConfig(output_path=out_dir, output_video_name=str(seed),
                                    save_video=True, return_frames=False),
            ))
            seconds = round(time.time() - t0, 1)
            path = getattr(result, "video_path", None) or sorted(glob.glob(f"{out_dir}/*.mp4"))[-1]
            # One line per clip, same shape as the Qwen app, so usage can be counted from logs.
            print(f"{width}x{height} {frames}f, seed {seed}: {seconds}s")
            with open(path, "rb") as f:
                videos.append({
                    "seed": seed, "width": width, "height": height, "num_frames": frames,
                    "fps": FPS, "duration": round(frames / FPS, 2), "has_audio": False,
                    "seconds": seconds, "data": f.read(),
                })

        return {"videos": videos, "prompt_used": r.prompt, "enhanced": False,
                "image_to_video": False, "seconds": round(time.time() - t_start, 1)}


def serialize_result(result: dict, job_id: str | None = None) -> dict:
    videos = []
    for i, v in enumerate(result["videos"]):
        meta = {k: val for k, val in v.items() if k != "data"}
        if job_id:
            meta["url"] = f"/v1/jobs/{job_id}/videos/{i}"
        else:
            meta["data_url"] = "data:video/mp4;base64," + base64.b64encode(v["data"]).decode()
        videos.append(meta)
    return {**{k: val for k, val in result.items() if k != "videos"}, "videos": videos}


@app.function(image=api_image, cpu=0.5, memory=1024, max_containers=1)
@modal.concurrent(max_inputs=50)
@modal.asgi_app(requires_proxy_auth=True)  # callers need Modal-Key / Modal-Secret
def api():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, Response

    web = FastAPI(title="FastWan 2.2 Video API", version="1.0.0",
                  description="3-step text-to-video (no audio). Use /v1/jobs for UIs, /v1/generate for scripts.")
    gpu = FastWan()

    # Finished results never change, so keep the last few in memory: <video> seeks
    # send many Range requests, and each FunctionCall.get() re-downloads every clip (~5 s).
    from collections import OrderedDict
    cache: OrderedDict = OrderedDict()

    async def job_result(job_id: str):
        if job_id in cache:
            cache.move_to_end(job_id)
            return cache[job_id]
        result = await _job_result(job_id)
        if result is not None:
            cache[job_id] = result
            while len(cache) > 8:
                cache.popitem(last=False)
        return result

    async def _job_result(job_id: str):
        if not job_id.startswith("fc-"):
            raise HTTPException(404, "unknown job id")
        try:
            return await modal.FunctionCall.from_id(job_id).get.aio(timeout=0)
        except (TimeoutError, modal.exception.TimeoutError):
            return None
        except modal.exception.OutputExpiredError:
            raise HTTPException(404, "job results expired (they're kept for 7 days)")
        except modal.exception.NotFoundError:
            raise HTTPException(404, "unknown job id")

    @web.get("/health")
    def health():
        return {"ok": True, "model": MODEL_ID}

    @web.get("/v1/options")
    def options():
        return {
            "model": MODEL_ID,
            "size_presets": {ar: {"width": w, "height": h} for ar, (w, h) in SIZE_PRESETS.items()},
            "frame_rates": [FPS],
            "limits": {"max_seconds": MAX_FRAMES / FPS, "max_side": 1280, "side_multiple": 32,
                       "max_videos_per_request": MAX_VIDEOS_PER_REQUEST},
            "features": {"audio": False, "image_to_video": False, "auto_duration": False,
                         "hd": False, "enhance_prompt": False},
            "seconds_per_clip": SECONDS_PER_CLIP,
            "request_schema": VideoRequest.model_json_schema(),
        }

    @web.post("/v1/jobs", status_code=202)
    async def create_job(req: VideoRequest):
        call = await gpu.generate.spawn.aio(req.model_dump())
        return {"id": call.object_id, "status": "starting", "status_url": f"/v1/jobs/{call.object_id}"}

    @web.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str):
        try:
            result = await job_result(job_id)
        except HTTPException:
            raise
        except Exception as e:
            return {"id": job_id, "status": "failed", "error": f"{type(e).__name__}: {e}"}
        if result is None:
            p = await progress.get.aio(job_id)
            return JSONResponse(status_code=202, content={
                "id": job_id, "status": "running" if p else "starting", "progress": p,
            })
        return {"id": job_id, "status": "done", **serialize_result(result, job_id)}

    @web.get("/v1/jobs/{job_id}/videos/{index}")
    async def get_job_video(job_id: str, index: int, request: Request):
        try:
            result = await job_result(job_id)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(409, f"job failed: {e}")
        if result is None:
            raise HTTPException(409, "job is still running")
        if not 0 <= index < len(result["videos"]):
            raise HTTPException(404, "no video at that index")
        v = result["videos"][index]
        return mp4_response(v["data"], f"fastwan_{v['seed']}.mp4", request.headers.get("range"))

    @web.post("/v1/generate")
    async def generate_sync(req: VideoRequest):
        return serialize_result(await gpu.generate.remote.aio(req.model_dump()))

    def mp4_response(data: bytes, filename: str, range_header: str | None):
        headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=86400",
                   "Content-Disposition": f'inline; filename="{filename}"'}
        size = len(data)
        if range_header and range_header.startswith("bytes="):
            first, _, last = range_header[6:].split(",")[0].partition("-")
            start = int(first) if first else max(size - int(last), 0)
            end = min(int(last) if first and last else size - 1, size - 1)
            if start > end:
                return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            return Response(data[start:end + 1], status_code=206, media_type="video/mp4", headers=headers)
        return Response(data, media_type="video/mp4", headers=headers)

    return web


@app.local_entrypoint()
def main(
    prompt: str = "A neon-lit alley in futuristic Tokyo during a heavy rainstorm at night. Puddles reflect "
                  "glowing kanji signs, a woman in a translucent raincoat walks past a steaming food cart.",
    prompts_file: str = "",
    aspect_ratio: str = "",
    duration: float = 5.0,
    n: int = 1,
    seed: int = -1,
    out_dir: str = "fastwan_outputs",
):
    from pathlib import Path

    prompts = ([p.strip() for p in Path(prompts_file).read_text().splitlines() if p.strip()]
               if prompts_file else [prompt])
    reqs = [VideoRequest(prompt=p, aspect_ratio=aspect_ratio or None, duration=duration, num_videos=n,
                         seed=seed if seed >= 0 else None).model_dump() for p in prompts]

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    stamp = int(time.time())
    t0 = time.time()
    count = 0
    for i, res in enumerate(FastWan().generate.map(reqs)):
        for k, v in enumerate(res["videos"]):
            path = out / f"{stamp}_{i:03d}_{k}_seed{v['seed']}.mp4"
            path.write_bytes(v["data"])
            count += 1
            print(f"saved {path} ({v['width']}x{v['height']}, {v['seconds']}s)")
    print(f"{count} clip(s) in {time.time() - t0:.0f}s (includes cold start)")
