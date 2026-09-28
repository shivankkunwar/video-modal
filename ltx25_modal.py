"""
LTX-2.5 (Lightricks, 22B, video + synced audio) on Modal.

Same shape as qwen-image-modal:
  * LTX25 (GPU, RTX PRO 6000) - loads the pipeline once per container.
  * api (small CPU container) - FastAPI app that validates requests and hands
    real work to the GPU, so typos, option lookups and status polling never
    wake the GPU. Interactive docs at <api url>/docs.

Model features exposed: text-to-video, image-to-video (start frame), audio,
auto duration (the model's duration head), HD two-stage (latent upsampler),
prompt enhancement (Gemma 4 enhancer), aspect presets or exact size,
frame rate, variations, seed.

One-time setup (the repo is gated: accept the terms on
https://huggingface.co/Lightricks/LTX-2.5-Diffusers with the account whose
token is in the `huggingface` Modal secret):
    modal run ltx25_modal.py::download_weights      # ~92 GB -> Volume, CPU only

CLI (MP4 files land in ltx_outputs/):
    modal run ltx25_modal.py --prompt "a red fox walking through snow at dawn, snow crunching underfoot"
    modal run ltx25_modal.py --prompts-file prompts.txt          # cheapest: one warm GPU for all
    modal run ltx25_modal.py --prompt "the woman turns and smiles" --image face.png --quality hd

Deploy the API (scales to zero, billed only while busy):
    modal deploy ltx25_modal.py

License: free only for entities with annual revenue under $10M (LTX-2 license).
"""

import base64
import io
import random
import time
from typing import Literal

import modal
from pydantic import BaseModel, Field, model_validator

APP_NAME = "ltx-25-video"
MODEL_ID = "Lightricks/LTX-2.5-Diffusers"
GPU = "RTX-PRO-6000"  # 96 GB: the whole bf16 pipeline (~73 GB) fits, so no CPU offload
WEIGHTS_DIR = "/models/ltx-2.5"
# Not in model_index.json but used here: the 2x latent upsampler (HD mode).
EXTRA_COMPONENTS = ["latent_upsampler"]

# ---------------------------------------------------------------------------
# Limits and presets (shared by the API, the GPU worker and the CLI)
# ---------------------------------------------------------------------------
AspectRatio = Literal["16:9", "9:16", "1:1", "4:3", "3:4"]
Quality = Literal["standard", "hd"]
# standard: one pass at ~0.5 MP (model card: 960x544).
# hd: two passes - half size, 2x latent upsample, 3 refine steps. Sides are multiples of 64.
SIZE_PRESETS = {
    "standard": {"16:9": (960, 544), "9:16": (544, 960), "1:1": (768, 768), "4:3": (832, 640), "3:4": (640, 832)},
    "hd": {"16:9": (1920, 1088), "9:16": (1088, 1920), "1:1": (1280, 1280), "4:3": (1664, 1280), "3:4": (1280, 1664)},
}
FRAME_RATES = (24, 25, 30)
MAX_SECONDS = {"standard": 20.0, "hd": 10.0}
MAX_VIDEOS_PER_REQUEST = 4
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_SIDE = 1920


class VideoRequest(BaseModel):
    """Everything the model can do, in one request. Only `prompt` is required."""

    prompt: str = Field(min_length=1, max_length=4000,
                        description="One detailed paragraph works best: shot, motion, lighting and sound.")
    image: str | None = Field(None, description="Start frame (base64 or data URL). Turns this into image-to-video.")
    quality: Quality = Field("standard", description="standard: one pass. hd: 2x size via two-stage upsampling, ~3x slower.")
    aspect_ratio: AspectRatio | None = Field(
        None, description="Preset shape. Default: 16:9, or the start frame's closest shape.")
    width: int | None = Field(None, ge=256, le=MAX_SIDE, multiple_of=32, description="Exact width; overrides aspect_ratio. Set with height.")
    height: int | None = Field(None, ge=256, le=MAX_SIDE, multiple_of=32, description="Exact height. Set with width.")
    duration: float | None = Field(5.0, ge=1.0, le=20.0,
                                   description="Seconds. null = the model picks a length that fits the prompt (standard only).")
    min_seconds: float = Field(2.0, ge=1.0, le=20.0, description="Lower bound for an auto duration.")
    max_seconds: float = Field(10.0, ge=1.0, le=20.0, description="Upper bound for an auto duration.")
    frame_rate: Literal[24, 25, 30] = 24
    num_videos: int = Field(1, ge=1, le=MAX_VIDEOS_PER_REQUEST, description="Variations; video k uses seed + k.")
    seed: int | None = Field(None, ge=0, le=2**32 - 1, description="Same seed + same settings = same video.")
    enhance_prompt: bool = Field(False, description="Rewrite the prompt with the Gemma 4 enhancer first (~5-10 s).")

    @model_validator(mode="after")
    def _check(self):
        if (self.width is None) != (self.height is None):
            raise ValueError("set both width and height, or neither")
        if self.width and self.aspect_ratio:
            raise ValueError("use either width/height or aspect_ratio, not both")
        if self.quality == "hd":
            if self.duration is None:
                raise ValueError("hd needs a fixed duration (auto duration is standard only)")
            if self.width and (self.width % 64 or self.height % 64):
                raise ValueError("hd sizes must be multiples of 64 (the first pass runs at half size)")
        if self.duration is not None and self.duration > MAX_SECONDS[self.quality]:
            raise ValueError(f"{self.quality} clips can be at most {MAX_SECONDS[self.quality]:.0f} s")
        if self.duration is None and self.min_seconds >= self.max_seconds:
            raise ValueError("min_seconds must be less than max_seconds")
        return self

    def num_frames(self) -> int | None:
        """8k+1 frames closest to the duration, or None for auto."""
        if self.duration is None:
            return None
        return round(self.duration * self.frame_rate / 8) * 8 + 1

    def target_size(self, image_aspect: float | None) -> tuple[int, int]:
        if self.width:
            return self.width, self.height
        presets = SIZE_PRESETS[self.quality]
        if self.aspect_ratio:
            return presets[self.aspect_ratio]
        if image_aspect:  # follow the start frame's shape
            return min(presets.values(), key=lambda wh: abs(wh[0] / wh[1] - image_aspect))
        return presets["16:9"]


# ---------------------------------------------------------------------------
# Container images
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch>=2.6",
        "torchvision",           # Gemma 4 processors import it
        "transformers==5.17.0",  # first line with Gemma4Unified (LTX-2.5 text encoder)
        "diffusers==0.40.0",     # first release with LTX-2.5 (duration_head, Gemma 4)
        "accelerate",
        "av",                    # diffusers.utils.encode_video writes MP4 + audio with PyAV
        "pillow",
        "pydantic>=2",
        "huggingface_hub[hf_xet]",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
# The API container doesn't need torch, so it gets a tiny image that starts fast.
api_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "fastapi[standard]", "pillow", "pydantic>=2"
)

with image.imports():
    import torch
    from diffusers import LTX2ImageToVideoPipeline, LTX2Pipeline
    from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.pipelines.ltx2.utils import (
        DEFAULT_NEGATIVE_PROMPT,
        DISTILLED_SIGMA_VALUES,
        LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT,
        LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT,
        STAGE_2_DISTILLED_SIGMA_VALUES,
    )
    from diffusers.utils import encode_video
    from transformers import Gemma4ForConditionalGeneration

app = modal.App(APP_NAME, image=image)
weights = modal.Volume.from_name("ltx-25-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
# Shared key-value store: the GPU writes progress here, the API reads it.
progress = modal.Dict.from_name("ltx-25-progress", create_if_missing=True)


# ---------------------------------------------------------------------------
# Step 1: download weights once into a Volume (CPU container, no GPU billing)
# ---------------------------------------------------------------------------
@app.function(volumes={"/models": weights}, secrets=[hf_secret], cpu=4, memory=8192, timeout=90 * 60)
def download_weights():
    """Download what we load (~92 GB of the 174 GB repo). Re-runs only fetch what's missing."""
    import glob
    import json

    from huggingface_hub import snapshot_download

    t0 = time.time()
    # Pass 1: configs, tokenizers and shard indexes (a few MB).
    snapshot_download(MODEL_ID, local_dir=WEIGHTS_DIR, allow_patterns=["*.json", "*.jinja", "*.txt", "*.model"])
    index = json.load(open(f"{WEIGHTS_DIR}/model_index.json"))
    components = [k for k, v in index.items() if not k.startswith("_") and isinstance(v, list) and v[0]]
    components += EXTRA_COMPONENTS

    # Pass 2: weights. Some folders hold two shard sets (4x10 GB and 8x5 GB) or a
    # single file next to shards; diffusers loads what the index names, so fetch only that.
    patterns = []
    for c in components:
        idx = glob.glob(f"{WEIGHTS_DIR}/{c}/*.safetensors.index.json")
        if idx:
            shards = set(json.load(open(idx[0]))["weight_map"].values())
            patterns += [f"{c}/{s}" for s in shards]
        else:
            patterns.append(f"{c}/*.safetensors")
    print("components:", components)
    snapshot_download(MODEL_ID, local_dir=WEIGHTS_DIR, allow_patterns=patterns)
    weights.commit()
    print(f"Downloaded {MODEL_ID} to {WEIGHTS_DIR} in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# Step 2: the GPU worker
# ---------------------------------------------------------------------------
@app.cls(
    gpu=GPU,
    volumes={"/models": weights},
    memory=32768,          # weights go straight to the GPU; RAM is billed on real use above this
    scaledown_window=60,   # idle GPU time was 79% of the Qwen bill; batch prompts instead
    max_containers=1,      # hard cap on burn rate (~$3.30/h with RAM)
    timeout=20 * 60,
)
class LTX25:
    @modal.enter()
    def load(self):
        t0 = time.time()
        self.t2v = LTX2Pipeline.from_pretrained(
            WEIGHTS_DIR, dtype=torch.bfloat16, device_map="cuda", prompt_enhancer=None,
        )
        self.t2v.vae.enable_tiling()  # keeps the decode of long/large clips inside VRAM
        self.t2v.set_progress_bar_config(disable=True)
        # Same modules, no second copy in VRAM. dtype is required: from_pipe
        # otherwise casts every shared module to float32 (doubles VRAM -> OOM).
        self.i2v = LTX2ImageToVideoPipeline.from_pipe(self.t2v, dtype=torch.bfloat16)
        self.i2v.set_progress_bar_config(disable=True)
        upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            f"{WEIGHTS_DIR}/latent_upsampler", dtype=torch.bfloat16).to("cuda")
        self.upsample = LTX2LatentUpsamplePipeline(vae=self.t2v.vae, latent_upsampler=upsampler)
        self.upsample.set_progress_bar_config(disable=True)
        # The 10 GB enhancer waits in CPU RAM and visits the GPU only when asked,
        # so VRAM stays free for HD clips.
        self.enhancer = Gemma4ForConditionalGeneration.from_pretrained(
            f"{WEIGHTS_DIR}/prompt_enhancer", dtype=torch.bfloat16)
        print(f"Model loaded in {time.time() - t0:.1f}s")

    @modal.method()
    def generate(self, req: dict, image_bytes: bytes | None = None) -> dict:
        """req: VideoRequest fields minus `image`; image_bytes: raw start frame."""
        from PIL import Image

        r = VideoRequest(**req)
        start = Image.open(io.BytesIO(image_bytes)).convert("RGB") if image_bytes else None
        width, height = r.target_size(start.width / start.height if start else None)
        num_frames = r.num_frames()
        base_seed = r.seed if r.seed is not None else random.randint(0, 2**32 - 1)
        job_id = modal.current_function_call_id()
        report = self._reporter(job_id, r.num_videos)

        prompt = r.prompt
        if r.enhance_prompt:
            report(1, "enhancing")
            prompt = self._enhance(prompt, start, base_seed)

        t_start = time.time()
        videos = []
        for k in range(r.num_videos):
            seed = (base_seed + k) % 2**32
            t0 = time.time()
            video, audio = self._run(r, prompt, start, width, height, num_frames, seed,
                                     lambda stage, step=None, steps=None: report(k + 1, stage, step, steps))
            report(k + 1, "encoding")
            frames = video[0].shape[0]
            path = f"/tmp/{seed}.mp4"
            encode_video(video[0], fps=r.frame_rate, output_path=path, audio=audio[0].float().cpu(),
                         audio_sample_rate=self.t2v.vocoder.config.output_sampling_rate)
            seconds = round(time.time() - t0, 1)
            # One line per clip, same shape as the Qwen app, so usage can be counted from logs.
            print(f"{width}x{height} {frames}f, seed {seed}, i2v={start is not None}, {r.quality}: {seconds}s")
            with open(path, "rb") as f:
                videos.append({
                    "seed": seed, "width": width, "height": height, "num_frames": frames,
                    "fps": r.frame_rate, "duration": round(frames / r.frame_rate, 2),
                    "has_audio": True, "seconds": seconds, "data": f.read(),
                })

        return {
            "videos": videos,
            "prompt_used": prompt,
            "enhanced": r.enhance_prompt,
            "image_to_video": start is not None,
            "seconds": round(time.time() - t_start, 1),
        }

    def _run(self, r: VideoRequest, prompt, start, width, height, num_frames, seed, report):
        pipe = self.i2v if start is not None else self.t2v
        common = dict(
            prompt=prompt, negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            frame_rate=float(r.frame_rate),
            # Distilled checkpoint: fixed sigmas, all guidance off (model card values).
            guidance_scale=1.0, audio_guidance_scale=1.0,
            stg_scale=0.0, audio_stg_scale=0.0,
            modality_scale=1.0, audio_modality_scale=1.0,
            generator=torch.Generator("cuda").manual_seed(seed),
            return_dict=False,
        )
        if start is not None:
            common["image"] = start
        if num_frames is None:  # auto duration: the duration head predicts the length
            common.update(min_seconds=r.min_seconds, max_seconds=r.max_seconds)
        else:
            common["num_frames"] = num_frames

        if r.quality == "standard":
            return pipe(width=width, height=height, sigmas=DISTILLED_SIGMA_VALUES, output_type="np",
                        callback_on_step_end=self._steps(report, "denoising", len(DISTILLED_SIGMA_VALUES)),
                        **common)

        # HD: half size, 2x latent upsample, then a short refine pass at full size.
        video_latent, audio_latent = pipe(
            width=width // 2, height=height // 2, sigmas=DISTILLED_SIGMA_VALUES, output_type="latent",
            callback_on_step_end=self._steps(report, "denoising", len(DISTILLED_SIGMA_VALUES)), **common)
        report("upscaling")
        upscaled = self.upsample(latents=video_latent, latents_normalized=False,
                                 output_type="latent", return_dict=False)[0]
        return pipe(
            width=width, height=height, latents=upscaled, audio_latents=audio_latent,
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES, noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            output_type="np",
            callback_on_step_end=self._steps(report, "refining", len(STAGE_2_DISTILLED_SIGMA_VALUES)), **common)

    def _enhance(self, prompt: str, start, seed: int) -> str:
        """Rewrite the prompt with the dedicated Gemma 4 enhancer, then park it back in RAM."""
        self.t2v.prompt_enhancer = self.enhancer
        self.enhancer.to("cuda")
        try:
            system = LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT if start is not None else LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT
            out = self.t2v.enhance_prompt(prompt, system, seed=seed, device="cuda", image=start)[0]
        finally:
            self.enhancer.to("cpu")
            self.t2v.prompt_enhancer = None
            torch.cuda.empty_cache()
        print(f"enhanced prompt: {out[:200]}")
        return out.strip() or prompt

    @staticmethod
    def _steps(report, stage: str, steps: int):
        def callback(pipe, step, timestep, callback_kwargs):
            report(stage, step + 1, steps)
            return callback_kwargs
        return callback

    @staticmethod
    def _reporter(job_id, total_videos):
        """Publish progress at most once a second (always on a stage change)."""
        last = {"t": 0.0, "stage": None}

        def report(index, stage, step=None, steps=None):
            now = time.time()
            if not job_id or (now - last["t"] < 1 and stage == last["stage"] and step != steps):
                return
            last.update(t=now, stage=stage)
            try:
                progress.put(job_id, {"video": index, "videos": total_videos,
                                      "stage": stage, "step": step, "steps": steps})
            except Exception:
                pass  # progress is nice-to-have; never fail a generation over it

        return report


# ---------------------------------------------------------------------------
# Step 3: the HTTP API (CPU). Validates, then hands work to the GPU.
# ---------------------------------------------------------------------------
def decode_start_image(encoded: str) -> bytes:
    from PIL import Image

    s = encoded.split(",", 1)[-1] if encoded.startswith("data:") else encoded
    try:
        raw = base64.b64decode("".join(s.split()), validate=True)
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
    except Exception:
        raise ValueError("image is not a valid base64-encoded image")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(f"image is over {MAX_UPLOAD_BYTES // 2**20} MB")
    return raw


def serialize_result(result: dict, job_id: str | None = None) -> dict:
    """Jobs get video URLs (fetched separately); sync calls get inline data URLs."""
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
@modal.concurrent(max_inputs=50)  # one small container serves many polls/requests
@modal.asgi_app(requires_proxy_auth=True)  # callers need Modal-Key / Modal-Secret
def api():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, Response

    web = FastAPI(
        title="LTX-2.5 Video API",
        version="1.0.0",
        description="Text-to-video and image-to-video with synced audio. "
                    "Use /v1/jobs for UIs (progress + no long-held requests), /v1/generate for scripts.",
    )
    gpu = LTX25()

    def start_image_or_400(req: VideoRequest) -> bytes | None:
        if not req.image:
            return None
        try:
            return decode_start_image(req.image)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

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
        """The finished result, None while running; raises the job's error if it failed."""
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
        """Everything a UI needs to build its form: presets, limits, full request schema."""
        return {
            "model": MODEL_ID,
            "size_presets": {q: {ar: {"width": w, "height": h} for ar, (w, h) in shapes.items()}
                             for q, shapes in SIZE_PRESETS.items()},
            "frame_rates": list(FRAME_RATES),
            "limits": {
                "max_seconds": MAX_SECONDS, "max_side": MAX_SIDE, "side_multiple": 32,
                "max_videos_per_request": MAX_VIDEOS_PER_REQUEST,
                "max_upload_mb": MAX_UPLOAD_BYTES // 2**20,
            },
            "features": {"audio": True, "image_to_video": True, "auto_duration": True,
                         "hd": True, "enhance_prompt": True},
            "request_schema": VideoRequest.model_json_schema(),
        }

    @web.post("/v1/jobs", status_code=202)
    async def create_job(req: VideoRequest):
        """Start a generation and return right away. Poll GET /v1/jobs/{id}."""
        start = start_image_or_400(req)
        call = await gpu.generate.spawn.aio(req.model_dump(exclude={"image"}), start)
        return {"id": call.object_id, "status": "starting", "status_url": f"/v1/jobs/{call.object_id}"}

    @web.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str):
        """202 while starting/running (with progress), 200 when done or failed."""
        try:
            result = await job_result(job_id)
        except HTTPException:
            raise
        except Exception as e:
            return {"id": job_id, "status": "failed", "error": f"{type(e).__name__}: {e}"}
        if result is None:
            p = await progress.get.aio(job_id)
            # No progress yet = waiting for a GPU / loading the model (cold start ~1 min).
            return JSONResponse(status_code=202, content={
                "id": job_id, "status": "running" if p else "starting", "progress": p,
            })
        return {"id": job_id, "status": "done", **serialize_result(result, job_id)}

    @web.get("/v1/jobs/{job_id}/videos/{index}")
    async def get_job_video(job_id: str, index: int, request: Request):
        """The MP4 file, usable directly as <video src>. Supports Range for seeking."""
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
        return mp4_response(v["data"], f"ltx25_{v['seed']}.mp4", request.headers.get("range"))

    @web.post("/v1/generate")
    async def generate_sync(req: VideoRequest):
        """Wait for the videos and return them inline as data URLs. Simple, for scripts."""
        start = start_image_or_400(req)
        result = await gpu.generate.remote.aio(req.model_dump(exclude={"image"}), start)
        return serialize_result(result)

    def mp4_response(data: bytes, filename: str, range_header: str | None):
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, max-age=86400",  # a job's videos never change
            "Content-Disposition": f'inline; filename="{filename}"',
        }
        size = len(data)
        if range_header and range_header.startswith("bytes="):
            first, _, last = range_header[6:].split(",")[0].partition("-")
            start = int(first) if first else max(size - int(last), 0)
            end = int(last) if first and last else size - 1
            end = min(end, size - 1)
            if start > end:
                return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            return Response(data[start:end + 1], status_code=206, media_type="video/mp4", headers=headers)
        return Response(data, media_type="video/mp4", headers=headers)

    return web


# ---------------------------------------------------------------------------
# Step 4: local CLI - runs on your laptop, sends work straight to the GPU
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    prompt: str = "A cinematic shot of a red fox walking through a snowy forest at dawn, golden light "
                  "filtering through pine trees, the camera tracking alongside, snow crunching underfoot.",
    prompts_file: str = "",
    image: str = "",          # optional start frame for image-to-video
    quality: str = "standard",
    aspect_ratio: str = "",
    duration: float = 5.0,    # 0 = auto
    frame_rate: int = 24,
    n: int = 1,
    seed: int = -1,
    enhance: bool = False,
    out_dir: str = "ltx_outputs",
):
    from pathlib import Path

    prompts = ([p.strip() for p in Path(prompts_file).read_text().splitlines() if p.strip()]
               if prompts_file else [prompt])
    image_bytes = Path(image).read_bytes() if image else None
    # Validate locally so a typo fails now, not after a GPU cold start.
    reqs = [VideoRequest(prompt=p, quality=quality, aspect_ratio=aspect_ratio or None,
                         duration=duration or None, frame_rate=frame_rate, num_videos=n,
                         seed=seed if seed >= 0 else None, enhance_prompt=enhance).model_dump(exclude={"image"})
            for p in prompts]

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    stamp = int(time.time())
    t0 = time.time()
    count = 0
    for i, res in enumerate(LTX25().generate.map(reqs, [image_bytes] * len(reqs))):
        if res["enhanced"]:
            print(f"enhanced prompt: {res['prompt_used']}")
        for k, v in enumerate(res["videos"]):
            path = out / f"{stamp}_{i:03d}_{k}_seed{v['seed']}.mp4"
            path.write_bytes(v["data"])
            count += 1
            print(f"saved {path} ({v['width']}x{v['height']}, {v['duration']}s clip, {v['seconds']}s)")
    print(f"{count} clip(s) in {time.time() - t0:.0f}s (includes cold start)")
