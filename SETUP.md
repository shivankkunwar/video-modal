# Video models on Modal

Two apps, next to the Qwen image app. Each scales to zero, so an app that is
not in use costs nothing. Weights sit in Volumes (first 1 TiB/month is free).

| App | Model | GPU | Audio | License | Model load | 5 s clip (measured 2026-09-28) |
|---|---|---|---|---|---|---|
| `ltx-25-video` (`ltx25_modal.py`) | LTX-2.5 distilled, 22B | RTX-PRO-6000, 96 GB, $3.03/h | Yes | Free under $10M revenue | 33.8 s | 14.6 s at 960×544 |
| `fastwan-22-video` (`fastwan_modal.py`) | FastWan 2.2 TI2V-5B, 3 steps | L40S, $1.95/h | No | Apache 2.0 | 87 s | 41.2 s at 1280×704 |

Cost per clip (GPU + RAM, includes cold start and the 60 s idle window):

| | One clip per session | 10 clips per session | Fully warm |
|---|---|---|---|
| LTX-2.5, 960×544 | ~$0.12 | ~$0.024 | ~$0.013 |
| FastWan, 1280×704 | ~$0.12 | ~$0.034 | ~$0.024 |

## One-time setup
1. Accept the LTX-2.5 terms on https://huggingface.co/Lightricks/LTX-2.5-Diffusers
   with the Hugging Face account whose token is in the `huggingface` Modal secret.
2. Download the weights (CPU only, a few cents):
   ```bash
   modal run fastwan_modal.py::download_weights   # ~24 GB
   modal run ltx25_modal.py::download_weights     # ~92 GB of the 174 GB repo
   ```

## Generate
```bash
modal run ltx25_modal.py --prompt "a red fox walking through snow at dawn, snow crunching underfoot"
modal run ltx25_modal.py --prompt "the woman turns and smiles" --image face.png   # image-to-video
modal run fastwan_modal.py --prompt "a neon-lit Tokyo alley in heavy rain at night"
```
Files land in `ltx_outputs/` and `fastwan_outputs/`.

LTX options: `--quality hd`, `--aspect-ratio 9:16`, `--duration 8` (`0` = auto),
`--frame-rate 30`, `--n 2`, `--seed`, `--enhance`. FastWan: `--aspect-ratio`,
`--duration` (≤5), `--n`, `--seed`.

## API and GUI
```bash
modal deploy ltx25_modal.py      # https://shivankkunwar100--ltx-25-video-api.modal.run
modal deploy fastwan_modal.py    # https://shivankkunwar100--fastwan-22-video-api.modal.run
```
Same shape as the Qwen API: a small CPU `api` in front of the GPU, proxy-token
auth (the same `wk-`/`ws-` keys as Qwen), docs at `<url>/docs`.

| Endpoint | Use |
|---|---|
| `GET /health`, `GET /v1/options` | Presets, limits, features, request schema. CPU only. |
| `POST /v1/jobs` | Start a job, returns `{id}` at once. |
| `GET /v1/jobs/{id}` | 202 + `progress` (`stage`, `step`/`steps`) while running, 200 when done. |
| `GET /v1/jobs/{id}/videos/{i}` | The MP4 (supports `Range`). Results are kept 7 days. |
| `POST /v1/generate` | Waits, returns data URLs. For scripts. |

LTX request fields: `prompt`, `image` (start frame), `quality` (`standard`/`hd`),
`aspect_ratio` or `width`+`height`, `duration` (`null` = auto, with
`min_seconds`/`max_seconds`), `frame_rate` (24/25/30), `num_videos` (1–4),
`seed`, `enhance_prompt`. FastWan: `prompt`, `aspect_ratio` or size,
`duration` (≤5 s), `num_videos`, `seed`.

The GUI (`../gui`, page `/video`) calls these through `/api/video/<ltx|fastwan>/...`.

## Save credits
- Batch prompts: `--prompts-file prompts.txt` (one prompt per line). One
  cold start then serves every prompt.
- One clip at a time costs about 5x more than a batch. Most of the cost is
  the cold start and the 60 s idle window, not the generation.
- Check spend: `modal billing summary`. Count clips:
  `modal app logs ltx-25-video --since 1d --search "f, seed"`.

## Notes
- LTX-2.5 loads the whole bf16 pipeline onto the GPU (`device_map="cuda"`).
  If it runs out of GPU memory, change the load to `enable_model_cpu_offload()`.
- LTX prompts work best as one long paragraph: shot, motion, lighting, sound.
- FastWan is text-to-video only: in fastvideo 0.2.1 the 3-step DMD stage never
  applies the start-image mask, so a start frame would be ignored.
- Measured 2026-09-28: LTX HD image-to-video with prompt enhancement,
  1920×1088, 5 s: 43.6 s on a warm GPU.
- FastWan needs the `fastvideo` package (plain diffusers can't run its DMD
  checkpoint), so it has its own container image with torch 2.12.
