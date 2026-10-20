"""Run an open-weight Qwen3-VL model over clips (the Kaggle job calls this).

Kept apart from gold.vlm so prompts and parsing stay importable without torch.
Frames are read with OpenCV and handed to the processor as arrays, so no video
decoding backend is needed on Kaggle.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cornerman.gold.vlm import PROMPTS, parse_direct, parse_questions, prompt

PARSERS = {"direct": parse_direct, "questions": parse_questions}


@dataclass(frozen=True)
class VlmSpec:
    name: str
    revision: str
    sample_fps: float = 4.0
    """Frames per second given to the model (clips are 10 fps)."""
    max_new_tokens: int = 160
    prompts: tuple[str, ...] = tuple(PROMPTS)
    """Which prompts to ask (a rerun can ask only the one that changed)."""
    max_memory: dict[int, str] | None = None
    """Per-GPU cap for weights when spread over GPUs, e.g. {0: "7GiB", 1: "14GiB"}:
    GPU 0 also holds the vision encoder's activations for a whole clip."""


def read_clip(path: Path, sample_fps: float) -> tuple[np.ndarray, float]:
    """RGB frames (T, H, W, 3) sampled to about `sample_fps`, and the clip's fps.

    Qwen3-VL merges frames in pairs, so an odd count is padded with the last frame.
    """
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    step = max(1, round(fps / sample_fps))
    kept = frames[::step]
    if len(kept) % 2:
        kept.append(kept[-1])
    return np.stack(kept), fps / step


def load(spec: VlmSpec, device_map: str | dict[str, int] = "auto") -> tuple[Any, Any]:
    """Model and processor in fp16 (the T4 has no bf16).

    `device_map` "auto" (or a dict) lets accelerate place the model across GPUs; a
    single device name ("cuda:0", "mps", "cpu") loads on the CPU and moves it, since
    loading straight onto mps hung on the laptop.
    """
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    spread = device_map == "auto" or isinstance(device_map, dict)
    loader: Any = Qwen3VLForConditionalGeneration
    model: Any = loader.from_pretrained(
        spec.name,
        revision=spec.revision,
        dtype=torch.float16,
        device_map=device_map if spread else None,
        max_memory=spec.max_memory if spread else None,
    )
    if not spread:
        model = model.to(device_map)
    auto: Any = AutoProcessor
    processor: Any = auto.from_pretrained(spec.name, revision=spec.revision)
    return model.eval(), processor


def ask(model: Any, processor: Any, frames: np.ndarray, fps: float, text: str, max_new: int) -> str:
    """One answer, greedy decoding."""
    import torch

    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    meta = {"fps": fps, "total_num_frames": len(frames), "frames_indices": list(range(len(frames)))}
    inputs = processor(
        text=[chat],
        videos=[frames],
        video_metadata=[meta],
        do_sample_frames=False,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    answer = out[0, inputs["input_ids"].shape[1] :]
    return str(processor.decode(answer, skip_special_tokens=True))


def label_clips(
    spec: VlmSpec, clips: list[Path], device_map: str | dict[str, int] = "auto"
) -> list[dict[str, object]]:
    """Both prompts on every clip: one row per (clip, prompt) with the raw answer,
    the parsed labels and the seconds taken."""
    model, processor = load(spec, device_map)
    rows: list[dict[str, object]] = []
    for path in clips:
        frames, fps = read_clip(path, spec.sample_fps)
        for name in spec.prompts:
            t0 = time.time()
            try:
                text = ask(model, processor, frames, fps, prompt(name), spec.max_new_tokens)
                error = None
            except Exception as e:  # keep going; one bad clip shouldn't end the job
                text, error = "", repr(e)[:500]
            parsed = PARSERS[name](text) if text else None
            rows.append(
                {
                    "clip_id": path.stem,
                    "model": spec.name,
                    "prompt": name,
                    "answer": text,
                    "error": error,
                    "live": None if parsed is None else parsed["live"],
                    "red": None if parsed is None else parsed["red"],
                    "blue": None if parsed is None else parsed["blue"],
                    "seconds": round(time.time() - t0, 2),
                }
            )
    return rows
