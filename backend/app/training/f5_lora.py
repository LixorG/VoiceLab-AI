"""Fine-tune F5-TTS (or the E2-TTS reproduction) on one voice, with LoRA, in a process of its own.

    python -m app.training.f5_lora spec.json

Same protocol as the Qwen trainer: one JSON event per line on stdout (stage / epoch / progress / done / error), so
the queue can show progress and kill the process when the user cancels.

What the official recipe does (`f5_tts/train/finetune_cli.py` + `model/trainer.py`, f5-tts 1.1.x) and what is kept
here: the same model configuration per architecture, the same flow-matching loss (`CFM.forward`, which masks a
random span and predicts the flow there), the pretrained checkpoint loaded exactly as inference loads it, and —
easy to miss — **the text converted with `convert_char_to_pinyin`**, because that is what the prepared datasets
store (`prepare_csv_wavs.py`, `prepare_emilia.py`) and what inference feeds the model. Training on raw characters
would teach the model a different alphabet from the one it reads at inference.

What is deliberately different: the official trainer fine-tunes all 336M parameters with accelerate, an EMA copy and
frame-based batches. With a handful of minutes of audio that overfits quickly and does not fit comfortably in a
16 GB laptop card together with everything else, so this trains LoRA adapters on the attention and feed-forward
projections of the DiT, freezes the rest, and merges them into the weights when saving. The result is an ordinary
checkpoint file: the app loads it like any other custom checkpoint, and it still needs a reference to clone.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from app.training.qwen_lora import ClipSpec, emit, load_clip, lora_layer, merge_lora

SAMPLE_RATE = 24_000
HOP_LENGTH = 256
MEL_KWARGS = dict(n_fft=1024, hop_length=HOP_LENGTH, win_length=1024, n_mel_channels=100,
                  target_sample_rate=SAMPLE_RATE, mel_spec_type="vocos")
# Straight from f5_tts/train/finetune_cli.py: the architecture must match the pretrained checkpoint.
MODEL_CFG: dict[str, dict[str, Any]] = {
    "F5TTS_v1_Base": dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    "F5TTS_Base": dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=False,
                       conv_layers=4, pe_attn_head=1),
    "E2TTS_Base": dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1),
}
MAX_CLIP_S = 15.0


class TrainSpec(BaseModel):
    arch: str = "F5TTS_v1_Base"
    base_ckpt: str
    vocab_path: str
    output_path: str  # the .safetensors file to write
    clips: list[ClipSpec]
    epochs: int = Field(default=10, ge=1, le=100)
    learning_rate: float = Field(default=1e-4, gt=0)
    batch_size: int = Field(default=2, ge=1, le=16)
    grad_accumulation: int = Field(default=4, ge=1, le=64)
    lora_rank: int = Field(default=16, ge=1, le=128)
    lora_alpha: float = Field(default=16.0, gt=0)
    merge_scale: float = Field(default=1.0, gt=0, le=2.0)
    seed: int = 1234


# ---------------------------------------------------------------------------------------------------- model
def build_model(spec: TrainSpec, device: str) -> Any:
    """CFM with the pretrained weights, exactly as the library builds it for inference."""
    import torch
    from f5_tts.infer.utils_infer import load_checkpoint
    from f5_tts.model import CFM, DiT, UNetT
    from f5_tts.model.utils import get_tokenizer

    if spec.arch not in MODEL_CFG:
        raise ValueError(f"Arquitectura desconocida: {spec.arch}")
    vocab_char_map, vocab_size = get_tokenizer(spec.vocab_path, "custom")
    backbone = UNetT if spec.arch.startswith("E2") else DiT
    model = CFM(
        transformer=backbone(**MODEL_CFG[spec.arch], text_num_embeds=vocab_size, mel_dim=MEL_KWARGS["n_mel_channels"]),
        mel_spec_kwargs=MEL_KWARGS,
        vocab_char_map=vocab_char_map,
    )
    return load_checkpoint(model, spec.base_ckpt, device, dtype=torch.float32, use_ema=True)


def add_lora(model: Any, rank: int, alpha: float) -> list[tuple[Any, str, Any]]:
    """Wrap the attention and feed-forward projections of every transformer block; returns (parent, name, wrapper)."""
    from torch import nn

    blocks = getattr(model.transformer, "transformer_blocks", None)
    if blocks is None:  # UNetT keeps them under `layers`
        blocks = model.transformer.layers
    wrapped = []
    for block in blocks:
        for path in ("attn", "ff"):
            part = getattr(block, path, None)
            if part is None:
                continue
            for name, module in list(part.named_modules()):
                if not isinstance(module, nn.Linear):
                    continue
                parent = part if "." not in name else part.get_submodule(name.rsplit(".", 1)[0])
                attribute = name.rsplit(".", 1)[-1]
                wrapper = lora_layer(module, rank, alpha)
                setattr(parent, attribute, wrapper)
                wrapped.append((parent, attribute, wrapper))
    return wrapped


# ---------------------------------------------------------------------------------------------------- data
def prepare(spec: TrainSpec, model: Any, device: str) -> list[tuple[Any, list[str], int]]:
    """Each clip -> (mel [n, 100] on the GPU, text as the model reads it, frames)."""
    import torch
    from f5_tts.model.utils import convert_char_to_pinyin

    items = []
    for clip in spec.clips:
        audio = load_clip(clip)
        if not (0.3 <= audio.size / SAMPLE_RATE <= MAX_CLIP_S):
            continue
        wave = torch.from_numpy(audio).unsqueeze(0).to(device)
        with torch.no_grad():
            mel = model.mel_spec(wave).squeeze(0).transpose(0, 1).contiguous()  # [n, 100]
        text = convert_char_to_pinyin([clip.text])[0]
        items.append((mel, text, mel.shape[0]))
    if not items:
        raise ValueError("No hay frases utilizables para entrenar.")
    return items


def batches(items: list[tuple[Any, list[str], int]], size: int, rng: Any) -> list[list[int]]:
    """Group clips of similar length (less padding) and shuffle the groups so every epoch differs."""
    order = sorted(range(len(items)), key=lambda i: items[i][2])
    groups = [order[i:i + size] for i in range(0, len(order), size)]
    rng.shuffle(groups)
    return groups


def collate(items: list[tuple[Any, list[str], int]], index: list[int], device: str) -> tuple[Any, list, Any]:
    import torch

    mels = [items[i][0] for i in index]
    lens = torch.tensor([m.shape[0] for m in mels], device=device)
    width = int(lens.max().item())
    padded = torch.stack([torch.nn.functional.pad(m, (0, 0, 0, width - m.shape[0])) for m in mels])
    return padded, [items[i][1] for i in index], lens


# ---------------------------------------------------------------------------------------------------- training
def save_model(spec: TrainSpec, model: Any, wrapped: list[tuple[Any, str, Any]]) -> Path:
    """Merge the adapters and write one .safetensors the app can load like any other checkpoint."""
    import torch
    from safetensors.torch import save_file

    merge_lora(wrapped, spec.merge_scale)
    state = {k: v.detach().to(torch.float32).contiguous().cpu() for k, v in model.state_dict().items()}
    out = Path(spec.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial")
    save_file(state, str(partial))
    partial.replace(out)  # atomic: a half-written checkpoint is never published
    return out


def train(spec: TrainSpec) -> tuple[Path, list[dict[str, Any]]]:
    import random

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(spec.seed)
    rng = random.Random(spec.seed)

    emit("stage", stage="load", message="Cargando el modelo base…")
    model = build_model(spec, device)
    model.requires_grad_(False)
    wrapped = add_lora(model, spec.lora_rank, spec.lora_alpha)
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()
    model.train()

    items = prepare(spec, model, device)
    minutes = sum(i[2] for i in items) * HOP_LENGTH / SAMPLE_RATE / 60
    emit("stage", stage="train", clips=len(items),
         message=f"Entrenando con {len(items)} frases ({minutes:.1f} min)")

    optimizer = torch.optim.AdamW(trainable, lr=spec.learning_rate, weight_decay=0.01)
    steps_per_epoch = max(1, math.ceil(len(items) / spec.batch_size / spec.grad_accumulation))
    total_steps = steps_per_epoch * spec.epochs
    warmup = max(1, int(total_steps * 0.05))

    def lr_at(step: int) -> float:
        if step < warmup:
            return spec.learning_rate * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return spec.learning_rate * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    history: list[dict[str, Any]] = []
    step = 0
    started = time.monotonic()
    for epoch in range(1, spec.epochs + 1):
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        groups = batches(items, spec.batch_size, rng)
        for position, group in enumerate(groups, start=1):
            mel, text, lens = collate(items, group, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss, _, _ = model(mel, text=text, lens=lens)
            (loss / spec.grad_accumulation).backward()
            losses.append(float(loss.detach()))
            if position % spec.grad_accumulation == 0 or position == len(groups):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                for group_param in optimizer.param_groups:
                    group_param["lr"] = lr_at(step)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                elapsed = time.monotonic() - started
                emit("progress", epoch=epoch, epochs=spec.epochs, step=step, total_steps=total_steps,
                     loss=round(float(np.mean(losses[-10:])), 4),
                     eta_s=round(elapsed / max(1, step) * (total_steps - step), 1))
        entry = {"epoch": epoch, "loss": round(float(np.mean(losses)), 4)}
        history.append(entry)
        emit("epoch", **entry)

    emit("stage", stage="save", message="Guardando la voz entrenada…",
         peak_vram_mb=round(torch.cuda.max_memory_allocated() / 1024 ** 2) if device == "cuda" else None)
    model.eval()
    return save_model(spec, model, wrapped), history


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        emit("error", detail="uso: python -m app.training.f5_lora spec.json")
        return 2
    try:
        spec = TrainSpec.model_validate_json(Path(argv[1]).read_text(encoding="utf-8"))
        output, history = train(spec)
    except Exception as exc:  # noqa: BLE001  (the parent process turns this into a message in Spanish)
        out_of_memory = "out of memory" in str(exc).lower()
        emit("error", code="OUT_OF_MEMORY" if out_of_memory else "TRAINING_FAILED", detail=str(exc)[:600])
        return 1
    emit("done", output=str(output), history=history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
