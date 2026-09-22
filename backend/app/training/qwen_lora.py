"""Fine-tune Qwen3-TTS Base on a single voice with LoRA; runs as its own process (``python -m app.training.qwen_lora
spec.json``) and reports progress as one JSON object per stdout line.

The recipe is the official one (QwenLM/Qwen3-TTS ``finetuning/sft_12hz.py`` + ``dataset.py``): the training
sequence is [chat prefix | codec tags + speaker slot | text | codec frames], the speaker slot holds the speaker
encoder's embedding of one fixed reference clip, the loss is the talker's first-codebook cross-entropy plus 0.3 × the
code predictor's cross-entropy for codebooks 1–15, and the result is saved as a ``custom_voice`` model whose single
speaker (id 3000) is that embedding. Three alignment bugs of the official script, measured by the community
(QwenLM/Qwen3-TTS issue #371, PR #278), are fixed here:

- text embeddings go through ``talker.text_projection`` like in generation (the 0.6B model crashes otherwise and the
  1.7B one silently trains on unprojected text);
- labels are not shifted by hand: the causal loss already shifts them (a double shift trains the model to predict two
  frames ahead: speech gets faster every epoch);
- the code predictor gets the talker hidden state of the *previous* position (what it receives when generating) and
  its labels are not shifted either.

LoRA instead of the official full fine-tune: a full fine-tune of the 1.7B model needs weights + gradients + optimizer
states (≈ 14 GB before activations), too much for a 16 GB laptop GPU; with LoRA the base stays frozen in bf16 and only
small fp32 adapters on the attention and MLP projections are trained, then merged into the saved weights.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

SAMPLE_RATE = 24_000
SPEAKER_ID = 3000  # the token id the official recipe uses for the new speaker
SUB_TALKER_WEIGHT = 0.3
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class ClipSpec(BaseModel):
    audio_path: Path
    text: str
    start_s: float | None = None
    end_s: float | None = None


class TrainSpec(BaseModel):
    base_model_path: Path  # a Qwen3-TTS Base snapshot (config.json, model.safetensors, speech_tokenizer/)
    output_dir: Path
    speaker_name: str
    clips: list[ClipSpec]
    reference: ClipSpec  # one fixed clip for the speaker embedding (the official recipe recommends a single one)
    epochs: int = Field(default=10, ge=1, le=100)
    learning_rate: float = Field(default=1e-4, gt=0, le=1e-2)
    batch_size: int = Field(default=2, ge=1, le=32)
    grad_accumulation: int = Field(default=4, ge=1, le=64)
    lora_rank: int = Field(default=16, ge=1, le=256)
    lora_alpha: float = Field(default=16.0, gt=0)
    merge_scale: float = Field(default=1.0, gt=0, le=2.0)
    seed: int = 1234


def emit(event: str, **data: Any) -> None:
    print(json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------------------------------------------- audio / data
def load_clip(clip: ClipSpec) -> np.ndarray:
    """Mono float32 at 24 kHz, cut to [start_s, end_s]."""
    from math import gcd

    import soundfile as sf
    from scipy.signal import resample_poly

    info = sf.info(str(clip.audio_path))
    start = int((clip.start_s or 0.0) * info.samplerate)
    stop = int(clip.end_s * info.samplerate) if clip.end_s is not None else None
    audio, sr = sf.read(str(clip.audio_path), start=start, stop=stop, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def encode_codes(tokenizer: Any, audios: list[np.ndarray], batch: int = 8) -> list[Any]:
    codes = []
    for i in range(0, len(audios), batch):
        out = tokenizer.encode(audios[i:i + batch], sr=SAMPLE_RATE)
        codes += [c.cpu() for c in out.audio_codes]
    return codes


def text_ids(processor: Any, text: str) -> Any:
    """Token ids of the assistant turn, trimmed exactly like the official dataset (the last 5 tokens are the next
    turn's header, which generation does not feed as text)."""
    ids = processor(text=f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n",
                    return_tensors="pt", padding=True)["input_ids"]
    ids = ids.unsqueeze(0) if ids.dim() == 1 else ids
    return ids[:, :-5]


def collate(items: list[tuple[Any, Any]], config: Any) -> dict[str, Any]:
    """Port of the official ``TTSDataset.collate_fn`` (language «Auto»: no language tag in the codec prefix)."""
    import torch

    tc = config.talker_config
    lengths = [ids.shape[1] + codes.shape[0] for ids, codes in items]
    b, t = len(items), max(lengths) + 8
    input_ids = torch.zeros((b, t, 2), dtype=torch.long)
    codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
    text_mask = torch.zeros((b, t), dtype=torch.bool)
    codec_emb_mask = torch.zeros((b, t), dtype=torch.bool)
    codec_mask = torch.zeros((b, t), dtype=torch.bool)
    attention = torch.zeros((b, t), dtype=torch.long)
    labels = torch.full((b, t), -100, dtype=torch.long)
    for i, (ids, codes) in enumerate(items):
        n_text, n_codec = ids.shape[1], codes.shape[0]
        first = 8 + n_text - 1  # position of the first codec frame
        input_ids[i, :3, 0] = ids[0, :3]
        input_ids[i, 3:7, 0] = config.tts_pad_token_id
        input_ids[i, 7, 0] = config.tts_bos_token_id
        input_ids[i, 8:8 + n_text - 3, 0] = ids[0, 3:]
        input_ids[i, 8 + n_text - 3, 0] = config.tts_eos_token_id
        input_ids[i, 8 + n_text - 2:first + n_codec + 1, 0] = config.tts_pad_token_id
        text_mask[i, :first + n_codec + 1] = True
        input_ids[i, 3:8, 1] = torch.tensor([tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id, 0,
                                             tc.codec_pad_id])
        input_ids[i, 8:first - 1, 1] = tc.codec_pad_id
        input_ids[i, first - 1, 1] = tc.codec_bos_id
        input_ids[i, first:first + n_codec, 1] = codes[:, 0]
        input_ids[i, first + n_codec, 1] = tc.codec_eos_token_id
        labels[i, first:first + n_codec] = codes[:, 0]
        labels[i, first + n_codec] = tc.codec_eos_token_id
        codec_ids[i, first:first + n_codec] = codes
        codec_emb_mask[i, 3:first + n_codec + 1] = True
        codec_emb_mask[i, 6] = False  # the speaker slot
        codec_mask[i, first:first + n_codec] = True
        attention[i, :first + n_codec + 1] = 1
    return {"input_ids": input_ids, "codec_ids": codec_ids, "text_mask": text_mask.unsqueeze(-1),
            "codec_emb_mask": codec_emb_mask.unsqueeze(-1), "codec_mask": codec_mask, "attention": attention,
            "labels": labels}


# ---------------------------------------------------------------------------------------------------- LoRA
def lora_layer(base: Any, rank: int, alpha: float) -> Any:
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base
            self.scale = alpha / rank
            self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        def forward(self, x):  # noqa: ANN001, ANN202
            delta = (x.float() @ self.lora_a.t()) @ self.lora_b.t()
            return self.base(x) + (delta * self.scale).to(x.dtype)

        def merged_weight(self, scale: float):  # noqa: ANN202
            w = self.base.weight.float() + scale * self.scale * (self.lora_b @ self.lora_a)
            return w.to(self.base.weight.dtype)

    return LoRALinear()


def add_lora(talker: Any, rank: int, alpha: float) -> list[tuple[Any, str, Any]]:
    """Wrap the attention/MLP projections of the talker and of the code predictor; returns (parent, name, wrapper)."""
    wrapped = []
    for layers in (talker.model.layers, talker.code_predictor.model.layers):
        for layer in layers:
            for parent in (layer.self_attn, layer.mlp):
                for name in LORA_TARGETS:
                    base = getattr(parent, name, None)
                    if base is None:
                        continue
                    wrapper = lora_layer(base, rank, alpha)
                    setattr(parent, name, wrapper)
                    wrapped.append((parent, name, wrapper))
    return wrapped


def merge_lora(wrapped: list[tuple[Any, str, Any]], scale: float) -> None:
    import torch

    with torch.no_grad():
        for parent, name, wrapper in wrapped:
            wrapper.base.weight.copy_(wrapper.merged_weight(scale))
            setattr(parent, name, wrapper.base)


# ---------------------------------------------------------------------------------------------------- training
def sub_talker_loss(talker: Any, codes: Any, hidden: Any) -> Any:
    """Code predictor loss for codebooks 1–15 given each frame's talker hidden state; labels already aligned."""
    import torch
    import torch.nn.functional as F  # noqa: N812

    embeds = [hidden.unsqueeze(1), talker.get_input_embeddings()(codes[:, :1])]
    embeds += [talker.code_predictor.get_input_embeddings()[i - 1](codes[:, i:i + 1]) for i in range(1, 15)]
    logits = talker.code_predictor.forward_finetune(inputs_embeds=torch.cat(embeds, dim=1)).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), codes[:, 1:].reshape(-1))


def step_loss(model: Any, batch: dict[str, Any], speaker: Any) -> Any:
    talker = model.talker
    ids = batch["input_ids"]
    text = talker.text_projection(talker.model.text_embedding(ids[:, :, 0])) * batch["text_mask"]
    codec = talker.model.codec_embedding(ids[:, :, 1]) * batch["codec_emb_mask"]
    codec[:, 6, :] = speaker
    embeds = text + codec
    for i in range(1, 16):
        extra = talker.code_predictor.get_input_embeddings()[i - 1](batch["codec_ids"][:, :, i])
        embeds = embeds + extra * batch["codec_mask"].unsqueeze(-1)
    out = talker(inputs_embeds=embeds, attention_mask=batch["attention"], labels=batch["labels"],
                 output_hidden_states=True)
    hidden = out.hidden_states[0][-1][:, :-1]
    frame_hidden = hidden[batch["codec_mask"][:, 1:]]  # the position *before* each frame, as in generation
    frame_codes = batch["codec_ids"][batch["codec_mask"]]
    return out.loss, sub_talker_loss(talker, frame_codes, frame_hidden)


def save_model(spec: TrainSpec, model: Any, config_dict: dict, speaker: Any) -> Path:
    """Write a complete custom_voice model directory (loadable with ``Qwen3TTSModel.from_pretrained``)."""
    from safetensors.torch import save_file

    out = spec.output_dir
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    base = Path(spec.base_model_path)

    def skip(directory: str, names: list[str]) -> list[str]:  # only the base weights; speech_tokenizer/ keeps its own
        top = Path(directory) == base
        return [n for n in names if n.startswith(".") or (top and (n == "model.safetensors" or n.endswith(".md")))]

    shutil.copytree(base, tmp, ignore=skip)
    state = {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()
             if not k.startswith("speaker_encoder")}
    emb = state["talker.model.codec_embedding.weight"]
    emb[SPEAKER_ID] = speaker.detach().to("cpu").to(emb.dtype)
    save_file(state, str(tmp / "model.safetensors"), metadata={"format": "pt"})
    cfg = dict(config_dict)
    cfg["tts_model_type"] = "custom_voice"
    talker_cfg = dict(cfg.get("talker_config", {}))
    talker_cfg["spk_id"] = {spec.speaker_name: SPEAKER_ID}
    talker_cfg["spk_is_dialect"] = {spec.speaker_name: False}
    cfg["talker_config"] = talker_cfg
    (tmp / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)  # atomic from the app's point of view: a half-written model never looks finished
    return out


def train(spec: TrainSpec) -> Path:
    import torch
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

    torch.manual_seed(spec.seed)
    rng = np.random.default_rng(spec.seed)
    started = time.perf_counter()
    emit("stage", stage="codes", message="Preparando el audio de entrenamiento…")
    audios = [load_clip(c) for c in spec.clips]
    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(spec.base_model_path / "speech_tokenizer"), device_map="cuda:0")
    codes = encode_codes(tokenizer, audios)
    del tokenizer
    torch.cuda.empty_cache()

    emit("stage", stage="load", message="Cargando el modelo base…")
    wrapper = Qwen3TTSModel.from_pretrained(str(spec.base_model_path), device_map="cuda:0", dtype=torch.bfloat16,
                                            attn_implementation="sdpa")
    model, processor = wrapper.model, wrapper.processor
    config = model.config
    config_dict = json.loads((spec.base_model_path / "config.json").read_text(encoding="utf-8"))
    items = [(text_ids(processor, c.text), code) for c, code in zip(spec.clips, codes, strict=True)]

    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    with torch.no_grad():
        ref = torch.from_numpy(load_clip(spec.reference)).unsqueeze(0)
        mel = mel_spectrogram(ref, n_fft=1024, num_mels=128, sampling_rate=SAMPLE_RATE, hop_size=256, win_size=1024,
                              fmin=0, fmax=12000).transpose(1, 2)
        speaker = model.speaker_encoder(mel.to(model.device).to(model.dtype))[0].detach()

    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = add_lora(model.talker, spec.lora_rank, spec.lora_alpha)
    params = [p for _, _, w in wrapped for p in (w.lora_a, w.lora_b)]
    optimizer = torch.optim.AdamW(params, lr=spec.learning_rate, weight_decay=0.0)
    batches_per_epoch = math.ceil(len(items) / spec.batch_size)
    total_updates = math.ceil(batches_per_epoch / spec.grad_accumulation) * spec.epochs
    warmup = max(1, total_updates // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, s / max(1, total_updates)))))
    emit("stage", stage="train", message="Entrenando…", clips=len(items), epochs=spec.epochs,
         total_steps=batches_per_epoch * spec.epochs, trainable_params=sum(p.numel() for p in params))

    model.train()
    step, history = 0, []
    for epoch in range(spec.epochs):
        order = rng.permutation(len(items))
        losses = []
        for b in range(batches_per_epoch):
            batch = collate([items[i] for i in order[b * spec.batch_size:(b + 1) * spec.batch_size]], config)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            talker_loss, predictor_loss = step_loss(model, batch, speaker)
            loss = talker_loss + SUB_TALKER_WEIGHT * predictor_loss
            (loss / spec.grad_accumulation).backward()
            last = b == batches_per_epoch - 1
            if (b + 1) % spec.grad_accumulation == 0 or last:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                schedule.step()
            step += 1
            losses.append((float(talker_loss), float(predictor_loss)))
            if step % 10 == 0 or last:
                elapsed = time.perf_counter() - started
                done = step / (batches_per_epoch * spec.epochs)
                emit("progress", epoch=epoch + 1, epochs=spec.epochs, step=step,
                     total_steps=batches_per_epoch * spec.epochs, talker_loss=round(float(talker_loss), 4),
                     predictor_loss=round(float(predictor_loss), 4),
                     eta_s=round(elapsed / done * (1 - done)) if done > 0 else None,
                     vram_mb=round(torch.cuda.max_memory_allocated() / 2**20))
        mean = np.mean(losses, axis=0)
        history.append({"epoch": epoch + 1, "talker_loss": round(float(mean[0]), 4),
                        "predictor_loss": round(float(mean[1]), 4)})
        emit("epoch", **history[-1])

    emit("stage", stage="save", message="Guardando el modelo entrenado…")
    model.eval()
    merge_lora(wrapped, spec.merge_scale)
    out = save_model(spec, model, config_dict, speaker)
    (out / "voicelab_training.json").write_text(json.dumps({
        "base_model": spec.base_model_path.name, "speaker_name": spec.speaker_name, "clips": len(items),
        "minutes": round(sum(a.size for a in audios) / SAMPLE_RATE / 60, 2), "epochs": spec.epochs,
        "learning_rate": spec.learning_rate, "batch_size": spec.batch_size, "grad_accumulation": spec.grad_accumulation,
        "lora_rank": spec.lora_rank, "lora_alpha": spec.lora_alpha, "merge_scale": spec.merge_scale,
        "history": history, "seconds": round(time.perf_counter() - started, 1)}, indent=2), encoding="utf-8")
    emit("done", output_dir=str(out), history=history, seconds=round(time.perf_counter() - started, 1))
    return out


def main(argv: list[str]) -> int:
    spec = TrainSpec.model_validate_json(Path(argv[1]).read_text(encoding="utf-8"))
    try:
        train(spec)
    except Exception as exc:  # noqa: BLE001  (reported to the parent process, which shows a Spanish message)
        import torch

        oom = isinstance(exc, torch.cuda.OutOfMemoryError)
        emit("error", code="OUT_OF_MEMORY" if oom else "TRAINING_FAILED", detail=f"{type(exc).__name__}: {exc}"[:500])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
