"""Live preview: chunk writer, Qwen code streamer windows, chunk counter in job events, chunk route (no GPU)."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
import torch

from app.engines.qwen3tts.streaming import CodeStreamer
from app.generation.streaming import StreamWriter, chunk_path, silence
from tests.test_advanced_generation import engines as adv_engines  # noqa: F401  (fixture re-export)
from tests.test_generation import engines as engines  # noqa: F401  (fixture re-export)
from tests.test_generation import wait_done


def test_stream_writer_numbers_chunks_and_notifies(tmp_path):
    counts = []
    writer = StreamWriter(tmp_path / "gen", counts.append)
    writer.write(np.ones(240, dtype=np.float32) * 0.5, 24_000)
    writer.write(np.zeros(0, dtype=np.float32), 24_000)  # empty: ignored
    writer.write(np.ones(480, dtype=np.float32) * 2.0, 24_000)  # clipped, never overflows
    assert counts == [1, 2]
    first, sr = sf.read(chunk_path(tmp_path / "gen", 0))
    second, _ = sf.read(chunk_path(tmp_path / "gen", 1))
    assert sr == 24_000 and first.size == 240 and second.size == 480 and second.max() <= 1.0
    assert not list((tmp_path / "gen").glob("*.tmp.wav"))  # atomic writes
    assert silence(250, 24_000).size == 6000 and silence(-5, 24_000).size == 0


def _frames(n, start=0):
    """Frame i carries the value i in every codebook, so a fake decoder can tell which frames it received."""
    return [torch.full((1, 16), start + i, dtype=torch.long) for i in range(n)]


def _streamer(prefix_len=0, **kw):
    calls, chunks = [], []

    def decode(codes):
        calls.append(codes[:, 0].tolist())
        return np.repeat(codes[:, 0].numpy().astype(np.float32), 4)  # 4 samples per frame, value = frame id

    prefix = torch.stack([f[0] for f in _frames(prefix_len, start=1000)]) if prefix_len else None
    streamer = CodeStreamer(decode, lambda audio, sr: chunks.append(audio), prefix=prefix, samples_per_frame=4, **kw)
    return streamer, calls, chunks


def test_code_streamer_schedule_and_exact_samples():
    streamer, calls, chunks = _streamer(schedule=(2, 2), every=3, max_context=None)
    for frame in _frames(9):
        streamer.add(frame)
    streamer.flush()
    # chunks of 2, 2, 3, then the remaining 2 on flush — and together exactly the 9 frames in order
    assert [c.size // 4 for c in chunks] == [2, 2, 3, 2]
    assert np.array_equal(np.concatenate(chunks), np.repeat(np.arange(9, dtype=np.float32), 4))
    streamer.flush()  # nothing left: no empty chunk
    assert len(chunks) == 4


def test_code_streamer_uses_the_reference_as_context_and_library_windows():
    streamer, calls, chunks = _streamer(prefix_len=5, schedule=(2,), every=2, window=4, window_context=1,
                                        max_context=None)
    for frame in _frames(6):
        streamer.add(frame)
    # absolute positions: 5 reference frames (1000…1004) then generated 0…5
    # chunk 1 = generated 0,1 = abs 5,6 → window starting at 4, with 1 frame of context → decode abs 3..6
    assert calls[0] == [1003, 1004, 0, 1]
    # chunk 2 = abs 7,8 → window [4, 8) for abs 7 → context from abs 3
    assert calls[1] == [1003, 1004, 0, 1, 2, 3]
    assert np.array_equal(np.concatenate(chunks), np.repeat(np.arange(6, dtype=np.float32), 4))


def test_code_streamer_caps_the_context():
    streamer, calls, _ = _streamer(prefix_len=50, schedule=(2,), every=2, max_context=3)
    for frame in _frames(4):
        streamer.add(frame)
    assert calls[0] == [1047, 1048, 1049, 0, 1]  # only 3 frames of left context
    assert calls[1] == [1049, 0, 1, 2, 3]


def test_streaming_engine_publishes_chunks(client, engines, settings):  # noqa: F811
    gen_id = client.post("/api/generation", json={"engine": "mock", "text": "Hola a todos, esto es una prueba.",
                                                  "params": {"seed": 3}}).json()["job_id"]
    body = wait_done(client, gen_id)
    assert body["status"] == "COMPLETED"
    first = client.get(f"/api/generation/{gen_id}/stream/0")
    assert first.status_code == 200 and first.headers["content-type"] == "audio/wav"
    assert first.headers["cache-control"] == "no-store"
    halves = [sf.read(io.BytesIO(client.get(f"/api/generation/{gen_id}/stream/{i}").content))[0] for i in (0, 1)]
    final, _ = sf.read(settings.data_dir / "generated" / gen_id / "raw.wav")
    assert sum(h.size for h in halves) == final.size  # the two halves are the whole take
    missing = client.get(f"/api/generation/{gen_id}/stream/7")
    assert missing.status_code == 404 and missing.json()["message"] == "Ese trozo de audio todavía no existe."
    assert client.get(f"/api/generation/{gen_id}/stream/-1").status_code == 422


def test_job_events_carry_the_chunk_count(client, engines):  # noqa: F811
    gen_id = client.post("/api/generation", json={"engine": "mock", "text": "Una frase corta."}).json()["job_id"]
    wait_done(client, gen_id)
    with client.stream("GET", f"/api/jobs/{gen_id}/events") as stream:
        payload = next(line for line in stream.iter_lines() if line.startswith("data:"))
    assert '"chunks"' in payload


def test_segmented_non_streaming_engine_sends_each_segment_with_its_pause(client, adv_engines, settings,  # noqa: F811
                                                                         monkeypatch):
    _seg, instr = adv_engines
    original = instr.capabilities
    monkeypatch.setattr(instr, "capabilities",
                        lambda variant=None: original(variant).model_copy(update={"supports_streaming": False}))
    body = {"engine": "instructmock", "text": "Primera parte. [pausa:500ms] Segunda parte."}
    gen_id = client.post("/api/generation", json=body).json()["job_id"]
    done = wait_done(client, gen_id)
    assert done["status"] == "COMPLETED" and done["metrics"]["segments"] == 2
    folder = settings.data_dir / "generated" / gen_id
    chunks = sorted(folder.glob("stream_*.wav"))
    assert len(chunks) == 2
    first, sr = sf.read(chunks[0])
    segment, _ = sf.read(folder / "seg_000.wav")
    assert first.size == segment.size + int(sr * 0.5)  # segment audio + its 500 ms pause
    assert np.allclose(first[segment.size:], 0.0)


@pytest.mark.parametrize("supports", [True, False])
def test_on_audio_only_for_engines_that_take_it(supports):
    from app.services.generation_service import _accepts_live_audio

    class Caps:
        supports_streaming = supports

    class Old:  # a plugin written before on_audio existed
        def capabilities(self, variant=None):
            return Caps()

        def generate(self, request, progress, cancel):
            return None

    class New(Old):
        def generate(self, request, progress, cancel, on_audio=None):
            return None

    assert _accepts_live_audio(Old(), None) is False
    assert _accepts_live_audio(New(), None) is supports


def test_qwen_stream_hook_turns_talker_steps_into_audio(monkeypatch):
    """The plugin hook feeds each step's codes to the streamer and flushes the rest when generation ends."""
    import types

    from app.engines.qwen3tts.plugin import Qwen3TTSBackend

    hooks = []

    class Talker:
        def register_forward_hook(self, hook):
            hooks.append(hook)
            return types.SimpleNamespace(remove=lambda: hooks.remove(hook))

    class Decoder(torch.nn.Module):
        total_upsample = 4

        def forward(self, codes):  # (1, quantizers, frames) → (1, 1, frames * 4)
            return codes[:, :1].repeat_interleave(4, dim=-1).float()

    tokenizer = types.SimpleNamespace(model=types.SimpleNamespace(decoder=Decoder()),
                                      get_output_sample_rate=lambda: 24_000)
    engine = Qwen3TTSBackend()
    engine._model = types.SimpleNamespace(model=types.SimpleNamespace(talker=Talker(), speech_tokenizer=tokenizer))
    received = []

    with engine._stream_hook(lambda audio, sr: received.append((audio.copy(), sr)), prefix=None):
        (hook,) = hooks
        hook(None, (), types.SimpleNamespace(hidden_states=((), None)))  # prefill step: no codes
        for i in range(10):
            hook(None, (), types.SimpleNamespace(hidden_states=((), torch.full((1, 16), i))))
    assert not hooks  # removed afterwards
    assert [a.size // 4 for a, _ in received] == [8, 2]  # first chunk at 8 frames, the rest on flush
    assert all(sr == 24_000 for _, sr in received)
    assert np.array_equal(np.concatenate([a for a, _ in received]), np.repeat(np.arange(10, dtype=np.float32), 4))

    with engine._stream_hook(None, prefix=None):  # no callback: nothing registered
        assert not hooks
