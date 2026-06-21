"""Stable Audio 3 sidecar — runs inside plugins/stable-audio-3/.venv.

Loads SA3 Medium and generates audio from a (already reprompted) text prompt.
Talks to the OmniVoice host over the stdio protocol via the pure-stdlib SDK that
the host puts on PYTHONPATH. No host packages are imported here — this process is
fully isolated in its own venv (torch 2.7.x + flash-attn 2 on x86_64; torch
2.10.x with SDPA fallback on aarch64 / Grace-Blackwell). See bootstrap.sh.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import sys

# macOS / Apple Silicon: SA3 runs on the Metal (MPS) backend, where a handful of
# ops the model uses aren't implemented yet. Allow torch to silently fall back to
# CPU for those ops instead of hard-crashing mid-generation — the difference is
# slower, not broken. Harmless on CUDA/CPU (the flag is MPS-only). Must be set
# before torch is imported anywhere, so do it at module import time.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# The host exports OMNIVOICE_PLUGIN_SDK; make the SDK importable even if launched
# directly for debugging.
_sdk = os.environ.get("OMNIVOICE_PLUGIN_SDK")
if _sdk and _sdk not in sys.path:
    sys.path.insert(0, _sdk)

from omnivoice_plugin import run  # noqa: E402

MODEL_ID = os.environ.get("SA3_MODEL", "medium")


@contextlib.contextmanager
def _stdout_to_stderr():
    """Divert Python-level stdout to stderr for the wrapped block.

    The stable_audio_3 package prints diagnostics straight to stdout (flash-attn
    import notices on load; a CUDA-centric "without a GPU … not designed to run
    on cpu" warning that also fires on MPS). The host protocol reserves stdout
    for JSON, so those prints would be stray noise on the wire. The SDK captured
    the real stdout at startup, so its protocol writes are unaffected — only the
    library's prints get routed to stderr → the plug-in's log file."""
    saved = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = saved


def _mps_available() -> bool:
    """True on Apple Silicon when the Metal (MPS) backend is usable."""
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        return False


def _filter_kwargs(fn, kwargs: dict) -> dict:
    """Keep only kwargs the target callable actually accepts (SA3's generate
    signature varies across versions; we pass what's supported and drop the
    rest instead of crashing)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    allowed = set(sig.parameters)
    return {k: v for k, v in kwargs.items() if k in allowed}


class StableAudio3:
    def __init__(self) -> None:
        self._model = None
        self._sr = 44100

    # ---- lifecycle ----
    def _model_repo(self) -> str:
        """The HuggingFace repo id for the weights (so we can pre-fetch with
        progress before handing off to from_pretrained)."""
        repo = os.environ.get("SA3_REPO")
        if repo:
            return repo
        return MODEL_ID if "/" in MODEL_ID else f"stabilityai/stable-audio-3-{MODEL_ID}"

    def _prefetch_weights(self, ctx) -> None:
        """Download the model snapshot up-front, surfacing live progress.

        HuggingFace's own tqdm writes to stderr (the plug-in log only), so a
        download looks like a frozen spinner from the app/CLI. Routing
        ``snapshot_download`` through a tqdm subclass lets us report real % into
        ``ctx.progress`` — which shows in the Sound Lab modal and the job poll.
        Cached weights make this a fast no-op. Download failures (gated repo, no
        token, network) propagate so the host can surface the gate help."""
        repo = self._model_repo()
        try:
            from huggingface_hub import snapshot_download
            from tqdm.auto import tqdm as _tqdm
        except Exception as e:  # noqa: BLE001
            ctx.log(f"download-progress unavailable ({e}); deferring to from_pretrained", "warn")
            return

        import time

        state = {"pct": -1, "t": 0.0}

        def report(n: int, total: int) -> None:
            if not total:
                return
            pct = int(n * 100 / total)
            now = time.monotonic()
            if pct == state["pct"] and now - state["t"] < 1.0:
                return
            state["pct"], state["t"] = pct, now
            msg = f"Downloading model… {pct}% ({n / 1048576:.0f}/{total / 1048576:.0f} MB)"
            ctx.progress(stage="downloading_model", message=msg, percent=pct)
            ctx.log(msg)

        class _ProgressTqdm(_tqdm):
            def __init__(self, *a, **k):
                k["disable"] = False  # force n/total updates even off a TTY
                super().__init__(*a, **k)

            def display(self, *a, **k):
                try:
                    # Only the per-file byte bars, not the "Fetching N files" counter.
                    if self.total and getattr(self, "unit", "") == "B":
                        report(self.n, self.total)
                except Exception:  # noqa: BLE001
                    pass
                return super().display(*a, **k)

        ctx.progress(stage="downloading_model", message="Preparing model weights…")
        ctx.log(f"Ensuring model weights present: {repo}")
        try:
            snapshot_download(repo, tqdm_class=_ProgressTqdm)
        except TypeError:
            snapshot_download(repo)  # hub without tqdm_class kwarg — fetch sans %
        ctx.log("Model weights present.")

    def _ensure_model(self, ctx):
        if self._model is not None:
            return self._model
        self._prefetch_weights(ctx)
        ctx.progress(stage="loading_model", message=f"Loading Stable Audio 3 ({MODEL_ID})…")
        ctx.log(f"Loading Stable Audio 3 model '{MODEL_ID}'")
        # SA3's import + load print library diagnostics to stdout; keep them off
        # the protocol stream (they land in the plug-in log instead).
        with _stdout_to_stderr():
            from stable_audio_3 import StableAudioModel

            # device=None lets SA3 auto-select: cuda → mps (Apple Silicon) → cpu.
            # On non-CUDA it also forces full precision (model_half=False).
            self._model = StableAudioModel.from_pretrained(MODEL_ID)
        # Resolve the model's native sample rate when it exposes one.
        for attr in ("sample_rate", "sampling_rate", "sr"):
            v = getattr(self._model, attr, None)
            if isinstance(v, int) and v > 0:
                self._sr = v
                break
        device = getattr(self._model, "device", "?")
        ctx.log(f"Model loaded (device={device}, sample_rate={self._sr})")
        if str(device) == "mps":
            ctx.log("Running on Apple Silicon (MPS) — generation is functional but "
                    "much slower than a discrete GPU; long takes need patience.", "warn")
        return self._model

    def load(self, ctx):
        self._ensure_model(ctx)

    def unload(self, ctx):
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Free the Metal (MPS) allocator's cache on Apple Silicon too — the
            # symmetric counterpart to cuda.empty_cache() so the host's GPU
            # serialization actually reclaims unified memory on a Mac.
            if _mps_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def health(self, ctx):
        info = {"ok": True, "loaded": self._model is not None, "model": MODEL_ID}
        try:
            import torch

            info["cuda"] = bool(torch.cuda.is_available())
            info["mps"] = _mps_available()
            info["torch"] = torch.__version__
            # The compute device SA3 selected (cuda → mps → cpu). Read it off the
            # loaded model when present; otherwise report what we'd pick.
            if self._model is not None and getattr(self._model, "device", None):
                info["device"] = str(self._model.device)
            else:
                info["device"] = "cuda" if torch.cuda.is_available() else ("mps" if info["mps"] else "cpu")
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                info["vram_free_mb"] = int(free // (1024 * 1024))
                info["vram_total_mb"] = int(total // (1024 * 1024))
        except Exception as e:  # noqa: BLE001
            info["torch_error"] = str(e)
        try:
            import flash_attn  # noqa: F401

            info["flash_attn"] = True
        except Exception:  # noqa: BLE001
            info["flash_attn"] = False
        return info

    # ---- prompt engineering ----
    def _reprompt(self, ctx, raw_prompt: str, category: str, duration: float, provider_id):
        """Rewrite the user's description into an SA3-tuned prompt via the host's
        configured Script-AI provider. The category system prompts live in this
        plug-in's own prompts.py (ported from the reference workflow) — we build
        the system/user messages here and call back into the host's `reprompt`
        hook, so the host stays generic and the prompt logic stays with the
        plug-in. Any failure (no provider, network) falls back to the raw text."""
        try:
            import prompts as sa3_prompts  # local module (CWD = plug-in dir)

            cat = sa3_prompts.normalize_category(category or "SFX")
            ctx.progress(stage="reprompt", message="Rewriting prompt…")
            res = ctx.host_call(
                "reprompt",
                system=sa3_prompts.system_prompt(cat),
                user=sa3_prompts.user_message(raw_prompt, duration),
                provider_id=provider_id,
                max_tokens=400,
            )
            text = (res or {}).get("text") if isinstance(res, dict) else None
            if text and text.strip():
                return text.strip(), True
        except Exception as e:  # noqa: BLE001
            ctx.log(f"reprompt skipped: {e}", "warn")
        return raw_prompt, False

    # ---- generation ----
    def generate(self, ctx, **payload):
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("A prompt is required.")
        duration = float(payload.get("duration") or 8.0)
        steps = int(payload.get("steps") or 8)
        cfg = float(payload.get("cfg") if payload.get("cfg") is not None else 1.0)
        seed = payload.get("seed")
        low_vram = bool(payload.get("low_vram", False))
        negative = payload.get("negative_prompt")
        category = (payload.get("category") or "").strip()
        do_reprompt = bool(payload.get("reprompt"))
        provider_id = payload.get("provider_id")

        raw_prompt = prompt
        reprompted = False
        if do_reprompt:
            prompt, reprompted = self._reprompt(ctx, raw_prompt, category, duration, provider_id)

        model = self._ensure_model(ctx)
        ctx.progress(stage="generating", message="Generating audio…", duration=duration)

        # Map our generic params onto SA3's generate() signature using its EXACT
        # parameter names. SA3's generate has a **sampler_kwargs catch-all that
        # forwards anything unrecognized straight down into the DiT forward, so we
        # must pass only real names (e.g. cfg_scale, NOT cfg/guidance_scale; steps,
        # NOT num_steps) or generation crashes deep in the transformer. cfg<=1
        # mirrors the reference workflow's distilled, low-CFG fast path.
        # SA3's generate() defaults sample_size to ~120s of samples (5292032 @
        # 44.1k) and clamps the requested length to it (_adapt_sample_size →
        # min(target, sample_size)), so anything past ~120s gets hard-truncated.
        # We don't impose a ceiling: size sample_size to cover the request (plus
        # the model's 6s schedule padding + a little slack), so the only real
        # limit is the user's VRAM. Never go below the model default for short
        # takes. (truncate_output_to_duration then trims back to `duration`.)
        pad_s = 6.0
        sample_size = max(5292032, int((duration + pad_s + 8.0) * float(self._sr or 44100)))
        candidate = {
            "prompt": prompt,
            "duration": duration,
            "steps": steps,
            "cfg_scale": cfg,
            "sample_size": sample_size,
            "duration_padding_sec": pad_s,
            "negative_prompt": negative,
            # SA3 uses -1 (not None) to mean "pick a random seed".
            "seed": int(seed) if seed is not None else -1,
        }
        if low_vram:
            # Chunked VAE decode trades speed for a lower decode-time VRAM peak.
            candidate["chunked_decode"] = True
        candidate = {k: v for k, v in candidate.items() if v is not None}
        kwargs = _filter_kwargs(model.generate, candidate)
        ctx.log(f"generate kwargs: {sorted(kwargs)} (steps={steps}, cfg={cfg}, dur={duration}, sample_size={sample_size})")

        with _stdout_to_stderr():
            out = model.generate(**kwargs)
        wav, sr = self._normalize_output(out)

        out_path = ctx.tmp_path(".wav")
        self._save_wav(out_path, wav, sr)
        dur_s = round(float(wav.shape[-1]) / float(sr), 2) if sr else duration
        ctx.progress(stage="done", message="Done")
        return {
            "audio_path": out_path,
            "sample_rate": int(sr),
            "duration_s": dur_s,
            "prompt": prompt,
            "raw_prompt": raw_prompt,
            "reprompted": reprompted,
            "category": category or None,
        }

    # ---- output handling ----
    def _normalize_output(self, out):
        """Coerce SA3's return value into (2D float tensor [C, N], sample_rate)."""
        import torch

        sr = self._sr
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], int):
            out, sr = out
        if hasattr(out, "detach"):  # torch tensor
            t = out.detach().to("cpu").float()
        else:
            t = torch.as_tensor(out).float()
        t = t.squeeze()
        if t.dim() == 1:
            t = t.unsqueeze(0)  # mono -> [1, N]
        elif t.dim() > 2:
            t = t.reshape(t.shape[-2], t.shape[-1])
        # Guard against [N, C] when channels look like the last axis.
        if t.shape[0] > t.shape[1] and t.shape[1] <= 2:
            t = t.transpose(0, 1)
        return t, int(sr)

    def _save_wav(self, path, wav, sr):
        try:
            import torchaudio

            torchaudio.save(path, wav, sr)
            return
        except Exception:  # noqa: BLE001
            pass
        # Fallback: soundfile expects [N, C].
        import soundfile as sf

        sf.write(path, wav.transpose(0, 1).numpy(), sr)


run(StableAudio3())
