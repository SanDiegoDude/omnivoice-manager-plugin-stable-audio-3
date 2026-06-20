# Stable Audio 3 — OmniVoice Manager plug-in

A drop-in plug-in for [OmniVoice Manager](https://github.com/SanDiegoDude/OmniVoice-Manager)
that adds **text-to-audio generation** — foley, SFX, music, instrument loops and
one-shots — powered by Stability AI's **Stable Audio 3 Medium** model.

It runs the model in its **own isolated Python environment** (a sidecar), so its
torch/flash-attn pins never collide with the host app, and it talks to OmniVoice
over the standard plug-in protocol. Nothing in the host app is hard-wired to this
plug-in: it contributes its UI (the "Sound Lab" modal, Sound Library and
multitrack entry points) declaratively through its `plugin.json` manifest.

> **Why a separate repo?** OmniVoice Manager is Apache-2.0. Stable Audio 3 ships
> under Stability AI's **Community License** with its own terms. Keeping this
> plug-in in its own repo keeps those licenses cleanly separated and lets the
> plug-in version independently of the app. See [Licensing](#licensing).

---

## What it does

- **Four generation modes** — **One-shot / SFX / Music / Instrument** — each with
  a tuned system prompt (ported from the reference ComfyUI workflow).
- **Smart reprompt:** optionally rewrites your short description into a detailed,
  SA3-shaped prompt using whatever Script-AI provider the host has configured. The
  sidecar calls the host's `reprompt` hook with its own category prompts, so the
  prompt engineering lives here and the host stays generic.
- **Sound Lab UI:** waveform preview, autoplay, speed/trim/dB, **Generate**
  (reprompt + render) vs **Reroll Generation** (re-render the same prompt on a new
  seed, no LLM), save to the Sound Library, or drop straight onto a timeline.
- **No duration ceiling:** generate well past the model's published 120 s if your
  VRAM allows (clean multi-minute takes are routine).

---

## Requirements

- A working **OmniVoice Manager** install (this plug-in lives inside its
  `plugins/` directory).
- An **NVIDIA GPU with CUDA** (x86_64 or ARM64 / Grace-Blackwell — see below).
  ~6.5 GB VRAM for SA3 Medium; honors the host's low-VRAM mode.
- A Hugging Face account that has **accepted the SA3 Medium license** (the weights
  are gated). See [Gated model setup](#gated-model-setup).

---

## Installation

The plug-in is just a correctly-shaped folder inside the host's `plugins/`
directory. Any of these get it there:

### 1. From the app (recommended)

```bash
# CLI — clone + build the sidecar env in one step
omnivoice-plugin install https://github.com/SanDiegoDude/omnivoice-manager-plugin-stable-audio-3
```

or trigger it from the running manager:

```bash
curl -X POST http://localhost:8200/api/plugins/install \
  -H 'Content-Type: application/json' \
  -d '{"git_url":"https://github.com/SanDiegoDude/omnivoice-manager-plugin-stable-audio-3","bootstrap":true}'
```

### 2. Manual git clone

```bash
cd /path/to/OmniVoice-Manager/plugins
git clone https://github.com/SanDiegoDude/omnivoice-manager-plugin-stable-audio-3 stable-audio-3
cd stable-audio-3
./bootstrap.sh        # builds the isolated .venv (add --with-model to pre-fetch weights)
```

### 3. Copy / installer

Copy (or have a Windows installer drop) this folder into `plugins/` as
`plugins/stable-audio-3/`, then run `./bootstrap.sh`. The host auto-discovers any
folder that contains a valid `plugin.json` — it doesn't care how it got there.

After installing, the plug-in appears automatically in OmniVoice (Sound Library →
**Sound Lab — Stable Audio 3**, and the empty-audio-track double-click menu).

---

## Building the sidecar environment (`bootstrap.sh`)

`bootstrap.sh` creates `./.venv` with the right torch build for your platform and
installs `stable-audio-3`. It is re-runnable / idempotent.

| Platform | torch | Notes |
| --- | --- | --- |
| Linux **x86_64** | `2.7.1` (cu126) | + prebuilt flash-attn 2 wheel (matches SA3's pin) |
| Linux **aarch64** (GB10 "DGX Spark"/EdgeXpert, GH200) | `2.10.0` (cu130) | torch 2.7.x has no aarch64 CUDA wheels; SA3 installed with `--no-deps` to override its pin. flash-attn falls back to torch SDPA |

```bash
./bootstrap.sh                 # build the env (no model download)
./bootstrap.sh --with-model    # also download SA3 Medium (needs an HF token)
```

**Overrides (env):** `SA3_CUDA`, `SA3_TORCH`, `SA3_PYTHON`, `SA3_FLASH_WHEEL`.
For an optional flash-attn source build on Blackwell ARM (~60–75 min; SDPA is
already fast and numerically identical there): `SA3_BUILD_FLASH=1`
`SA3_FLASH_VER=2.8.3` `SA3_FLASH_ARCH=12.0` (sm_120 runs on GB10's sm_121 via
CUDA forward-compat; use `9.0` for GH200) `MAX_JOBS=4` `CUDA_HOME=/usr/local/cuda`.

---

## Gated model setup

SA3 Medium weights are gated on Hugging Face. They **auto-download on first
generate**, but only once you've granted access:

1. Accept the license at
   <https://huggingface.co/stabilityai/stable-audio-3-medium>.
2. Authenticate this machine (browser sign-in is **not** enough — the download
   runs from the CLI/sidecar):
   ```bash
   huggingface-cli login        # paste an hf_… read token
   # or: export HF_TOKEN=hf_… before launching OmniVoice / bootstrap --with-model
   ```

If a generate fails on the gate, OmniVoice shows a help note linking the license
page and a full troubleshooting page (bundled here as [`HELP.html`](HELP.html)).
Once the weights are local, the note disappears on its own.

---

## How it plugs in (for plug-in authors)

This repo doubles as a reference implementation of the OmniVoice plug-in contract.
The host injects its pure-stdlib SDK onto the sidecar's `PYTHONPATH`, so the
sidecar simply does `from omnivoice_plugin import run`. The pieces:

| File | Role |
| --- | --- |
| `plugin.json` | Manifest: identity, isolation, GPU/VRAM hints, `capabilities`, `needs` (model, gating, help, bootstrap), and the declarative `ui` (contributions + Sound Lab schema). |
| `sidecar.py` | Entrypoint — runs in the plug-in's own venv. Loads SA3, handles `generate` / `load` / `unload` / `health`, and calls the host's `reprompt` hook. |
| `prompts.py` | Plug-in-private prompt construction (category system prompts). |
| `category_prompts.json` | The SA3 category prompt data. |
| `bootstrap.sh` | One-shot, per-platform environment builder. |
| `HELP.html` | Bundled troubleshooting page, served by the host at `/api/plugins/{id}/help`. |

Key concepts demonstrated: **isolation** (own venv), **GPU serialization +
low-VRAM**, **reprompting via the host LLM hook** (prompt logic stays in the
plug-in), **declarative UI contributions** (`ui.contributions` + `ui.lab`), and
**sound-library + project-data** integration. See the host's
`docs/plugins.md` for the full authoring guide and the wire protocol.

### Generate (generic route — works for any audio-generator plug-in)

```jsonc
POST /api/plugins/stable-audio-3/generate
{
  "fields": {                  // this plug-in's ui.lab schema payload
    "prompt": "heavy wooden door creaking open in a stone hall",
    "category": "SFX",         // Music | Instrument | SFX | One-shot
    "duration": 8.0,
    "steps": 8,
    "cfg": 1.0,
    "seed": null
  },
  "reprompt": true,            // sidecar rewrites via the host's Script-AI provider
  "save": false,               // false → temp preview; true → ingest immediately
  "save_path": null,           // library folder/name when save=true
  "session_id": "..."          // optional: tag the open project
}
```

The job's `result` carries the final `prompt`, `raw_prompt`, `reprompted`, the
sample rate, an `audio_url`, and either a saved `sound` descriptor (`save=true`)
or a `temp` handle to save later via `POST /api/sounds/import-temp`.

---

## Licensing

- **This plug-in's code** (`sidecar.py`, `prompts.py`, `bootstrap.sh`,
  `plugin.json`, etc.) is licensed **Apache-2.0** — see [`LICENSE`](LICENSE).
- **Stable Audio 3 Medium** (the model weights) and the **`stable-audio-3`**
  Python package are **not** covered by that license. They are governed by
  **Stability AI's licenses** (the Stability AI Community License and the model's
  gated terms). You must review and accept those separately:
  <https://huggingface.co/stabilityai/stable-audio-3-medium> and
  <https://github.com/Stability-AI/stable-audio-3>.

Installing/using this plug-in downloads and runs that model under Stability AI's
terms; the Apache-2.0 grant here applies only to the integration code.
