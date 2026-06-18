"""Generic subprocess worker for venv-isolated voice engines.

Runs INSIDE a per-engine venv (its own torch/transformers/etc). The host
(comfyui_voice.runtime.SubprocessRuntime) launches one persistent worker per
engine and talks to it over line-delimited JSON.

Protocol (the worker's REAL stdout only):
  -> {"ready": true}                               once the model is loaded
  <- {"op":"generate","task":"tts","req":{...}}    request from host
  -> {"waveform_npy": "...", "sample_rate": N}     tts result (audio via .npy)
  -> {"result": {...}}                             non-audio result (e.g. asr)
  -> {"error": {"type","message","traceback"}}     surfaced failure
  <- {"op":"shutdown"}                             clean exit

All library/model noise (prints, tqdm) is redirected to stderr so it can never
corrupt the JSON channel on stdout.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
import traceback

# Protocol uses the real stdout; everything else goes to stderr.
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr


def _send(obj: dict) -> None:
    _REAL_STDOUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _REAL_STDOUT.flush()


def _materialize(req: dict) -> dict:
    """Turn .npy audio references from the host back into AUDIO dicts."""
    import numpy as np

    out = {}
    for key, value in req.items():
        if isinstance(value, dict) and "__audio_npy__" in value:
            arr = np.load(value["__audio_npy__"])
            out[key] = {"waveform": arr, "sample_rate": int(value.get("sample_rate", 0))}
        else:
            out[key] = value
    return out


def _serialize_output(out: dict) -> dict:
    import numpy as np

    if isinstance(out, dict) and "waveform" in out:
        wav = out["waveform"]
        arr = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
        fd, path = tempfile.mkstemp(suffix=".npy", prefix="voice_out_")
        os.close(fd)
        np.save(path, arr)
        return {"waveform_npy": path, "sample_rate": int(out.get("sample_rate", 0))}
    return {"result": out}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--engine", required=True)
    args = parser.parse_args()

    if args.root not in sys.path:
        sys.path.insert(0, args.root)

    try:
        importlib.import_module(args.module)  # self-registers the engine
        from comfyui_voice.registry import ENGINE_REGISTRY

        engine = ENGINE_REGISTRY[args.engine]()
        engine.load()
    except Exception as exc:  # noqa: BLE001
        _send({"ready": False, "error": {"type": type(exc).__name__, "message": str(exc),
                                          "traceback": traceback.format_exc()}})
        return 1

    _send({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            op = msg.get("op")
            if op == "shutdown":
                break
            if op == "generate":
                req = _materialize(msg.get("req", {}))
                out = engine.generate(msg.get("task", "tts"), req)
                _send(_serialize_output(out))
            else:
                _send({"error": {"type": "ValueError", "message": f"unknown op {op!r}"}})
        except Exception as exc:  # noqa: BLE001
            _send({"error": {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()}})

    try:
        engine.unload()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
