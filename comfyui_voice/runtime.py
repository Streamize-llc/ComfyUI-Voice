"""Runtime routing: in-process vs subprocess isolation, behind one interface.

An engine declares ``isolation`` in its capabilities. Nodes never know which
runtime they hit — they call ``get_runtime(id).generate(task, req)``. That's what
makes "isolate this engine" a one-field change instead of a rewrite, and what
lets the suite host engines with mutually conflicting dependency pins.

Subprocess engines run in their own venv (``runtimes/<id>/.venv``) via a
persistent worker process. Protocol: line-delimited JSON on the worker's real
stdout (model/library noise is redirected to stderr so it can't corrupt the
channel). Audio crosses the boundary as ``.npy`` temp files, never as JSON. UTF-8
is forced so Korean text survives the Windows cp949 console.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from typing import Any

from .manager import MANAGER
from .paths import SUITE_ROOT, venv_python
from .registry import get_engine

log = logging.getLogger("comfyui_voice.runtime")

_WORKER = os.path.join(os.path.dirname(__file__), "runtimes", "worker.py")


class InprocRuntime:
    """Run the engine in this process (lazy-imported deps)."""

    def __init__(self, engine_id: str) -> None:
        self.engine_id = engine_id

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        engine = MANAGER.get(self.engine_id)
        return engine.generate(task, req)


def _module_path(engine_id: str) -> str:
    """Normalize an engine class module to the top-level ``comfyui_voice.*`` path."""
    mod = get_engine(engine_id).__module__
    suffix = mod.split("comfyui_voice.")[-1]
    return f"comfyui_voice.{suffix}"


def _externalize(req: dict[str, Any], tmpdir: str) -> dict[str, Any]:
    """Replace non-JSON values (audio tensors) with .npy file references."""
    import numpy as np

    out: dict[str, Any] = {}
    for key, value in req.items():
        if isinstance(value, dict) and "waveform" in value:  # an AUDIO dict
            wav = value["waveform"]
            arr = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
            path = os.path.join(tmpdir, f"in_{key}.npy")
            np.save(path, arr)
            out[key] = {"__audio_npy__": path, "sample_rate": int(value.get("sample_rate", 0))}
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, dict)):
            try:
                json.dumps(value)
                out[key] = value
            except TypeError:
                pass  # drop unserializable extras
    return out


class SubprocessRuntime:
    """Run a venv-isolated engine via a persistent worker process."""

    _procs: dict[str, subprocess.Popen] = {}
    _lock = threading.Lock()

    def __init__(self, engine_id: str, caps) -> None:
        self.engine_id = engine_id
        self.caps = caps

    def _python(self) -> str:
        # Dedicated venv if set up; else fall back to the host interpreter
        # (used by the isolation self-test and engines with no conflicting pins).
        return venv_python(self.engine_id) or sys.executable

    def _proc(self) -> subprocess.Popen:
        with SubprocessRuntime._lock:
            proc = SubprocessRuntime._procs.get(self.engine_id)
            if proc is not None and proc.poll() is None:
                return proc

            env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            cmd = [
                self._python(),
                _WORKER,
                "--root", SUITE_ROOT,
                "--module", _module_path(self.engine_id),
                "--engine", self.engine_id,
            ]
            log.info("starting subprocess worker for '%s'", self.engine_id)
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=env,
                bufsize=1,
            )
            threading.Thread(target=self._drain_stderr, args=(proc,), daemon=True).start()
            ready = self._read_msg(proc)
            if not ready.get("ready"):
                raise RuntimeError(
                    f"worker for '{self.engine_id}' failed to start: {ready}"
                )
            SubprocessRuntime._procs[self.engine_id] = proc
            return proc

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        for line in proc.stderr:
            log.debug("[%s worker] %s", self.engine_id, line.rstrip())

    def _read_msg(self, proc: subprocess.Popen) -> dict:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"worker '{self.engine_id}' closed the channel unexpectedly")
        return json.loads(line)

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        proc = self._proc()
        with tempfile.TemporaryDirectory(prefix="voice_io_") as tmp:
            payload = {"op": "generate", "task": task, "req": _externalize(req, tmp)}
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            resp = self._read_msg(proc)

            if "error" in resp:
                err = resp["error"]
                raise RuntimeError(
                    f"engine '{self.engine_id}' failed: {err.get('type')}: "
                    f"{err.get('message')}\n{err.get('traceback', '')}"
                )
            if "waveform_npy" in resp:
                arr = np.load(resp["waveform_npy"])
                import torch

                return {"waveform": torch.from_numpy(arr), "sample_rate": int(resp["sample_rate"])}
            return resp.get("result", {})

    @classmethod
    def shutdown_all(cls) -> None:
        with cls._lock:
            for engine_id, proc in list(cls._procs.items()):
                try:
                    proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                    proc.stdin.flush()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                cls._procs.pop(engine_id, None)


def get_runtime(engine_id: str):
    caps = get_engine(engine_id).CAPS
    if caps.isolation == "subprocess":
        return SubprocessRuntime(engine_id, caps)
    return InprocRuntime(engine_id)
