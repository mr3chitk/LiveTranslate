import io
import json
import logging
import shutil
import subprocess
import tarfile
import tempfile
import time
import wave
import zipfile
from pathlib import Path

import httpx
import numpy as np

log = logging.getLogger("LiveTranslate.ASR.Haas")

# speech.cpp release asset (GPU build). CPU-only builds are not downloaded here.
_RELEASE_ASSET = "-windows-x64-vulkan.zip"
_RELEASE_API = "https://api.github.com/repos/nyosegawa/speech.cpp/releases/latest"

# Model served by the speech.cpp backend. `haas-ja-v1.0` is the ASR model;
# silero-vad is passed so speech.cpp can chunk transcription by speech regions.
_DEFAULT_MODEL = "haas-ja-v1.0"
_DEFAULT_PORT = 8766
_DEFAULT_URL = f"http://127.0.0.1:{_DEFAULT_PORT}"


def _find_exe():
    """Look for an existing `speech.exe` in the app's speech.cpp folder, the
    app folder, project root, or on PATH."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "speech.cpp" / "speech.exe",
        here / "speech.exe",
        Path.cwd() / "speech.cpp" / "speech.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    found = shutil.which("speech.exe")
    return Path(found) if found else None


def _ensure_speech_bin(folder: Path) -> Path:
    """Make sure the speech.cpp `speech.exe` exists under `folder`.

    If missing, download the latest Windows Vulkan release from GitHub and
    extract `speech.exe` into `folder`. Returns the path to the executable.
    """
    exe = folder / "speech.exe"
    if exe.exists():
        return exe

    folder.mkdir(parents=True, exist_ok=True)
    log.info(f"speech.cpp not found under {folder}; downloading latest release")

    tmp_dir = None
    try:
        from model_manager import apply_cache_env

        apply_cache_env()
        import urllib.request

        tmp_dir = tempfile.mkdtemp(prefix="speech-cpp-dl-")
        zip_path = Path(tmp_dir) / "release.zip"

        # Fetch the release asset URL (skips the JSON body).
        req = urllib.request.Request(
            _RELEASE_API, headers={"Accept": "application/vnd.github+json"}
        )
        download_url = None
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for asset in data.get("assets", []):
                if _RELEASE_ASSET in asset["browser_download_url"]:
                    download_url = asset["browser_download_url"]
                    break
        if not download_url:
            raise FileNotFoundError(
                f"Release asset '{_RELEASE_ASSET}' not found in latest release"
            )

        with urllib.request.urlopen(download_url, timeout=600) as src, open(
            zip_path, "wb"
        ) as dst:
            shutil.copyfileobj(src, dst)

        exe_parent = exe.parent
        if zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path) as zf:
                members = zf.namelist()
                if not any(m.lower().endswith("speech.exe") for m in members):
                    raise RuntimeError(
                        "speech.cpp archive does not contain speech.exe"
                    )
                # Extract the whole archive: the backend needs models/DLLs/config
                # alongside speech.exe, not just the executable itself.
                zf.extractall(exe_parent)
        elif zip_path.suffix == ".tar.gz":
            with tarfile.open(zip_path) as tf:
                names = tf.getnames()
                if not any(m.lower().endswith("speech.exe") for m in names):
                    raise RuntimeError(
                        "speech.cpp archive does not contain speech.exe"
                    )
                tf.extractall(exe_parent)
        else:
            raise RuntimeError(f"Unsupported archive format: {zip_path.suffix}")

        log.info(f"Extracted speech.cpp to {exe}")
        return exe
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class HaasASREngine:
    """Speech-to-text via the speech.cpp `speech serve` backend.

    The backend is launched (once) as a subprocess and audio is sent to it over
    HTTP using OpenAI's `/v1/audio/transcriptions` API. Implements the same
    interface as ASREngine for drop-in replacement.
    """

    def __init__(
        self,
        model=_DEFAULT_MODEL,
        server_url=_DEFAULT_URL,
        timeout=30.0,
        diarize=False,
        diarize_model="nemotron-3-diarization",
    ):
        self._model = model
        self._url = (server_url or _DEFAULT_URL).rstrip("/") + "/v1/audio/transcriptions"
        self._health_url = server_url.rstrip("/") + "/health"
        # Enable speaker diarization via the Nemotron-3-Diarization backend model.
        # When off, the backend still works but returns only `text` (response_format=json).
        self._diarize = diarize
        self._diarize_model = diarize_model
        # trust_env=False bypasses the system / HTTP(S)_PROXY: the URL points
        # straight at a localhost backend, and routing localhost through a proxy
        # returns a non-JSON error page instead of connecting.
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=5.0), trust_env=False
        )
        self.language = None
        self._proc = None
        self._start_backend()

    def _start_backend(self):
        exe = _find_exe()
        if exe is None:
            exe = _ensure_speech_bin(Path(__file__).resolve().parent / "speech.cpp")
        log.info(f"Starting speech.cpp backend: {exe}")
        serve_args = [
            str(exe),
            "serve",
            self._model,
            "--host",
            "127.0.0.1",
            "--port",
            str(_DEFAULT_PORT),
        ]
        # Load the diarization model so `response_format=diarized_json` returns
        # per-speaker turns. The model is loaded lazily on first diarization request.
        # silero-vad is needed for diarization
        if self._diarize:
            serve_args += ["silero-vad", self._diarize_model]
            log.info(f"Speaker diarization enabled using {self._diarize_model}")
        self._proc = subprocess.Popen(
            serve_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_ready()

    def _wait_ready(self, timeout=120.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise ConnectionError(
                    "speech.cpp backend exited during startup "
                    f"(code={self._proc.returncode})"
                )
            try:
                r = self._client.get(self._health_url, timeout=2.0)
                if r.status_code == 200:
                    log.info("speech.cpp backend is ready")
                    return
            except Exception:
                time.sleep(1.0)
        raise ConnectionError(
            f"speech.cpp backend did not become ready at {self._health_url} within "
            f"{timeout:g}s"
        )

    # --- ASRClient-compatible shim ----------------------------------------
    # The pipeline drives the active ASR backend through the ASRClient interface
    # (a worker-process proxy). This engine runs in-process like RemoteASREngine,
    # so it presents the same surface to slot into _switch_asr_engine.

    @property
    def status(self) -> str:
        return "ready"

    @property
    def pid(self):
        # pid=None makes the memory monitor and worker-recycle logic skip it.
        return None

    def shutdown(self):
        self.unload()

    def terminate(self):
        self.unload()

    def set_input_padding(self, pad_seconds):
        pass

    def set_language(self, language: str):
        old = self.language
        self.language = language if language != "auto" else None
        log.info(f"Haas ASR language: {old} -> {self.language}")

    def to_device(self, device: str):
        # The backend handles device management.
        log.info(f"Haas ASR ignores device change (backend-side): {device}")
        return True

    def unload(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                log.info("Stopping speech.cpp backend")
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        self._client.close()
        log.info("Haas ASR backend connection closed")

    def transcribe(
        self, audio: np.ndarray, word_timestamps: bool = False, **kwargs
    ) -> dict | None:
        """Send audio to the speech.cpp backend and return transcription."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # float32 [-1, 1] -> int16 PCM (OpenAI audio API expects 16kHz mono PCM).
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        payload = {"file": ("audio.wav", _encode_pcm_wav(pcm), "audio/wav")}
        if self.language and self.language != "auto":
            payload["language"] = self.language
        # Use diarized_json when diarization is enabled so per-speaker turns are
        # returned; fall back to plain json otherwise. `chunking_strategy` is
        # required by the backend whenever `response_format=diarized_json`.
        if self._diarize:
            payload["response_format"] = "diarized_json"
            payload["chunking_strategy"] = "auto"
        else:
            payload["response_format"] = "json"

        try:
            r = self._client.post(self._url, files=payload)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.error(f"Haas ASR request failed: {e}")
            return None

        turns = data.get("segments")
        if turns:
            # `segments` is a list of per-speaker turns, each with a dominant
            # `speaker` label (A, B, ...) and its `text`. Concatenate them into a
            # single transcript, prefixing each turn with its speaker label.
            parts = []
            labels = []
            for turn in turns:
                t = (turn.get("text") or "").strip()
                if t:
                    speaker = turn.get("speaker", "") or ""
                    if speaker:
                        parts.append(f"[{speaker}] {t}")
                        labels.append(speaker)
                    else:
                        parts.append(t)
            text = " ".join(p for p in parts if p).strip()
            if not text:
                return None
            log.debug(f"Haas ASR diarized -> {labels}")
        else:
            text = (data.get("text") or "").strip()
            if not text:
                return None
            detected_lang = (
                data.get("language") or data.get("detected_language") or "ja"
            )
            log.debug(f"Haas ASR -> {text[:60]} ({detected_lang})")
            return {
                "text": text,
                "language": detected_lang,
                "language_name": detected_lang,
            }

        # When the backend returns only diarized segments without a top-level
        # language, keep the language the client was asked to use.
        detected_lang = self.language or "ja"
        log.debug(f"Haas ASR -> {text[:60]} ({detected_lang})")
        return {
            "text": text,
            "language": detected_lang,
            "language_name": detected_lang,
        }


def _encode_pcm_wav(pcm: np.ndarray) -> bytes:
    """Encode int16 PCM to a minimal WAV container in memory."""
    n_channels = 1
    sampwidth = 2
    rate = 16000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(n_channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
