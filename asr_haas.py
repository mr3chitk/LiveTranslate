import io
import json
import logging
import shutil
import subprocess
import tempfile
import time
import wave
import zipfile
from pathlib import Path, PurePosixPath

import httpx
import numpy as np

log = logging.getLogger("LiveTranslate.ASR.Haas")

use_crispasr = True # CONFIG
if(use_crispasr):
    _BIN_FOLDER         = Path(__file__).resolve().parent / "crispasr"
    _BIN_EXE            = "crispasr.exe"
    _BIN_GITHUB_API     = "https://api.github.com/repos/CrispStrobe/CrispASR/releases/latest"
    _BIN_GITHUB_PATTERN = "crispasr-windows-x86_64-vulkan.zip"
    _DEFAULT_MODEL      = "haas"
    _DEFAULT_DIARIZE    = "pyannote" # foxnose / pyannote 
else:
    _BIN_FOLDER         = Path(__file__).resolve().parent / "speech.cpp"
    _BIN_EXE            = "speech.exe"
    _BIN_GITHUB_API     = "https://api.github.com/repos/nyosegawa/speech.cpp/releases/latest"
    _BIN_GITHUB_PATTERN = "-windows-x64-vulkan.zip"
    _DEFAULT_MODEL      = "haas-ja-v1.0"
    _DEFAULT_DIARIZE    = "nemotron-3-diarization"

_DEFAULT_PORT = 8766
_DEFAULT_URL = f"http://127.0.0.1:{_DEFAULT_PORT}"


def _ensure_haas_bin(force_dl: bool = False) -> Path:
    """Make sure the selected backend binaries exist.
    """
    exe = _BIN_FOLDER / _BIN_EXE
    if exe.exists() and not force_dl:
        return exe

    _BIN_FOLDER.mkdir(parents=True, exist_ok=True)
    log.info(f"Haas: downloading latest release into {_BIN_FOLDER}")

    tmp_dir = None
    try:
        from model_manager import apply_cache_env

        apply_cache_env()
        import urllib.request

        tmp_dir = tempfile.mkdtemp(prefix="haas-bin-dl-")
        zip_path = Path(tmp_dir) / "release.zip"

        # Fetch the release asset URL (skips the JSON body).
        req = urllib.request.Request(
            _BIN_GITHUB_API, headers={"Accept": "application/vnd.github+json"}
        )
        download_url = None
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for asset in data.get("assets", []):
                if _BIN_GITHUB_PATTERN in asset["browser_download_url"]:
                    download_url = asset["browser_download_url"]
                    break
        if not download_url:
            raise FileNotFoundError(
                f"Release asset '{_BIN_GITHUB_PATTERN}' not found in latest release"
            )

        with urllib.request.urlopen(download_url, timeout=600) as src, open(
            zip_path, "wb"
        ) as dst:
            shutil.copyfileobj(src, dst)

        exe_parent = exe.parent
        if zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                exe_name = next(
                    (name for name in names if PurePosixPath(name).name.lower() == _BIN_EXE),
                    None,
                )
                if exe_name is None:
                    raise FileNotFoundError(f"{_BIN_EXE} was not found in the ZIP file")
                folder = PurePosixPath(exe_name).parent
                for name in names:
                    member = PurePosixPath(name)
                    # Select files and subfolders inside that folder, excluding the folder itself.
                    if member == folder or folder not in member.parents:
                        continue
                    # Keep only the path relative to the folder containing xyz.exe.
                    relative = member.relative_to(folder)
                    if not relative.parts:
                        continue
                    # Prevent paths in the ZIP from escaping the destination directory.
                    target = (exe_parent / Path(*relative.parts)).resolve()
                    if not target.is_relative_to(exe_parent.resolve()):
                        raise ValueError(f"Unsafe path in ZIP: {name}")
                    if name.endswith("/"):
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(name) as source, target.open("wb") as output:
                            output.write(source.read())
        else:
            raise RuntimeError(f"Unsupported archive format: {zip_path.suffix}")

        log.info(f"Extracted to {exe}")
        return exe
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class HaasASREngine:
    def __init__(
        self,
        timeout=30.0,
        diarize=False,
    ):
        self._url = _DEFAULT_URL.rstrip("/") + "/v1/audio/transcriptions"
        self._health_url = _DEFAULT_URL.rstrip("/") + "/health"
        self._diarize = diarize
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
        exe = _ensure_haas_bin()
        log.info(f"Starting backend: {exe}")
        if _BIN_EXE == "speech.exe":
            serve_args = [
                str(exe),
                "serve",
                str(_DEFAULT_MODEL),
                "--host",
                "127.0.0.1",
                "--port",
                str(_DEFAULT_PORT),
            ]
            if self._diarize:
                serve_args += ["silero-vad", str(_DEFAULT_DIARIZE)]
                log.info(f"Speaker diarization: {_DEFAULT_DIARIZE}.")
        elif _BIN_EXE == "crispasr.exe":
            serve_args = [
                str(exe),
                "--server",
                "--backend",
                str(_DEFAULT_MODEL),
                "-m",
                "auto",
                "--host",
                "127.0.0.1",
                "--port",
                str(_DEFAULT_PORT),
            ]
            if self._diarize:
                serve_args += ["--diarize", "--diarize-method", str(_DEFAULT_DIARIZE), "--diarize-embedder", "auto"]
                log.info(f"Speaker diarization: {_DEFAULT_DIARIZE}.")
        else:
            raise RuntimeError(f"Unsupported _BIN_EXE: {_BIN_EXE}")
        self._proc = subprocess.Popen(
            serve_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_ready()

    def _wait_ready(self, timeout=1800.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise ConnectionError(
                    "Haas backend exited during startup "
                    f"(code={self._proc.returncode})"
                )
            try:
                r = self._client.get(self._health_url, timeout=2.0)
                if r.status_code == 200:
                    log.info("Haas backend is ready")
                    return
            except Exception:
                time.sleep(1.0)
        raise ConnectionError(
            f"Haas backend did not become ready at {self._health_url} within "
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
                log.info("Stopping haas backend")
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        self._client.close()
        log.info("Haas backend connection closed")

    def reload_speech(self):
        """Force-redownload and restart the backend (update flow).
        """
        self.unload()
        log.info("Haas: force re-download binaries for update")
        _ensure_haas_bin(force_dl=True)
        self._client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=5.0), trust_env=False
        )
        self._start_backend()

    def transcribe(
        self, audio: np.ndarray, word_timestamps: bool = False, **kwargs
    ) -> dict | None:
        """Send audio to the backend and return transcription."""
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
            if _BIN_EXE == "speech.exe":
                payload["response_format"] = "diarized_json"
                payload["chunking_strategy"] = "auto"
            elif _BIN_EXE == "crispasr.exe":
                payload["response_format"] = "diarized_json"
                payload["diarize"] = "true"
                payload["diarize_method"] = str(_DEFAULT_DIARIZE)
                payload["diarize-embedder"] = "auto"
            else:
                raise RuntimeError(f"Unsupported _BIN_EXE: {_BIN_EXE}")
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
        # print(data)
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
