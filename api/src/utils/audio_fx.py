import subprocess, tempfile, os
from typing import Optional
import numpy as np

def _ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg","-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        raise RuntimeError("ffmpeg is required for prosody/emphasis FX but was not found in PATH") from e

def _chain_atempo(val: float) -> str:
    if val <= 0: val = 1.0
    parts = []
    while val < 0.5 or val > 2.0:
        if val < 0.5: parts.append("0.5"); val /= 0.5
        else:         parts.append("2.0");  val /= 2.0
    parts.append(f"{val:.6f}")
    return ",".join([f"atempo={p}" for p in parts])

def apply_fx_np(audio: np.ndarray, sample_rate: int, tempo: float = 1.0, pitch_cents: int = 0, gain_db: float = 0.0) -> np.ndarray:
    _ensure_ffmpeg()
    if audio.dtype != np.int16:
        audio = audio.astype(np.int16)
    if abs(tempo-1.0)<1e-6 and pitch_cents==0 and abs(gain_db)<1e-6:
        return audio
    with tempfile.TemporaryDirectory() as td:
        in_wav  = os.path.join(td,"in.wav")
        out_wav = os.path.join(td,"out.wav")

        import soundfile as sf
        sf.write(in_wav, audio, samplerate=sample_rate, subtype="PCM_16")

        R = 2.0 ** (pitch_cents/1200.0) if pitch_cents else 1.0
        atempo_total = tempo * (1.0 / R)

        filters = []
        if pitch_cents:
            filters += [f"asetrate={int(sample_rate*R)}", f"aresample={sample_rate}"]
        if atempo_total != 1.0:
            filters.append(_chain_atempo(atempo_total))
        if abs(gain_db) > 1e-6:
            filters.append(f"volume={gain_db:.2f}dB")
        afilter = ",".join(filters) if filters else "anull"

        subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",in_wav,"-af",afilter,out_wav], check=True)

        data, sr = sf.read(out_wav, dtype="int16")
        if data.ndim > 1: data = data[:,0]
        return data

def gen_silence_np(duration_s: float, sample_rate: int) -> np.ndarray:
    import numpy as np
    n = max(0, int(round(duration_s * sample_rate)))
    return np.zeros(n, dtype=np.int16)
