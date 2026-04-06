from pathlib import Path

import numpy as np
import soundfile as sf


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_2d(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio[:, None]
    return audio


def get_wav_duration_seconds(wav_path: str | Path) -> float:
    info = sf.info(str(wav_path))
    if not info.samplerate:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def _format_lrc_timestamp(seconds: float) -> str:
    total = max(0.0, float(seconds))
    minutes = int(total // 60)
    remain = total - minutes * 60
    return f"{minutes:02d}:{remain:05.2f}"


def build_lrc(
    wav_paths: list[str | Path],
    lyric_lines: list[str],
    output_path: str | Path,
    silence_ms: int = 350,
) -> str:
    if len(wav_paths) != len(lyric_lines):
        raise ValueError("wav_paths 和 lyric_lines 数量必须一致")

    elapsed = 0.0
    lrc_lines = []
    silence_seconds = max(0, silence_ms) / 1000.0

    for idx, wav_path in enumerate(wav_paths):
        text = str(lyric_lines[idx]).strip()
        if text:
            lrc_lines.append(f"[{_format_lrc_timestamp(elapsed)}]{text}")
        elapsed += get_wav_duration_seconds(wav_path)
        if idx != len(wav_paths) - 1:
            elapsed += silence_seconds

    Path(output_path).write_text("\n".join(lrc_lines) + "\n", encoding="utf-8")
    return str(output_path)


def join_wavs(wav_paths: list[str | Path], output_path: str | Path, silence_ms: int = 350) -> str:
    if not wav_paths:
        raise ValueError("wav_paths 不能为空")

    audios = []
    sample_rate = None
    channels = None

    for wav_path in wav_paths:
        audio, sr = sf.read(str(wav_path), always_2d=False)
        audio = _to_2d(audio)

        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"采样率不一致: {wav_path} 是 {sr}, 预期 {sample_rate}")

        if channels is None:
            channels = audio.shape[1]
        elif audio.shape[1] != channels:
            raise ValueError(f"声道数不一致: {wav_path} 是 {audio.shape[1]}, 预期 {channels}")

        audios.append(audio.astype(np.float32))

    silence_samples = int(sample_rate * silence_ms / 1000)
    silence = np.zeros((silence_samples, channels), dtype=np.float32)

    merged_parts = []
    for idx, audio in enumerate(audios):
        merged_parts.append(audio)
        if idx != len(audios) - 1:
            merged_parts.append(silence)

    merged = np.concatenate(merged_parts, axis=0)
    sf.write(str(output_path), merged, sample_rate)
    return str(output_path)
