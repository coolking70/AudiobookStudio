from pathlib import Path
import re

import numpy as np
import soundfile as sf

DEFAULT_MAX_OUTPUT_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_SEGMENTS_PER_FILE = 750
PCM16_BYTES_PER_SAMPLE = 2
DEFAULT_TTS_CHARS_PER_SECOND = 3.8
DEFAULT_TTS_WORDS_PER_SECOND = 2.4
DEFAULT_TTS_MIN_DURATION_RATIO = 0.35
DEFAULT_TTS_MAX_DURATION_RATIO = 2.8
DEFAULT_TTS_MIN_ABSOLUTE_SECONDS = 0.45
DEFAULT_TTS_MAX_ABSOLUTE_SECONDS = 4.0
DEFAULT_TTS_MIN_UNITS_TO_VALIDATE = 2


class AudioDurationValidationError(RuntimeError):
    def __init__(self, message: str, details: dict[str, float | int | str | bool]):
        super().__init__(message)
        self.details = details


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_wav_duration_seconds(wav_path: str | Path) -> float:
    info = sf.info(str(wav_path))
    if not info.samplerate:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def count_tts_text_units(text: str) -> dict[str, int]:
    normalized = str(text or "")
    cjk_chars = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", normalized)
    latin_words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", normalized)
    punctuation_pauses = re.findall(r"[，。！？、；：,.!?;:]", normalized)
    return {
        "cjk_chars": len(cjk_chars),
        "latin_words": len(latin_words),
        "punctuation_pauses": len(punctuation_pauses),
        "units": len(cjk_chars) + len(latin_words),
    }


def estimate_tts_duration_bounds(text: str) -> dict[str, float | int | str | bool]:
    units = count_tts_text_units(text)
    if units["units"] < DEFAULT_TTS_MIN_UNITS_TO_VALIDATE:
        return {
            **units,
            "enabled": False,
            "estimated_seconds": 0.0,
            "min_seconds": 0.0,
            "max_seconds": 0.0,
            "reason": "text_too_short",
        }

    cjk_seconds = units["cjk_chars"] / DEFAULT_TTS_CHARS_PER_SECOND
    latin_seconds = units["latin_words"] / DEFAULT_TTS_WORDS_PER_SECOND
    pause_seconds = min(units["punctuation_pauses"] * 0.12, (cjk_seconds + latin_seconds) * 0.25)
    estimated = max(DEFAULT_TTS_MIN_ABSOLUTE_SECONDS, cjk_seconds + latin_seconds + pause_seconds)
    min_seconds = max(DEFAULT_TTS_MIN_ABSOLUTE_SECONDS, estimated * DEFAULT_TTS_MIN_DURATION_RATIO)
    max_seconds = max(DEFAULT_TTS_MAX_ABSOLUTE_SECONDS, estimated * DEFAULT_TTS_MAX_DURATION_RATIO + 2.0)
    return {
        **units,
        "enabled": True,
        "estimated_seconds": round(estimated, 3),
        "min_seconds": round(min_seconds, 3),
        "max_seconds": round(max_seconds, 3),
        "reason": "ok",
    }


def validate_audio_duration_for_text(
    wav_path: str | Path,
    text: str,
    *,
    raise_on_invalid: bool = True,
) -> dict[str, float | int | str | bool]:
    bounds = estimate_tts_duration_bounds(text)
    duration = get_wav_duration_seconds(wav_path)
    result = {
        **bounds,
        "duration_seconds": round(duration, 3),
        "path": str(wav_path),
        "valid": True,
    }
    if not bounds.get("enabled"):
        return result

    min_seconds = float(bounds["min_seconds"])
    max_seconds = float(bounds["max_seconds"])
    if duration < min_seconds or duration > max_seconds:
        result["valid"] = False
        message = (
            "生成音频时长异常："
            f"文本约 {bounds['units']} 字/词，预计 {bounds['estimated_seconds']} 秒，"
            f"允许范围 {min_seconds:.2f}-{max_seconds:.2f} 秒，实际 {duration:.2f} 秒。"
        )
        if raise_on_invalid:
            raise AudioDurationValidationError(message, result)
    return result


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


def _inspect_wav(wav_path: str | Path) -> dict:
    path = Path(wav_path)
    info = sf.info(str(path))
    return {
        "path": path,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
    }


def _estimate_pcm16_bytes(frames: int, channels: int) -> int:
    return max(0, int(frames)) * max(1, int(channels)) * PCM16_BYTES_PER_SAMPLE


def split_wav_groups(
    wav_paths: list[str | Path],
    silence_ms: int = 350,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    max_segments_per_file: int = DEFAULT_MAX_SEGMENTS_PER_FILE,
) -> list[list[dict]]:
    if not wav_paths:
        raise ValueError("wav_paths 不能为空")

    inspected = [_inspect_wav(wav_path) for wav_path in wav_paths]
    sample_rate = inspected[0]["sample_rate"]
    channels = inspected[0]["channels"]
    silence_frames = int(sample_rate * max(0, silence_ms) / 1000)
    silence_bytes = _estimate_pcm16_bytes(silence_frames, channels)

    groups: list[list[dict]] = []
    current_group: list[dict] = []
    current_bytes = 0

    for item in inspected:
        if item["sample_rate"] != sample_rate:
            raise ValueError(f"采样率不一致: {item['path']} 是 {item['sample_rate']}, 预期 {sample_rate}")
        if item["channels"] != channels:
            raise ValueError(f"声道数不一致: {item['path']} 是 {item['channels']}, 预期 {channels}")

        item_bytes = _estimate_pcm16_bytes(item["frames"], channels)
        projected_bytes = current_bytes + item_bytes + (silence_bytes if current_group else 0)
        should_split = (
            bool(current_group)
            and (
                len(current_group) >= max_segments_per_file
                or (max_output_bytes > 0 and projected_bytes > max_output_bytes)
            )
        )
        if should_split:
            groups.append(current_group)
            current_group = []
            current_bytes = 0
            projected_bytes = item_bytes

        current_group.append(item)
        current_bytes = projected_bytes

    if current_group:
        groups.append(current_group)

    return groups


def _write_wav_group(group: list[dict], output_path: str | Path, silence_ms: int = 350) -> dict:
    if not group:
        raise ValueError("group 不能为空")

    output_path = Path(output_path)
    sample_rate = int(group[0]["sample_rate"])
    channels = int(group[0]["channels"])
    silence_samples = int(sample_rate * max(0, silence_ms) / 1000)
    silence = np.zeros((silence_samples, channels), dtype=np.float32) if silence_samples > 0 else None
    total_frames = 0

    with sf.SoundFile(
        str(output_path),
        mode="w",
        samplerate=sample_rate,
        channels=channels,
        format="WAV",
        subtype="PCM_16",
    ) as writer:
        for idx, item in enumerate(group):
            with sf.SoundFile(str(item["path"]), mode="r") as reader:
                if reader.samplerate != sample_rate:
                    raise ValueError(f"采样率不一致: {item['path']} 是 {reader.samplerate}, 预期 {sample_rate}")
                if reader.channels != channels:
                    raise ValueError(f"声道数不一致: {item['path']} 是 {reader.channels}, 预期 {channels}")

                while True:
                    chunk = reader.read(65536, dtype="float32", always_2d=True)
                    if len(chunk) == 0:
                        break
                    writer.write(chunk)
                    total_frames += len(chunk)

            if silence is not None and idx != len(group) - 1:
                writer.write(silence)
                total_frames += len(silence)

    return {
        "path": str(output_path),
        "duration_seconds": float(total_frames) / float(sample_rate) if sample_rate else 0.0,
        "size_bytes": output_path.stat().st_size,
        "sample_rate": sample_rate,
        "channels": channels,
    }


def join_wavs_auto(
    wav_paths: list[str | Path],
    output_path: str | Path,
    silence_ms: int = 350,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    max_segments_per_file: int = DEFAULT_MAX_SEGMENTS_PER_FILE,
) -> dict:
    if not wav_paths:
        raise ValueError("wav_paths 不能为空")

    output_path = Path(output_path)
    groups = split_wav_groups(
        wav_paths,
        silence_ms=silence_ms,
        max_output_bytes=max_output_bytes,
        max_segments_per_file=max_segments_per_file,
    )

    part_results = []
    cursor = 0
    split_applied = len(groups) > 1
    manifest_path = None

    if split_applied:
        parts_dir = ensure_dir(output_path.parent / f"{output_path.stem}_parts")
        manifest_lines = ["index\tfile\tsegments\tstart_index\tend_index\tduration_seconds\tsize_bytes"]
        for idx, group in enumerate(groups, start=1):
            part_path = parts_dir / f"{output_path.stem}_part_{idx:03d}.wav"
            write_result = _write_wav_group(group, part_path, silence_ms=silence_ms)
            start_index = cursor + 1
            end_index = cursor + len(group)
            cursor = end_index
            part_entry = {
                "index": idx,
                "path": write_result["path"],
                "segment_count": len(group),
                "start_index": start_index,
                "end_index": end_index,
                "duration_seconds": write_result["duration_seconds"],
                "size_bytes": write_result["size_bytes"],
                "sample_rate": write_result["sample_rate"],
                "channels": write_result["channels"],
            }
            part_results.append(part_entry)
            manifest_lines.append(
                f"{idx}\t{Path(write_result['path']).name}\t{len(group)}\t{start_index}\t{end_index}\t"
                f"{write_result['duration_seconds']:.2f}\t{write_result['size_bytes']}"
            )

        manifest_path = parts_dir / f"{output_path.stem}_manifest.tsv"
        manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        primary_path = part_results[0]["path"]
    else:
        write_result = _write_wav_group(groups[0], output_path, silence_ms=silence_ms)
        part_results.append(
            {
                "index": 1,
                "path": write_result["path"],
                "segment_count": len(groups[0]),
                "start_index": 1,
                "end_index": len(groups[0]),
                "duration_seconds": write_result["duration_seconds"],
                "size_bytes": write_result["size_bytes"],
                "sample_rate": write_result["sample_rate"],
                "channels": write_result["channels"],
            }
        )
        primary_path = write_result["path"]

    return {
        "split_applied": split_applied,
        "primary_path": primary_path,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "parts": part_results,
    }


def join_wavs(wav_paths: list[str | Path], output_path: str | Path, silence_ms: int = 350) -> str:
    return str(join_wavs_auto(wav_paths, output_path, silence_ms=silence_ms)["primary_path"])
