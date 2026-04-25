from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field


class Segment(BaseModel):
    speaker: str = Field(default="旁白")
    text: str
    emotion: str = Field(default="neutral")
    style: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    voice_engine: Optional[str] = None
    voice_name: Optional[str] = None
    voice_locale: Optional[str] = None
    cfg_value: Optional[float] = None
    inference_timesteps: Optional[int] = None


class RoleProfile(BaseModel):
    speaker: str
    style: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    voice_engine: Optional[str] = None
    voice_name: Optional[str] = None
    voice_locale: Optional[str] = None
    cfg_value: Optional[float] = None
    inference_timesteps: Optional[int] = None


class TTSRequest(BaseModel):
    text: str
    instruct: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    voice_engine: Optional[str] = None
    voice_name: Optional[str] = None
    voice_locale: Optional[str] = None
    cfg_value: Optional[float] = None
    inference_timesteps: Optional[int] = None
    output_name: str = "result"
    inference_device: Optional[str] = None
    backend: Optional["AudioBackendConfig"] = None


class NarrateRequest(BaseModel):
    segments: List[Segment]
    silence_ms: int = 350
    output_name: str = "narration"
    role_profiles: Optional[Dict[str, RoleProfile]] = None
    inference_device: Optional[str] = None
    backend: Optional["AudioBackendConfig"] = None


class AnalysisContext(BaseModel):
    """跨块分析上下文状态，在同一章节的连续 chunk 间传递。"""
    known_characters: List[str] = Field(default_factory=list)
    character_aliases: Dict[str, str] = Field(default_factory=dict)  # alias → canonical
    last_context_summary: str = ""
    last_active_speakers: List[str] = Field(default_factory=list)
    chunk_index: int = 0


class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model: str
    local_model_path: Optional[str] = None
    local_runtime: Optional[str] = None
    local_engine: Optional[str] = None
    local_device: Optional[str] = None
    local_load_profile: Optional[str] = None
    local_ctx_tokens: Optional[int] = None
    local_gpu_layers: Optional[int] = None
    local_threads: Optional[int] = None
    local_batch_size: Optional[int] = None
    temperature: float = 0.2
    max_tokens: int = 2000
    compatibility_mode: str = "strict_json"
    chapter_sample_chars: Optional[int] = None
    chapter_prompt: Optional[str] = None
    chapter_regex_override: Optional[str] = None
    segment_target_chars: Optional[int] = None
    analysis_combo: Optional[str] = None
    system_prompt: Optional[str] = None
    segment_prompt: Optional[str] = None
    optimize_prompt: Optional[str] = None
    segment_optimize_prompt: Optional[str] = None
    optimize_analyze_prompt: Optional[str] = None
    speaker_verification_prompt: Optional[str] = None
    alias_resolution_prompt: Optional[str] = None
    workers: int = 5
    # ── 上下文感知分析控制 ─────────────────────────────────────────────────
    # "off"     保持原来的纯并发模式，无跨块上下文传递（向后兼容）
    # "chapter" 章节内串行传递上下文，章节间并行（推荐默认）
    # "full"    全文串行传递上下文（最高质量，最慢）
    context_mode: str = "chapter"
    enable_speaker_verification: bool = False   # 二轮 speaker 复核（额外 LLM 调用）
    enable_character_alias_merge: bool = True    # 别名归并归一化


class ParseRequest(BaseModel):
    text: str
    llm: LLMConfig


class TextChunk(BaseModel):
    title: str = "全文"
    content: str


class SegmentPlanRequest(BaseModel):
    text: str
    llm: LLMConfig


class ChunkOptimizeRequest(BaseModel):
    chunks: List[TextChunk]
    llm: LLMConfig


class AnalyzeChunksRequest(BaseModel):
    chunks: List[TextChunk]
    llm: LLMConfig


class AutoNarrateRequest(BaseModel):
    text: str
    llm: LLMConfig
    silence_ms: int = 350
    output_name: str = "auto_narration"
    role_profiles: Optional[Dict[str, RoleProfile]] = None
    inference_device: Optional[str] = None
    backend: Optional["AudioBackendConfig"] = None


class LyricLine(BaseModel):
    speaker: Optional[str] = None
    text: str


class MergeRequest(BaseModel):
    wav_files: List[str]
    output_name: str = "merged"
    silence_ms: int = 350
    lyrics: Optional[List[LyricLine]] = None


class ImportSegmentsRequest(BaseModel):
    content: str


class ConnectivityTestRequest(BaseModel):
    llm: LLMConfig


class UploadRefAudioRequest(BaseModel):
    filename: str
    content_base64: str


class TranscribeRefAudioRequest(BaseModel):
    ref_audio: str
    backend: Optional["AudioBackendConfig"] = None


class AudioBackendConfig(BaseModel):
    mode: str = "local"
    remote_base_url: Optional[str] = None
    remote_api_key: Optional[str] = None
    inference_device: Optional[str] = None
    local_audio_engine: Optional[str] = None
    local_audio_model_path: Optional[str] = None
    local_asr_model_path: Optional[str] = None

class ModelLoadRequest(BaseModel):
    backend: Optional[AudioBackendConfig] = None


class VoiceLibraryImportItem(BaseModel):
    filename: str
    content_base64: str


class BatchImportVoiceLibraryRequest(BaseModel):
    files: List[VoiceLibraryImportItem]
    transcribe_ref_text: bool = False
    backend: Optional[AudioBackendConfig] = None


class ListLLMModelsRequest(BaseModel):
    base_url: str
    api_key: Optional[str] = None


class ScanLocalModelsRequest(BaseModel):
    root_path: str


class VerifySpeakersRequest(BaseModel):
    segments: List[Dict[str, Any]]
    llm: LLMConfig
    context: Optional[AnalysisContext] = None


class MergeAliasesRequest(BaseModel):
    segments: List[Dict[str, Any]]
    llm: LLMConfig


TTSRequest.model_rebuild()
NarrateRequest.model_rebuild()
AutoNarrateRequest.model_rebuild()
TranscribeRefAudioRequest.model_rebuild()
