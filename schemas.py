from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class Segment(BaseModel):
    speaker: str = Field(default="旁白")
    text: str
    emotion: str = Field(default="neutral")
    style: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None


class RoleProfile(BaseModel):
    speaker: str
    style: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None


class TTSRequest(BaseModel):
    text: str
    instruct: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
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


class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 2000
    compatibility_mode: str = "strict_json"
    system_prompt: Optional[str] = None
    workers: int = 5


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


class AudioBackendConfig(BaseModel):
    mode: str = "local"
    remote_base_url: Optional[str] = None
    remote_api_key: Optional[str] = None
    inference_device: Optional[str] = None

class ModelLoadRequest(BaseModel):
    backend: Optional[AudioBackendConfig] = None


class VoiceLibraryImportItem(BaseModel):
    filename: str
    content_base64: str


class BatchImportVoiceLibraryRequest(BaseModel):
    files: List[VoiceLibraryImportItem]


TTSRequest.model_rebuild()
NarrateRequest.model_rebuild()
AutoNarrateRequest.model_rebuild()
