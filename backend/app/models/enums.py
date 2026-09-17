from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    LOADING_MODEL = "LOADING_MODEL"
    PROCESSING_AUDIO = "PROCESSING_AUDIO"
    GENERATING = "GENERATING"
    POST_PROCESSING = "POST_PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


class ReferenceStatus(StrEnum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    ANALYZED = "ANALYZED"
    TRANSCRIBED = "TRANSCRIBED"
    READY = "READY"
    FAILED = "FAILED"


class TranscriptSource(StrEnum):
    ASR = "asr"
    MANUAL = "manual"
