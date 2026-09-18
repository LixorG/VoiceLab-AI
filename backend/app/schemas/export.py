from pydantic import BaseModel


class ExportFormatInfo(BaseModel):
    id: str
    label: str
    available: bool
    reason: str | None = None
