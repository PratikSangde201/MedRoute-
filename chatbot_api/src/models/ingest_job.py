from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class IngestJob(BaseModel):
    job_id: str
    filename: str
    content_type: str
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    classification: Optional[Dict[str, Any]] = None
    extracted_text: Optional[str] = None
    structured: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None
    dedup_result: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    entities_extracted: Optional[int] = None
    merge_decisions: Optional[list[Dict[str, Any]]] = None
    inserted: Optional[Dict[str, int]] = None
    error_message: Optional[str] = None

    def touch(self) -> None:
        self.updated_at = datetime.utcnow()
