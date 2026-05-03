from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any


class ChatbotQueryInput(BaseModel):
    query: str
    text: Optional[str] = Field(default=None, alias="query")
    chat_history: Optional[str] = None
    routing: Optional[bool] = None

    @model_validator(mode="after")
    def normalize_query(self):
        if not self.text:
            self.text = self.query
        return self


class ChatbotQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]
    answer: Optional[str] = None
    response: Optional[str] = None
    sources: Optional[List[Any]] = None
    source_documents: Optional[List[Any]] = None
    route: Optional[str] = None
    routing_enabled: Optional[bool] = None
    answer_mode: Optional[str] = None
    graph_target: Optional[str] = None
    graph_data: Optional[Dict[str, Any]] = None
    debug_context: Optional[Dict[str, Any]] = None
