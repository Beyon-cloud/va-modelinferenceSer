from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class DynSelectEntity(BaseModel):
    column_names: List[str] = None
    conditions: Dict[str, Any] = None
    table_name: str = ""
    top_k: Optional[int] = None
    order_by: Dict[str, List[str]] = None

class SentenceEmbeddingsEntity(BaseModel):
    doc_id: str
    sentence_idx: int
    sentence: str
    embedding: str

class DocumentImagesEntity(BaseModel):
    doc_id: str
    image_idx: int
    image_path: str
    embedding: str

