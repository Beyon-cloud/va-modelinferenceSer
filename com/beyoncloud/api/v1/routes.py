from fastapi import APIRouter
from com.beyoncloud.api.v1.rag_processing_routes import rag_process_router
from com.beyoncloud.api.v1.prompt_processing_routes import prompt_process_router


v1_router = APIRouter()

v1_router.include_router(
    rag_process_router, prefix="/rag_processing", tags=["RAG Processing Endpoints"]
)

v1_router.include_router(
    prompt_process_router, prefix="/generate_prompt", tags=["Prompt Generation Endpoints"]
)

