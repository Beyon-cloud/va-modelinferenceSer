from com.beyoncloud.services.schema_prompt_service import SchemaPromptService
from com.beyoncloud.services.rag_proc_service import InfrenceService

def get_schema_prompt_service() -> SchemaPromptService:
    return SchemaPromptService()

def get_inf_service() -> InfrenceService:
    return InfrenceService()