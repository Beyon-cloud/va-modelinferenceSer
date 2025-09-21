import logging
from fastapi import APIRouter,HTTPException, Depends

from com.beyoncloud.services.prompt_gen_service import generate_prompt_json
from com.beyoncloud.schemas.prompt_gen_reqres_datamodel import (
    SchemaPromptRequest,
    SchemaPromptResponse, 
    EntityPromptRequest, 
    EntityPromptResponse
)
from com.beyoncloud.services.schema_prompt_service import SchemaPromptService
from com.beyoncloud.api.v1.dependencies import get_schema_prompt_service
from com.beyoncloud.common.constants import HTTPStatusCodes

logger = logging.getLogger(__name__)

prompt_process_router = APIRouter()

@prompt_process_router.post("/generate/")
async def generate_prompt():
    response = await generate_prompt_json()
    return response


@prompt_process_router.post("/generate_schema/")
async def generate_schema(
    schema_prompt_request: SchemaPromptRequest,
    service: SchemaPromptService = Depends(get_schema_prompt_service),
) -> SchemaPromptResponse:
    """
    Submit a new schema generation process.

    Workflow:

    1) Validate and persist job metadata in DB.  
    2) Add background task for async job execution.  
    3) Return a task response containing task ID and status.  

    Args:
        schema_prompt_request (SchemaPromptRequest): Input data source details.
        service (SchemaPromptService): Injected service layer for task processing.

    Returns:
        SchemaPromptResponse: JSON response containing schema prompt metadata.

    Raises:
        HTTPException:
            400 - Bad Request (invalid payload or validation error)  
            500 - Internal Server Error (unexpected failure)
    """
    try:
        response = await service.generate_schema_prompt(schema_prompt_request)
        return response
    except ValueError as e:
        logger.warning(f"Validation failed: {e}")
        raise HTTPException(
            status_code=HTTPStatusCodes.BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Unexpected error in process_task {str(e)}")
        raise HTTPException(
            status_code=HTTPStatusCodes.INTERNAL_SERVER_ERROR,
            detail="Internal Server Error"
        )


@prompt_process_router.post("/generate_entity_prompt/")
async def generate_final_prompt(
    entity_prompt_request: EntityPromptRequest,
    service: SchemaPromptService = Depends(get_schema_prompt_service)
) -> EntityPromptResponse:
    """
    Submit a entity prompt generation process.

    Workflow:

    1) Validate and persist job metadata in DB.  
    2) Add background task for async job execution.  
    3) Return a task response containing task ID and status.  

    Args:
        schema_prompt_request (SchemaPromptRequest): Input data source details.
        service (SchemaPromptService): Injected service layer for task processing.

    Returns:
        SchemaPromptResponse: JSON response containing schema prompt metadata.

    Raises:
        HTTPException:
            400 - Bad Request (invalid payload or validation error)  
            500 - Internal Server Error (unexpected failure)
    """
    try:
        response = await service.generate_entity_prompt(entity_prompt_request)
        return response
    except ValueError as e:
        logger.warning(f"Validation failed: {e}")
        raise HTTPException(
            status_code=HTTPStatusCodes.BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Unexpected error in process_task {str(e)}")
        raise HTTPException(
            status_code=HTTPStatusCodes.INTERNAL_SERVER_ERROR,
            detail="Internal Server Error"
        )