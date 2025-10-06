from datetime import datetime
import logging
from fastapi import APIRouter,HTTPException, Depends
from com.beyoncloud.common.constants import HTTPStatusCodes
from com.beyoncloud.api.v1.dependencies import get_inf_service
from com.beyoncloud.services.rag_proc_service import InfrenceService
from com.beyoncloud.services.rag_proc_service import rag_chat_process
from com.beyoncloud.schemas.rag_reqres_data_model import (
    RagReqDataModel, 
    StructureRespProcessRequest,
    StructureRespProcessResponse
)
import com.beyoncloud.config.settings.env_config as config

logger = logging.getLogger(__name__)

rag_process_router = APIRouter()

@rag_process_router.post("/process/")
async def rag_process(rag_req_data_model: RagReqDataModel):
    starttime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    response = await rag_chat_process(rag_req_data_model)
    
    endtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logger.info("Process taking time from '"+starttime+"' to '"+endtime+"'")
    return response

@rag_process_router.post("/structure_resp_process/")
async def structure_resp_process(
    structure_resp_process_request: StructureRespProcessRequest,
    service: InfrenceService = Depends(get_inf_service),
) -> StructureRespProcessResponse:

    try:

        if config.TEMP_FLOW_YN == "Y":
            response = await service.temp_structure_resp_process(structure_resp_process_request)
        else:
            response = await service.inf_structure_resp_process(structure_resp_process_request)
        
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

