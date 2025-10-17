from grpclib.server import Stream
from fastapi import HTTPException
from com.beyoncloud.common.constants import HTTPStatusCodes
import time
from datetime import datetime
from com.beyoncloud.grpc.protos.dps_ocr.dps_ocr_to_infrence_service_grpc import DpsOcrInfServiceBase
from com.beyoncloud.grpc.protos.dps_ocr import dps_ocr_to_infrence_service_pb2 as pb
from com.beyoncloud.schemas.rag_reqres_data_model import StructureRespProcessRequest
from com.beyoncloud.grpc.server.dps_ocr.utils.grpc_dps_ocr_util import get_grpc_dpsocr_inf_resp_datamodel
from com.beyoncloud.services.rag_proc_service import InfrenceService
import com.beyoncloud.config.settings.env_config as config

import logging

logger = logging.getLogger(__name__)

class DpsOcrInfService(DpsOcrInfServiceBase):
    async def process(self, stream: Stream[pb.DpsOcrToInferenceRequest, pb.DpsOcrToInferenceResponse]) -> None:
        request = await stream.recv_message()
        logger.info(f"Received request: {request}")

        if not request:
            logger.warning("Received empty or null request.")
            return

        structure_resp_process_request = request or pb.RagToInfrenceRequest

        try:
            infrence_service = InfrenceService()
            if config.TEMP_FLOW_YN == "Y":
                structure_resp_process_response = await infrence_service.temp_structure_resp_process(structure_resp_process_request)
            else:
                structure_resp_process_response = await infrence_service.inf_structure_resp_process(structure_resp_process_request)

            logger.info(f"Processed response: {structure_resp_process_response}")
            grpc_response = get_grpc_dpsocr_inf_resp_datamodel(structure_resp_process_response)
            logger.info(f"Sending response: {grpc_response}")
            await stream.send_message(grpc_response)
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
