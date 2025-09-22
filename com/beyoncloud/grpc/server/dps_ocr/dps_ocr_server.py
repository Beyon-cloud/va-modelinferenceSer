from grpclib.server import Stream
import time
from datetime import datetime
from com.beyoncloud.grpc.protos.dps_ocr.dps_ocr_to_infrence_service_grpc import DpsOcrInfServiceBase
from com.beyoncloud.grpc.protos.dps_ocr import dps_ocr_to_infrence_service_pb2 as pb
from com.beyoncloud.schemas.rag_reqres_data_model import StructureRespProcessRequest
from com.beyoncloud.grpc.server.dps_ocr.utils.grpc_dps_ocr_util import getGrpcDpsOcrInfRespDataModel
from com.beyoncloud.services.rag_proc_service import InfrenceService

import logging

logger = logging.getLogger(__name__)

class DpsOcrInfService(DpsOcrInfServiceBase):
    async def GetDpsOcrInfResponse(self, stream: Stream[pb.DpsOcrToInferenceRequest, pb.DpsOcrToInferenceResponse]) -> None:
        request = await stream.recv_message()
        logger.info(f"Received request: {request}")

        if not request:
            logger.warning("Received empty or null request.")
            return

        structure_resp_process_request = request or pb.RagToInfrenceRequest

        infrence_service = InfrenceService()
        structure_resp_process_response = await infrence_service.inf_structure_resp_process(structure_resp_process_request)

        logger.info(f"Processed response: {structure_resp_process_response}")
        grpc_response = getGrpcDpsOcrInfRespDataModel(structure_resp_process_response)
        logger.info(f"Sending response: {grpc_response}")
        await stream.send_message(grpc_response)
