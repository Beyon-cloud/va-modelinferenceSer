from grpclib.server import Stream
import time
from datetime import datetime
from com.beyoncloud.grpc.protos.infrence_model.rag_to_infrence_service_grpc import RAGInfServiceBase
from com.beyoncloud.grpc.protos.infrence_model import rag_to_infrence_service_pb2 as pb
from com.beyoncloud.schemas.rag_reqres_data_model import RagReqDataModel, RagRespDataModel, SessionContext, DialogDetail, Metadata, UserInput, InputMetadata, EntityItem
from com.beyoncloud.grpc.server.infrence_model.utils.grpc_inf_util import getGrpcInfRespDataModel
from com.beyoncloud.services.rag_proc_service import rag_chat_process
from com.beyoncloud.processing.generation.generator import RagGeneratorProcess

import logging

logger = logging.getLogger(__name__)

class RAGInfService(RAGInfServiceBase):
    async def GetInfResponse(self, stream: Stream[pb.RagToInfrenceRequest, pb.RagToInfrenceResponse]) -> None:
        request = await stream.recv_message()
        logger.info(f"Received request: {request}")

        if not request:
            logger.warning("Received empty or null request.")
            return

        request_model = request.dms_request or pb.RagToInfrenceRequest.DMSRequest()
        search_result = request.search_result or []

        starttime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        start_time = time.time()

        rag_generator_process = RagGeneratorProcess()
        response = await rag_generator_process.generateAnswer(request_model, search_result)

        endtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        end_time = time.time()
        elapsed = end_time - start_time  # in seconds (float)
        print(f"Start time : {starttime} --> End time : {endtime} --> elapsed: {elapsed}")

        #response = await rag_chat_process(request_model)
        logger.info(f"Processed response: {response}")
        grpc_response = getGrpcInfRespDataModel(response)
        logger.info(f"Sending response: {grpc_response}")
        await stream.send_message(grpc_response)
