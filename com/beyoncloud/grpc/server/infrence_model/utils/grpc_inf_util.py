from typing import Any
from com.beyoncloud.grpc.protos.infrence_model import rag_to_infrence_service_pb2 as pb
from com.beyoncloud.schemas.rag_reqres_data_model import RagRespDataModel


def get_grpc_inf_resp_datamodel(model_response: Any) -> pb.RagToInfrenceResponse:

    grpc_inf_resp_datamodel = pb.RagToInfrenceResponse(
        response=pb.RagToInfrenceResponse.InfResponse(
            text = model_response
        )
    )

    return grpc_inf_resp_datamodel