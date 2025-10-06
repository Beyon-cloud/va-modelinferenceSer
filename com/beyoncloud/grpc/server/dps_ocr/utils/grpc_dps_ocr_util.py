from typing import Any
from com.beyoncloud.grpc.protos.dps_ocr import dps_ocr_to_infrence_service_pb2 as pb
from com.beyoncloud.schemas.rag_reqres_data_model import StructureRespProcessResponse


def get_grpc_dpsocr_inf_resp_datamodel(structure_resp_process_response: StructureRespProcessResponse) -> pb.DpsOcrToInferenceResponse:

    grpc_dpsocr_inf_resp_datamodel = pb.DpsOcrToInferenceResponse(
        request_reference_id = structure_resp_process_response.request_reference_id,
        document_batch_id = structure_resp_process_response.document_batch_id,
        organization_id = structure_resp_process_response.organization_id,
        domain_id = structure_resp_process_response.domain_id,
        user_id = structure_resp_process_response.user_id,
        desired_output_mode = structure_resp_process_response.desired_output_mode,
        desired_output_format = structure_resp_process_response.desired_output_format,
        source_file_path = structure_resp_process_response.source_file_path,
        source_lang = structure_resp_process_response.source_lang,
        document_type = structure_resp_process_response.document_type,
        output_filename = structure_resp_process_response.output_filename,
        ocr_result_storage_path = structure_resp_process_response.ocr_result_storage_path,
        inference_result_storage_path = structure_resp_process_response.inference_result_storage_path,
        ocr_result_file_path = structure_resp_process_response.ocr_result_file_path,
        inference_result_file_path = structure_resp_process_response.inference_result_file_path,
        status = structure_resp_process_response.status,
        message = structure_resp_process_response.message
    )

    return grpc_dpsocr_inf_resp_datamodel