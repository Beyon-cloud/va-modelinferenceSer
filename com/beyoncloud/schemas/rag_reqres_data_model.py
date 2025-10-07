from typing import List, Optional, Any, Self
from pydantic import BaseModel, Field, StrictInt, StrictStr
from datetime import datetime
from com.beyoncloud.common.constants import Numerical, CommonPatterns, CommonConstants

class InputMetadata(BaseModel):
    content_language: str

# --- User Input Block ---
class UserInput(BaseModel):
    type: str
    content: str
    translate_content: str
    media_url: str
    metadata: InputMetadata


# --- Metadata ---
class Metadata(BaseModel):
    preferred_language: str
    channel: str
    platform: str

# --- Entity Recognition ---
class EntityItem(BaseModel):
    entity_group: str
    word: str

class ResponseData(BaseModel):
    text: str

# --- Individual Dialog Entry ---
class DialogDetail(BaseModel):
    timestamp: str
    user_input: List[UserInput]
    metadata: Metadata
    intent: str
    response: ResponseData
    sentiment: str
    entities: List[EntityItem]

# --- Session Context Info ---
class SessionContext(BaseModel):
    created_at: str
    last_updated: str
    session_id: str
    user_id: str
    group_id: str
    profile_id: str
    org_id: int
    domain_id: str
    id: str
    dialogDetails: List[DialogDetail]

class ChatHistory(BaseModel):
    query: str
    response: str

# --- Root RAG Request/Response Model ---
class RagReqDataModel(BaseModel):
    session_id: str
    user_id: str
    group_id: str
    profile_id: str
    org_id: int
    domain_id: str
    response_type: str
    user_input: List[UserInput]
    metadata: Metadata
    intent: str
    sentiment: str
    entity: List[EntityItem]
    context: SessionContext

# --- Root RAG Request/Response Model ---
class RagRespDataModel(BaseModel):
    session_id: str
    user_id: str
    group_id: str
    profile_id: str
    org_id: int
    domain_id: str
    user_input: List[UserInput]
    metadata: Metadata
    response: ResponseData

# 
class SearchResult(BaseModel):
    sno: str
    search_result: str

class QryLogMetadata(BaseModel):
    domain: str
    startdate: str
    enddate: str

class EntityResponse(BaseModel):
    response: str
    status: str

# ======================= RagLogQryModel model Start =======================
class RagLogQryModel(BaseModel):
    orgId: int
    query: str
    response : str
    search_result_json: List[Any]
    time_elapsed: float
    metadata: Any

class RagLogQryModelBuilder:
    """
    Builder for RagLogQryModel model.
    Author: Jenson (26-09-2025)
    """
    def __init__(self):
        self.rag_log_qry_model = RagLogQryModel(
            orgId = Numerical.ZERO, 
            query = CommonPatterns.EMPTY_SPACE, 
            response = CommonPatterns.EMPTY_SPACE,
            search_result_json = [],
            time_elapsed = Numerical.ZERO,
            metadata =  {}
        )

    def with_org_id(self, org_id: int) -> Self:
        self.rag_log_qry_model.orgId = org_id
        return self

    def with_query(self, query: str) -> Self:
        self.rag_log_qry_model.query = query
        return self

    def with_response(self, response: str) -> Self:
        self.rag_log_qry_model.response = response
        return self

    def with_search_result_json(self, search_result_json: str) -> Self:
        self.rag_log_qry_model.search_result_json = search_result_json
        return self

    def with_time_elapsed(self, time_elapsed: float) -> Self:
        self.rag_log_qry_model.time_elapsed = time_elapsed
        return self

    def with_metadata(self, metadata: dict[str,Any]) -> Self:
        self.rag_log_qry_model.metadata = metadata
        return self

    def build(self) -> RagLogQryModel:
        return self.rag_log_qry_model
    
# ======================= RagLogQryModel model End =======================

# ======================= Structure input data model Start =======================
class StructureInputData(BaseModel):
    source_path: str = Field(..., description="Path to the source text file")
    context_data: str = Field(..., description="Input source text context for the source_path")
    prompt_type: str = Field(..., description="Document type")
    organization_id: int = Field(..., description="ID of the requesting organization")
    domain_id: str = Field(..., description="ID representing the business domain")
    document_type: str = Field(..., description="Type of document Ex: Property, Insurance,Logistic,...")
    user_id: Optional[int] = Field(None, description="ID of the user requesting extraction")
    source_lang: str = Field(..., description="Source file language")
    output_mode: str = Field(..., description="Mode of the output Ex :File,DB,API")
    output_format: str = Field(..., description="Format of the output file")
    schema_template_filepath: str = Field(..., description="Path for structured template")

class StructureInputDataBuilder:
    """
    Builder for StructureInputData model.
    Author: Jenson (09-09-2025)
    """
    def __init__(self):
        self.structure_input_data = StructureInputData(
            source_path = CommonPatterns.EMPTY_SPACE, 
            context_data = CommonPatterns.EMPTY_SPACE, 
            prompt_type = CommonPatterns.EMPTY_SPACE,
            organization_id = Numerical.ZERO,
            domain_id = CommonPatterns.ASTRICK,
            document_type =  CommonPatterns.EMPTY_SPACE,
            user_id = Numerical.ZERO,
            source_lang = CommonConstants.DFLT_LANG,
            output_mode = CommonPatterns.EMPTY_SPACE,
            output_format =  CommonPatterns.EMPTY_SPACE,
            schema_template_filepath =  CommonPatterns.EMPTY_SPACE
        )

    def with_source_path(self, source_path: str) -> Self:
        self.structure_input_data.source_path = source_path
        return self

    def with_context_data(self, context_data: str) -> Self:
        self.structure_input_data.context_data = context_data
        return self

    def with_prompt_type(self, prompt_type: str) -> Self:
        self.structure_input_data.prompt_type = prompt_type
        return self

    def with_organization_id(self, organization_id: str) -> Self:
        self.structure_input_data.organization_id = organization_id
        return self

    def with_domain_id(self, domain_id: str) -> Self:
        self.structure_input_data.domain_id = domain_id
        return self

    def with_document_type(self, document_type: str) -> Self:
        self.structure_input_data.document_type = document_type
        return self

    def with_user_id(self, user_id: str) -> Self:
        self.structure_input_data.user_id = user_id
        return self

    def with_source_lang(self, source_lang: str) -> Self:
        self.structure_input_data.source_lang = source_lang
        return self

    def with_output_mode(self, output_mode: str) -> Self:
        self.structure_input_data.output_mode = output_mode
        return self

    def with_output_format(self, output_format: str) -> Self:
        self.structure_input_data.output_format = output_format
        return self

    def with_schema_template_filepath(self, schema_template_filepath: str) -> Self:
        self.structure_input_data.schema_template_filepath = schema_template_filepath
        return self

    def build(self) -> StructureInputData:
        return self.structure_input_data

# ======================= Structure input data model End =======================

# ======================= Structure process pequest and Response model start =======================
class StructureRespProcessRequest(BaseModel):
    """
    Request model for PDF text extraction.
 
    Author: Jenson (19-09-2025)
 
    Attributes:
        request_reference_id (str): Unique identifier for the request.
        document_batch_id (str): Batch ID representing a group of documents.
        organization_id (int): ID of the requesting organization.
        domain_id (str): ID representing the business domain.
        user_id (Optional[int]): ID of the user requesting schema.
        desired_output_mode (str): Mode of the output Ex :File,DB,API.
        desired_output_format (str): Format of the output file.
        source_file_path (str): Source data file path for the structure response process request.
        source_lang (str): Source file language.
        document_type (str): Type of document Ex: Property, Insurance,Logistic,....
        output_filename (str): Name of the generated output file.
        ocr_result_storage_path (str): Location to store OCR results.
        inference_result_storage_path (str): Location to store inference results.
        ocr_result_file_path (str): OCR Extracted file path to process.
    """
    request_reference_id: StrictStr = Field(..., description="Unique identifier for the request")
    document_batch_id: StrictStr = Field(..., description="Batch ID representing a group of documents")
    organization_id: StrictInt = Field(..., description="ID of the requesting organization")
    domain_id: StrictStr = Field(..., description="ID representing the business domain")
    user_id: Optional[StrictInt] = Field(None, description="ID of the user requesting extraction")
    desired_output_mode: StrictStr = Field(..., description="Mode of the output Ex :File,DB,API")
    desired_output_format: StrictStr = Field(..., description="Format of the output file")
    source_file_path: StrictStr = Field(..., description="Source data file path for the structure response process request")
    source_lang: StrictStr = Field(..., description="Source file language")
    document_type: StrictStr = Field(..., description="Type of document Ex: Property, Insurance,Logistic,...")
    output_filename: Optional[StrictStr] = Field(None, description="Name of the generated output file")
    ocr_result_storage_path: Optional[StrictStr] = Field(None, description="Location to store OCR results")
    inference_result_storage_path: Optional[StrictStr] = Field(None, description="Location to store inference results")
    ocr_result_file_path: StrictStr = Field(None, description="OCR Extracted file path to process")


class StructureRespProcessResponse(StructureRespProcessRequest):
    """
    Response model for schema prompt.
 
    Author: Jenson (10-09-2025)
 
    Attributes:
        inference_result_file_path (str): Generated inference result file path.
        status (str): Status of the OCR and inference process.
        message (str): Detailed message about the processing status.
    """
    inference_result_file_path: StrictStr = Field(..., description="Location to store inference results file")
    status: StrictStr = Field(..., description="Status of the OCR and inference process")
    message: Optional[StrictStr] = Field(None, description="Detailed message about the processing status")


class StructureRespProcessResponseBuilder:
    def __init__(self, structure_resp_process_request: StructureRespProcessRequest):
        self.structure_resp_process_response = StructureRespProcessResponse(
            request_reference_id = structure_resp_process_request.request_reference_id or CommonPatterns.EMPTY_SPACE,
            document_batch_id = structure_resp_process_request.document_batch_id or CommonPatterns.EMPTY_SPACE,
            organization_id = structure_resp_process_request.organization_id or Numerical.ZERO,
            domain_id = structure_resp_process_request.domain_id or CommonPatterns.EMPTY_SPACE,
            user_id = structure_resp_process_request.user_id or Numerical.ZERO,
            desired_output_mode = structure_resp_process_request.desired_output_mode or CommonPatterns.EMPTY_SPACE,
            desired_output_format = structure_resp_process_request.desired_output_format or CommonPatterns.EMPTY_SPACE,
            source_file_path = structure_resp_process_request.source_file_path or CommonPatterns.EMPTY_SPACE,
            source_lang = structure_resp_process_request.source_lang or CommonPatterns.EMPTY_SPACE,
            document_type = structure_resp_process_request.document_type or CommonPatterns.EMPTY_SPACE,
            output_filename = structure_resp_process_request.output_filename or CommonPatterns.EMPTY_SPACE,
            ocr_result_storage_path = structure_resp_process_request.ocr_result_storage_path or CommonPatterns.EMPTY_SPACE,
            inference_result_storage_path = structure_resp_process_request.inference_result_storage_path or CommonPatterns.EMPTY_SPACE,
            ocr_result_file_path = structure_resp_process_request.ocr_result_file_path or CommonPatterns.EMPTY_SPACE,
            inference_result_file_path = CommonPatterns.EMPTY_SPACE,
            status = CommonPatterns.EMPTY_SPACE,
            message = CommonPatterns.EMPTY_SPACE
        )

    def with_request_reference_id(self, request_reference_id: str) -> Self:
        self.structure_resp_process_response.request_reference_id = request_reference_id
        return self

    def with_document_batch_id(self, document_batch_id: str) -> Self:
        self.structure_resp_process_response.document_batch_id = document_batch_id
        return self

    def with_organization_id(self, organization_id: int) -> Self:
        self.structure_resp_process_response.organization_id = organization_id
        return self

    def with_domain_id(self, domain_id: str) -> Self:
        self.structure_resp_process_response.domain_id = domain_id
        return self

    def with_user_id(self, user_id: int) -> Self:
        self.structure_resp_process_response.user_id = user_id
        return self

    def with_desired_output_mode(self, desired_output_mode: str) -> Self:
        self.structure_resp_process_response.desired_output_mode = desired_output_mode
        return self

    def with_desired_output_format(self, desired_output_format: str) -> Self:
        self.structure_resp_process_response.desired_output_format = desired_output_format
        return self

    def with_source_file_path(self, source_file_path: str) -> Self:
        self.structure_resp_process_response.source_file_path = source_file_path
        return self

    def with_source_lang(self, source_lang: str) -> Self:
        self.structure_resp_process_response.source_lang = source_lang
        return self

    def with_document_type(self, document_type: str) -> Self:
        self.structure_resp_process_response.document_type = document_type
        return self

    def with_output_filename(self, output_filename: str) -> Self:
        self.structure_resp_process_response.output_filename = output_filename
        return self

    def with_ocr_result_storage_path(self, ocr_result_storage_path: str) -> Self:
        self.structure_resp_process_response.ocr_result_storage_path = ocr_result_storage_path
        return self

    def with_inference_result_storage_path(self, inference_result_storage_path: str) -> Self:
        self.structure_resp_process_response.inference_result_storage_path = inference_result_storage_path
        return self

    def with_ocr_result_file_path(self, ocr_result_file_path: str) -> Self:
        self.structure_resp_process_response.ocr_result_file_path = ocr_result_file_path
        return self

    def with_inference_result_file_path(self, inference_result_file_path: str) -> Self:
        self.structure_resp_process_response.inference_result_file_path = inference_result_file_path
        return self

    def with_status(self, status: str) -> Self:
        self.structure_resp_process_response.status = status
        return self

    def with_message(self, message: str) -> Self:
        self.structure_resp_process_response.message = message
        return self

    def build(self) -> StructureRespProcessResponse:
        return self.structure_resp_process_response

# ======================= Structure process pequest and Response model end =======================