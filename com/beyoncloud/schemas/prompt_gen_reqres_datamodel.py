from pydantic import BaseModel, Field, StrictInt, StrictStr
from typing import Dict, List, Any, Optional, Union, Callable, Self
from enum import Enum
from dataclasses import dataclass, field, asdict
from com.beyoncloud.common.constants import Numerical, CommonPatterns

# Enums and data classes
class DocumentType(Enum):
    PROPERTY = "property"
    INSURANCE = "insurance"
    EDUCATION = "education"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    LEGAL = "legal"
    EMPLOYMENT = "employment"
    GOVERNMENT = "government"
    BUSINESS = "business"
    TECHNICAL = "technical"
    LAND_REGISTRATION= "landRegistration"
    LEASE_DEED="lease_deed"
    INVOICE = "invoice"
  

class FieldType(Enum):
    STRING = "string"
    NUMBER = "number"
    DATE = "date"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    TEXT="text"
    PERSON_NAME="person_name"

@dataclass
class SchemaField:
    name: str
    field_type: FieldType
    description: str
    required: bool = True
    validation_rules: Optional[Dict[str, Any]] = None
    examples: Optional[List[str]] = None
    sub_fields: Optional[List['SchemaField']] = None

@dataclass
class DocumentSchema:
    document_type: DocumentType
    schema_name: str
    description: str
    fields: List[SchemaField]
    validation_rules: Optional[Dict[str, Any]] = None
    processing_hints: Optional[List[str]] = None

@dataclass
class ExtractionResult:
    """Result of document extraction process"""
    success: bool
    extracted_data: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None
    schema_used: Optional[DocumentSchema] = None
    processing_metadata: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None

@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    max_retries: int = 1
    timeout_seconds: int = 30
    enable_validation: bool = True
    enable_post_processing: bool = True
    save_intermediate_results: bool = False
    output_directory: Optional[str] = None
    custom_validators: Optional[List[Callable]] = None

class SchemaPromptRequest(BaseModel):
    """
    Request model for PDF text extraction.
 
    Author: Jenson (05-09-2025)
 
    Attributes:
        request_reference_id (str): Unique identifier for the request.
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
    organization_id: StrictInt = Field(..., description="ID of the requesting organization")
    domain_id: StrictStr = Field(..., description="ID representing the business domain")
    user_id: Optional[StrictInt] = Field(None, description="ID of the user requesting extraction")
    desired_output_mode: StrictStr = Field(..., description="Mode of the output Ex :File,DB,API")
    desired_output_format: StrictStr = Field(..., description="Format of the output file")
    source_file_path: StrictStr = Field(..., description="Path to the source text file")
    source_lang: StrictStr = Field(..., description="Source file language")
    document_type: StrictStr = Field(..., description="Type of document Ex: Property, Insurance,Logistic,...")
    output_filename: Optional[StrictStr] = Field(None, description="Name of the generated output file")
    ocr_result_storage_path: Optional[StrictStr] = Field(None, description="Location to store OCR results")
    inference_result_storage_path: Optional[StrictStr] = Field(None, description="Location to store inference results")
    ocr_result_file_path: StrictStr = Field(None, description="Extracted file path to process")


class SchemaPromptResponse(SchemaPromptRequest):
    """
    Response model for schema prompt.
 
    Author: Jenson (10-09-2025)
 
    Attributes:
        schema_template_filepath (str): Generated schema template file path.
    """
    schema_template_filepath: StrictStr = Field(..., description="Generated schema template file path")


class SchemaPromptResponseBuilder:
    def __init__(self, schema_prompt_request: SchemaPromptRequest):
        self.schema_prompt_response = SchemaPromptResponse(
            request_reference_id = schema_prompt_request.request_reference_id or CommonPatterns.EMPTY_SPACE,
            organization_id = schema_prompt_request.organization_id or Numerical.ZERO,
            domain_id = schema_prompt_request.domain_id or CommonPatterns.EMPTY_SPACE,
            user_id = schema_prompt_request.user_id or Numerical.ZERO,
            desired_output_mode = schema_prompt_request.desired_output_mode or CommonPatterns.EMPTY_SPACE,
            desired_output_format = schema_prompt_request.desired_output_format or CommonPatterns.EMPTY_SPACE,
            source_file_path = schema_prompt_request.source_file_path or CommonPatterns.EMPTY_SPACE,
            source_lang = schema_prompt_request.source_lang or CommonPatterns.EMPTY_SPACE,
            document_type = schema_prompt_request.document_type or CommonPatterns.EMPTY_SPACE,
            output_filename = schema_prompt_request.output_filename or CommonPatterns.EMPTY_SPACE,
            ocr_result_storage_path = schema_prompt_request.ocr_result_storage_path or CommonPatterns.EMPTY_SPACE,
            inference_result_storage_path = schema_prompt_request.inference_result_storage_path or CommonPatterns.EMPTY_SPACE,
            ocr_result_file_path = schema_prompt_request.ocr_result_file_path or CommonPatterns.EMPTY_SPACE,
            schema_template_filepath = CommonPatterns.EMPTY_SPACE
        )

    def with_request_reference_id(self, request_reference_id: str) -> Self:
        self.schema_prompt_response.request_reference_id = request_reference_id
        return self

    def with_organization_id(self, organization_id: str) -> Self:
        self.schema_prompt_response.organization_id = organization_id
        return self

    def with_domain_id(self, domain_id: str) -> Self:
        self.schema_prompt_response.domain_id = domain_id
        return self

    def with_user_id(self, user_id: str) -> Self:
        self.schema_prompt_response.user_id = user_id
        return self

    def with_desired_output_mode(self, desired_output_mode: str) -> Self:
        self.schema_prompt_response.desired_output_mode = desired_output_mode
        return self

    def with_desired_output_format(self, desired_output_format: str) -> Self:
        self.schema_prompt_response.desired_output_format = desired_output_format
        return self

    def with_source_file_path(self, source_file_path: str) -> Self:
        self.schema_prompt_response.source_file_path = source_file_path
        return self

    def with_source_lang(self, source_lang: str) -> Self:
        self.schema_prompt_response.source_lang = source_lang
        return self

    def with_document_type(self, document_type: str) -> Self:
        self.schema_prompt_response.document_type = document_type
        return self

    def with_output_filename(self, output_filename: str) -> Self:
        self.schema_prompt_response.output_filename = output_filename
        return self

    def with_ocr_result_storage_path(self, ocr_result_storage_path: str) -> Self:
        self.schema_prompt_response.ocr_result_storage_path = ocr_result_storage_path
        return self

    def with_inference_result_storage_path(self, inference_result_storage_path: str) -> Self:
        self.schema_prompt_response.inference_result_storage_path = inference_result_storage_path
        return self

    def with_ocr_result_file_path(self, ocr_result_file_path: str) -> Self:
        self.schema_prompt_response.ocr_result_file_path = ocr_result_file_path
        return self

    def with_schema_template_filepath(self, schema_template_filepath: str) -> Self:
        self.schema_prompt_response.schema_template_filepath = schema_template_filepath
        return self

    def build(self) -> SchemaPromptResponse:
        return self.schema_prompt_response

class EntityPromptRequest(BaseModel):
    """
    Request model for the entity prompt.
 
    Author: Jenson (10-09-2025)
 
    Attributes:
        request_reference_id (str): Unique identifier for the request.
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
        schema_template_filepath (str): Updated schemea template file path
    """
    request_reference_id: StrictStr = Field(..., description="Unique identifier for the request")
    organization_id: StrictInt = Field(..., description="ID of the requesting organization")
    domain_id: StrictStr = Field(..., description="ID representing the business domain")
    user_id: Optional[StrictInt] = Field(None, description="ID of the user requesting extraction")
    desired_output_mode: StrictStr = Field(..., description="Mode of the output Ex :File,DB,API")
    desired_output_format: StrictStr = Field(..., description="Format of the output file")
    source_file_path: StrictStr = Field(..., description="Path to the source text file")
    source_lang: StrictStr = Field(..., description="Source file language")
    document_type: StrictStr = Field(..., description="Type of document Ex: Property, Insurance,Logistic,...")
    schema_template_filepath: StrictStr = Field(..., description="Updated schemea template file path")

class EntityPromptResponse(EntityPromptRequest):
    """
    Response model for the entity prompt.
 
    Author: Jenson (10-09-2025)
 
    Attributes:
        system_prompt (str): generated system prompt.
        user_prompt (str): generated use prompt.
        user_prompt (str): Input propmpt variables.
    """

    system_prompt: StrictStr = Field(..., description="Schema file path")
    user_prompt: StrictStr = Field(..., description="Schema file path")
    input_variables: StrictStr = Field(..., description="Schema file path")

class EntityPromptResponseBuilder:
    def __init__(self, entity_prompt_request: EntityPromptRequest) -> EntityPromptResponse:
        self.entity_prompt_response = EntityPromptResponse(
            request_reference_id = entity_prompt_request.request_reference_id or CommonPatterns.EMPTY_SPACE,
            organization_id = entity_prompt_request.organization_id or  Numerical.ZERO,
            domain_id = entity_prompt_request.domain_id or  CommonPatterns.EMPTY_SPACE,
            user_id = entity_prompt_request.user_id or  Numerical.ZERO,
            desired_output_mode = entity_prompt_request.desired_output_mode or  CommonPatterns.EMPTY_SPACE,
            desired_output_format = entity_prompt_request.desired_output_format or  CommonPatterns.EMPTY_SPACE,
            source_file_path = entity_prompt_request.source_file_path or  CommonPatterns.EMPTY_SPACE,
            source_lang = entity_prompt_request.source_lang or  CommonPatterns.EMPTY_SPACE,
            document_type = entity_prompt_request.document_type or  CommonPatterns.EMPTY_SPACE,
            schema_template_filepath = entity_prompt_request.schema_template_filepath or  CommonPatterns.EMPTY_SPACE,
            system_prompt = CommonPatterns.EMPTY_SPACE,
            user_prompt = CommonPatterns.EMPTY_SPACE,
            input_variables = CommonPatterns.EMPTY_SPACE
        )

    def with_request_reference_id(self, request_reference_id: str) -> Self:
        self.entity_prompt_response.request_reference_id = request_reference_id
        return self

    def with_organization_id(self, organization_id: str) -> Self:
        self.entity_prompt_response.organization_id = organization_id
        return self

    def with_domain_id(self, domain_id: str) -> Self:
        self.entity_prompt_response.domain_id = domain_id
        return self

    def with_user_id(self, user_id: str) -> Self:
        self.entity_prompt_response.user_id = user_id
        return self

    def with_desired_output_mode(self, desired_output_mode: str) -> Self:
        self.entity_prompt_response.desired_output_mode = desired_output_mode
        return self

    def with_desired_output_format(self, desired_output_format: str) -> Self:
        self.entity_prompt_response.desired_output_format = desired_output_format
        return self

    def with_source_file_path(self, source_file_path: str) -> Self:
        self.entity_prompt_response.source_file_path = source_file_path
        return self

    def with_source_lang(self, source_lang: str) -> Self:
        self.entity_prompt_response.source_lang = source_lang
        return self

    def with_document_type(self, document_type: str) -> Self:
        self.entity_prompt_response.document_type = document_type
        return self

    def with_schema_template_filepath(self, schema_template_filepath: str) -> Self:
        self.entity_prompt_response.schema_template_filepath = schema_template_filepath
        return self

    def with_system_prompt(self, system_prompt: str) -> Self:
        self.entity_prompt_response.system_prompt = system_prompt
        return self

    def with_user_prompt(self, user_prompt: str) -> Self:
        self.entity_prompt_response.user_prompt = user_prompt
        return self

    def with_input_variables(self, input_variables: str) -> Self:
        self.entity_prompt_response.input_variables = input_variables
        return self

    def build(self) -> EntityPromptResponse:
        return self.entity_prompt_response