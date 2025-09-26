import logging
import os
import json
from typing import Dict, List, Any, Optional
from com.beyoncloud.schemas.prompt_gen_reqres_datamodel import (
    SchemaPromptRequest,SchemaPromptResponse, EntityPromptRequest, EntityPromptResponse, 
    SchemaPromptResponseBuilder, EntityPromptResponseBuilder
)
from com.beyoncloud.processing.prompt.prompt_generation.schema_prompt import SchemaPrompt
from com.beyoncloud.processing.prompt.prompt_generation.enhance_document_processor import EnhancedDocumentProcessor
from com.beyoncloud.utils.file_util import TextLoader, FileCreation, PathValidator, FetchContent
from com.beyoncloud.schemas.prompt_gen_reqres_datamodel import (
    DocumentType, DocumentSchema
)
import com.beyoncloud.config.settings.env_config as config
from com.beyoncloud.utils.date_utils import current_timestamp_trim
from com.beyoncloud.common.constants import FileExtension, Status, PromptType, CommonPatterns, FileFormats
from com.beyoncloud.services.database_service import DataBaseService
from com.beyoncloud.schemas.rag_reqres_data_model import StructureInputData, StructureInputDataBuilder
from com.beyoncloud.processing.rag_process_impl import RagProcessImpl
from com.beyoncloud.utils.date_utils import current_date_trim

logger = logging.getLogger(__name__)

class SchemaPromptService:
    """
    Service responsible for managing the lifecycle of schema prompt processing tasks.

    Responsibilities:
    ----------------
     Save or update data source job metadata in the database.
     Dispatch background tasks for processing jobs asynchronously.
     Select appropriate handler (Website/API/etc.) based on the data source.
     Update job status in the database in case of success/failure.

    Attributes:
    -----------
    db_service : DataBaseService
        Service layer for database interactions.
    handlers : dict
        Mapping of data source types to their respective handlers.
    """

    def __init__(self, db_service: DataBaseService | None = None):
        """
        Initialize the DataProcessingService with handlers and DB service.

        Args:
            db_service (DataBaseService, optional): Dependency-injected DB service.
                                                   If None, a new instance is created.
        """
        print(db_service)
        self.db_service = db_service or DataBaseService()
        print(self.db_service)

    async def generate_schema_prompt(self, schema_prompt_request: SchemaPromptRequest) -> SchemaPromptResponse:

        # Step 1: Load document content
        source_filepath = schema_prompt_request.ocr_result_file_path
        logger.info(f"Loading document: {source_filepath}")
        file_context =""
        if not source_filepath or not isinstance(source_filepath, str):
            raise ValueError("Source filepath is required and must be a non-empty string")
    
        fetch_content = FetchContent()
        file_context = CommonPatterns.EMPTY_SPACE
        file_path = schema_prompt_request.ocr_result_file_path.lower()
        if file_path.endswith(FileExtension.JSON):
            file_context  = fetch_content.fetch_ocr_content(schema_prompt_request.ocr_result_file_path)
        else:
            text_loader = TextLoader()
            file_context = text_loader.get_text_content(schema_prompt_request.ocr_result_file_path)

        if not file_context:
            raise ValueError("Document content is empty or could not be extracted")

        # Step 2: Build the entity request using the builder pattern, ensuring the builder returns expected shape.
        structure_input_data = (
            StructureInputDataBuilder()
            .with_source_path(schema_prompt_request.ocr_result_file_path)
            .with_context_data(file_context)
            .with_prompt_type(PromptType.SCHEMA_PROMPT)
            .with_organization_id(schema_prompt_request.organization_id)
            .with_domain_id(schema_prompt_request.domain_id)
            .with_document_type(schema_prompt_request.document_type)
            .with_user_id(schema_prompt_request.user_id)
            .with_source_lang(schema_prompt_request.source_lang)
            .with_output_mode(schema_prompt_request.desired_output_mode)
            .with_output_format(schema_prompt_request.desired_output_format)
            .build()
        )

        # Step 3: Detect document type
        enhanced_document_processor = EnhancedDocumentProcessor()
        document_type = schema_prompt_request.document_type
        """
        doc_type = schema_prompt_request.document_type
        detection_hints: Optional[List[str]] = None
        if doc_type:
            document_type = DocumentType(doc_type)
            logger.info(f"Using provided document type: {doc_type}")
        else:
            document_type = enhanced_document_processor.detect_document_type(content, detection_hints)
            logger.info(f"Detected document type: {document_type.value}")
        """

        # Step 4: Generate schema template
        ragProcessImpl = RagProcessImpl()
        response = None
        try:
            response = await ragProcessImpl.generate_structured_response(structure_input_data)
        except Exception as exc:
            # Log or re-raise depending on your application's logging strategy
            # For example: logger.exception("generate_structured_response failed")
            raise

        # Step 4: Generate schema template
        #schema_json = await enhanced_document_processor.generate_schema_template(schema_prompt_request)
        # Step 4: Clean response and extract specific file content
        cleaned_response = fetch_content.fetch_schema_content(response, schema_prompt_request.desired_output_format)

        if not cleaned_response:
            raise ValueError("Schema JSON not extracted")

        # Step 4: Generate output file
        sys_gen_schema_temp = cleaned_response
        file_extension = FileExtension.TEXT
        if schema_prompt_request.desired_output_format == FileFormats.JSON:
            file_extension = FileExtension.JSON
            sys_gen_schema_temp = json.dumps(cleaned_response)
        elif schema_prompt_request.desired_output_format == FileFormats.CSV:
            file_extension = FileExtension.CSV
        elif schema_prompt_request.desired_output_format == FileFormats.XLSX:
            file_extension = FileExtension.XLSX

        print(f"file_extension ---> {file_extension}")
        file_creation = FileCreation()
        output_filename = file_creation.generate_file_name(
            schema_prompt_request.output_filename,
            config.SCHEMA_PROMPT_FILENAME,
            schema_prompt_request.request_reference_id,None,
            file_extension
        )

        print(f"output_filename ---> {output_filename}")
        path_validator = PathValidator()
        is_valid_directory = path_validator.is_directory(schema_prompt_request.inference_result_storage_path)

        if is_valid_directory:
            output_dir_path = schema_prompt_request.inference_result_storage_path
        else:
            output_dir_path = os.path.join(
                config.SCHEMA_PROMPT_DIR_PATH,
                str(schema_prompt_request.organization_id),
                current_date_trim()
            )

        print(f"output_dir_path ------> {output_dir_path}")
        if file_extension == FileExtension.JSON:
            generated_filepath = file_creation.create_json_file(output_dir_path,output_filename,cleaned_response)
        else:
            generated_filepath = file_creation.create_text_file(output_dir_path,output_filename,cleaned_response)

        # Step 5: Save date in DB
        print(f"Before save log table")
        print(self.db_service)

        oth_val = {
            "status" : Status.SUBMITTED,
            "document_type" : document_type,
            "schema_dir_path" : output_dir_path,
            "schema_filename" : output_filename,
            "generated_filepath" : generated_filepath,
            "schema_json" : sys_gen_schema_temp
        }
        uuid_id = await self.db_service.save_or_update_schema_prompt_log(schema_prompt_request, oth_val)
        print(f"after save log table -- {uuid_id}")

        # Step 6: Generate Response
        schema_prompt_response = (SchemaPromptResponseBuilder(schema_prompt_request)
                    .with_inference_result_storage_path(output_dir_path)
                    .with_output_filename(output_filename)
                    .with_schema_template_filepath(generated_filepath)
                    .build()
        )
        return schema_prompt_response

    async def generate_entity_prompt(self, entity_prompt_request: EntityPromptRequest):

        # Step 1: Extract source document content
        text_loader = TextLoader()
        source_content = ""

        # Step 2: Extract schema template

        schema_filepath = entity_prompt_request.schema_template_filepath
        logger.info(f"Loading document: {schema_filepath}")
        if not schema_filepath:
            raise ValueError("Schema file not availbale to generate prompt")

        schema_template = text_loader.get_text_content(schema_filepath)
        schema_template = schema_template.replace("{", "{{").replace("}", "}}")
        logger.info(f"Schema template --> {schema_template}")
        print(f"Schema template --> {schema_template}")  
        if not schema_template:
            raise ValueError("Schema template is empty or could not be extracted")

        # Step 3: Detect document type
        enhanced_document_processor = EnhancedDocumentProcessor()
        document_type = entity_prompt_request.document_type

        """
        doc_type = entity_prompt_request.document_type
        detection_hints: Optional[List[str]] = None
        if doc_type:
            document_type = DocumentType(doc_type)
            logger.info(f"Using provided document type: {doc_type}")
        else:
            document_type = enhanced_document_processor.detect_document_type(source_content, detection_hints)
            logger.info(f"Detected document type: {document_type.value}")
        """

        # Step 4: Get schema object
        custom_requirements: Optional[Dict[str, Any]] = None
        #schema = enhanced_document_processor.generate_schema(document_type, custom_requirements)
        #schema = DocumentSchema()

        print("before builder.........")
        # Build the entity request using the builder pattern, ensuring the builder returns expected shape.
        structure_input_data = (
            StructureInputDataBuilder()
            .with_source_path(entity_prompt_request.source_file_path)
            .with_prompt_type(PromptType.GEN_PROMPT)
            .with_organization_id(entity_prompt_request.organization_id)
            .with_domain_id(entity_prompt_request.domain_id)
            .with_document_type(entity_prompt_request.document_type)
            .with_user_id(entity_prompt_request.user_id)
            .with_source_lang(entity_prompt_request.source_lang)
            .with_output_mode(entity_prompt_request.desired_output_mode)
            .with_output_format(entity_prompt_request.desired_output_format)
            .with_schema_template_filepath(entity_prompt_request.schema_template_filepath)
            .build()
        )
        print("After builder")
        prompts = await enhanced_document_processor.generate_extraction_prompts(source_content, document_type, schema_template, structure_input_data)

        # Step 5: Save into database

        # Step 5: Generate Response
        entity_prompt_response = (EntityPromptResponseBuilder(entity_prompt_request)
                    #.with_request_reference_id(entity_prompt_request.request_reference_id)
                    #.with_organization_id(entity_prompt_request.organization_id)
                    #.with_domain_id(entity_prompt_request.domain_id)
                    #.with_user_id(entity_prompt_request.user_id)
                    #.with_source_path(entity_prompt_request.source_file_path)
                    #.with_desired_output_mode(entity_prompt_request.desired_output_mode)
                    #.with_document_type(entity_prompt_request.document_type)
                    #.with_desired_output_format(entity_prompt_request.desired_output_format)
                    #.with_schema_template_filepath(entity_prompt_request.schema_template_filepath)
                    .with_system_prompt(prompts["system_prompt"])
                    .with_user_prompt(prompts["user_prompt"])
                    .with_input_variables(prompts["input_variables"])
                    .build()
        )
        return entity_prompt_response

