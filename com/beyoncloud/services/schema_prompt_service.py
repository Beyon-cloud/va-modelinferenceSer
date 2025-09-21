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
from com.beyoncloud.utils.file_util import TextLoader, FileCreation
from com.beyoncloud.schemas.prompt_gen_reqres_datamodel import (
    DocumentType, DocumentSchema
)
import com.beyoncloud.config.settings.env_config as config
from com.beyoncloud.utils.date_utils import current_timestamp_trim
from com.beyoncloud.common.constants import FileExtension, Status, PromptType
from com.beyoncloud.services.database_service import DataBaseService
from com.beyoncloud.schemas.rag_reqres_data_model import StructureInputData, StructureInputDataBuilder

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

        #schema_prompt = SchemaPrompt()
        #scheme_prompt_template = await schema_prompt.generate_schema_prompt(schema_prompt_request)
        #print(scheme_prompt_template)

        # Step 1: Load document content
        source_filepath = schema_prompt_request.source_path
        logger.info(f"Loading document: {source_filepath}")
        content =""
        if not source_filepath:
            raise ValueError("Source file not availbale to generate prompt")
    
        text_loader = TextLoader()
        content = text_loader.get_text_content(source_filepath)

        if not content:
            raise ValueError("Document content is empty or could not be extracted")

        # Step 2: Detect document type
        enhanced_document_processor = EnhancedDocumentProcessor()
        doc_type = schema_prompt_request.document_type
        detection_hints: Optional[List[str]] = None
        if doc_type:
            document_type = DocumentType(doc_type)
            logger.info(f"Using provided document type: {doc_type}")
        else:
            document_type = enhanced_document_processor.detect_document_type(content, detection_hints)
            logger.info(f"Detected document type: {document_type.value}")

        # Step 3: Generate schema
        schema_json = await enhanced_document_processor.generate_schema_template(schema_prompt_request)

        if not schema_json:
            raise ValueError("Schema JSON not extracted")

        # Step 4: Generate output file
        current_time_stamp = current_timestamp_trim()
        schema_dir_path = config.SCHEMA_PROMPT_DIR_PATH
        schema_filename = f"{config.SCHEMA_PROMPT_FILENAME}_{current_time_stamp}{FileExtension.TEXT}"
        file_creation = FileCreation()
        generated_filepath = file_creation.create_text_file(schema_dir_path,schema_filename,schema_json)

        # Step 5: Save date in DB
        print(f"Before save log table")
        print(self.db_service)
        oth_val = {
            "status" : Status.SUBMITTED,
            "document_type" : document_type,
            "schema_dir_path" : schema_dir_path,
            "schema_filename" : schema_filename,
            "generated_filepath" : generated_filepath,
            "schema_json" : schema_json
        }
        uuid_id = await self.db_service.save_or_update_schema_prompt_log(schema_prompt_request, oth_val)
        print(f"after save log table -- {uuid_id}")
        # Step 6: Generate Response
        schema_prompt_response = (SchemaPromptResponseBuilder(schema_prompt_request)
                    #.with_request_reference_id(schema_prompt_request.request_reference_id)
                    #.with_organization_id(schema_prompt_request.organization_id)
                    #.with_domain_id(schema_prompt_request.domain_id)
                    #.with_user_id(schema_prompt_request.user_id)
                    #.with_source_path(schema_prompt_request.source_path)
                    #.with_desired_output_mode(schema_prompt_request.desired_output_mode)
                    #.with_document_type(schema_prompt_request.document_type)
                    #.with_desired_output_format(schema_prompt_request.desired_output_format)
                    .with_output_directory(schema_dir_path)
                    .with_output_filename(schema_filename)
                    .with_schema_path(generated_filepath)
                    .build()
        )
        return schema_prompt_response

    async def generate_entity_prompt(self, entity_prompt_request: EntityPromptRequest):

        #schema_prompt = SchemaPrompt()
        #scheme_prompt_template = await schema_prompt.generate_entity_prompt(entity_prompt_request)
        #print(scheme_prompt_template)

        # Step 1: Extract source document content
        source_filepath = entity_prompt_request.source_path
        logger.info(f"Loading document: {source_filepath}")

        if not source_filepath:
            raise ValueError("Source file not availbale to generate prompt")
    
        text_loader = TextLoader()
        source_content = text_loader.get_text_content(source_filepath)
        #logger.info(f"Source Content --> {source_content}")
        #print(f"Source Content --> {source_content}")   
        if not source_content:
            raise ValueError("Document content is empty or could not be extracted")

        # Step 2: Extract schema template

        schema_filepath = entity_prompt_request.schema_path
        logger.info(f"Loading document: {schema_filepath}")
        if not source_filepath:
            raise ValueError("Source file not availbale to generate prompt")
        schema_template = text_loader.get_text_content(schema_filepath)
        schema_template = schema_template.replace("{", "{{").replace("}", "}}")
        logger.info(f"Schema template --> {schema_template}")
        print(f"Schema template --> {schema_template}")  
        if not source_content:
            raise ValueError("Schema template is empty or could not be extracted")

        # Step 3: Detect document type
        enhanced_document_processor = EnhancedDocumentProcessor()
        doc_type = entity_prompt_request.document_type
        detection_hints: Optional[List[str]] = None
        if doc_type:
            document_type = DocumentType(doc_type)
            logger.info(f"Using provided document type: {doc_type}")
        else:
            document_type = enhanced_document_processor.detect_document_type(source_content, detection_hints)
            logger.info(f"Detected document type: {document_type.value}")

        # Step 4: Get schema object
        custom_requirements: Optional[Dict[str, Any]] = None
        schema = enhanced_document_processor.generate_schema(document_type, custom_requirements)

        print("before builder.........")
        # Build the entity request using the builder pattern, ensuring the builder returns expected shape.
        structure_input_data = (
            StructureInputDataBuilder()
            .with_source_path(entity_prompt_request.source_path)
            .with_prompt_type(PromptType.GEN_PROMPT)
            .with_organization_id(entity_prompt_request.organization_id)
            .with_domain_id(entity_prompt_request.domain_id)
            .with_document_type(entity_prompt_request.document_type)
            .with_user_id(entity_prompt_request.user_id)
            .with_source_lang(entity_prompt_request.source_lang)
            .with_output_mode(entity_prompt_request.desired_output_mode)
            .with_output_format(entity_prompt_request.desired_output_format)
            .with_schema_path(entity_prompt_request.schema_path)
            .build()
        )
        print("After builder")
        prompts = await enhanced_document_processor.generate_extraction_prompts(schema, source_content, document_type, schema_template, structure_input_data)

        # Step 5: Save into database

        # Step 5: Generate Response
        entity_prompt_response = (EntityPromptResponseBuilder()
                    .with_request_reference_id(entity_prompt_request.request_reference_id)
                    .with_organization_id(entity_prompt_request.organization_id)
                    .with_domain_id(entity_prompt_request.domain_id)
                    .with_user_id(entity_prompt_request.user_id)
                    .with_source_path(entity_prompt_request.source_path)
                    .with_desired_output_mode(entity_prompt_request.desired_output_mode)
                    .with_document_type(entity_prompt_request.document_type)
                    .with_desired_output_format(entity_prompt_request.desired_output_format)
                    .with_schema_path(entity_prompt_request.schema_path)
                    .with_system_prompt(prompts["system_prompt"])
                    .with_user_prompt(prompts["user_prompt"])
                    .with_input_variables(prompts["input_variables"])
                    .build()
        )
        return entity_prompt_response

