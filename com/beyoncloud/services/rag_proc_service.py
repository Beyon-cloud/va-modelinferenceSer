import os
from typing import Any
import logging
from com.beyoncloud.processing.rag_process_impl import RagProcessImpl
from com.beyoncloud.schemas.rag_reqres_data_model import (
    RagReqDataModel, 
    RagRespDataModel, 
    StructureInputData, 
    StructureRespProcessRequest,
    StructureRespProcessResponse,
    StructureInputDataBuilder,
    StructureRespProcessResponseBuilder
)
from com.beyoncloud.schemas.prompt_gen_reqres_datamodel import (
    SchemaPromptRequest,
    EntityPromptRequest, 
)
from com.beyoncloud.processing.generation.generator import RagGeneratorProcess
from com.beyoncloud.services.database_service import DataBaseService
from com.beyoncloud.common.constants import (
    PromptType, FileExtension, 
    FileFormats, Status, 
    CommonPatterns, Numerical
)
from com.beyoncloud.utils.file_util import FetchContent, FileCreation, TextLoader, PathValidator
from com.beyoncloud.utils.date_utils import current_date_trim
import com.beyoncloud.config.settings.env_config as config
from com.beyoncloud.services.schema_prompt_service import SchemaPromptService
from com.beyoncloud.processing.prompt.prompt_template import get_temp_prompt_template
from com.beyoncloud.utils.date_utils import get_current_timestamp_string

logger = logging.getLogger(__name__)

async def rag_chat_process(rag_req_data_model: RagReqDataModel) -> RagRespDataModel:

    starttime = get_current_timestamp_string()
    rag_process_impl = RagProcessImpl()
    response = await rag_process_impl.generate_rag_response(rag_req_data_model)
    endtime = get_current_timestamp_string()
    logger.info("Process taking time from '"+starttime+"' to '"+endtime+"'")
    return response

class InfrenceService:

    def __init__(self, db_service: DataBaseService | None = None):
        """
        Initialize the InfrenceService with DB service.

        Args:
            db_service (DataBaseService, optional): Dependency-injected DB service.
                                                   If None, a new instance is created.
        """
        self.db_service = db_service or DataBaseService()

    async def inf_structure_resp_process(self, structure_resp_process_request: StructureRespProcessRequest) -> StructureRespProcessResponse:

        # Step 1: Read source file context data
        if not structure_resp_process_request.ocr_result_file_path:
            logger.info("OCR result file path is empty")
            raise ValueError("OCR result file path is empty")

        fetch_content = FetchContent()
        file_path = structure_resp_process_request.ocr_result_file_path.lower()
        if file_path.endswith(FileExtension.JSON):
            file_context  = fetch_content.fetch_ocr_content(structure_resp_process_request.ocr_result_file_path)
        else:
            text_loader = TextLoader()
            file_context = text_loader.get_text_content(structure_resp_process_request.ocr_result_file_path)

        # Step 2: Build the entity request using the builder pattern, ensuring the builder returns expected shape.
        structure_input_data = (
            StructureInputDataBuilder()
            .with_source_path(structure_resp_process_request.ocr_result_file_path)
            .with_context_data(file_context)
            .with_prompt_type(PromptType.RESP_PROMPT)
            .with_organization_id(structure_resp_process_request.organization_id)
            .with_domain_id(structure_resp_process_request.domain_id)
            .with_document_type(structure_resp_process_request.document_type)
            .with_user_id(structure_resp_process_request.user_id)
            .with_source_lang(structure_resp_process_request.source_lang)
            .with_output_mode(structure_resp_process_request.desired_output_mode)
            .with_output_format(structure_resp_process_request.desired_output_format)
            .build()
        )

        # Step 3: Call Inference model
        rag_process_impl = RagProcessImpl()
        response = await rag_process_impl.generate_structured_response(structure_input_data)


        # Step 4: Generate output file
        file_extension = FileExtension.TEXT
        final_output_format = FileFormats.JSON
        if (
            structure_resp_process_request.desired_output_format == FileFormats.JSON
            or structure_resp_process_request.desired_output_format == FileFormats.CSV
            or structure_resp_process_request.desired_output_format == FileFormats.XLSX
            or structure_resp_process_request.desired_output_format == FileFormats.PDF
        ):
            file_extension = FileExtension.JSON
            final_output_format = FileFormats.JSON
        elif structure_resp_process_request.desired_output_format == FileFormats.CSV:
            file_extension = FileExtension.CSV


        # Step 5: Clean response and extract specific file content
        cleaned_response = fetch_content.fetch_schema_content(response, final_output_format)
        print(f"cleaned_response --> {cleaned_response}")

        file_creation = FileCreation()
        output_filename = file_creation.generate_file_name(
            structure_resp_process_request.output_filename,
            config.CLARIDATA_FILENAME,
            structure_resp_process_request.document_batch_id,
            structure_resp_process_request.request_reference_id,
            file_extension
        )
        print(f"output_filename ---> {output_filename}")

        path_validator = PathValidator()
        is_valid_directory = path_validator.is_directory(structure_resp_process_request.inference_result_storage_path)

        if is_valid_directory:
            output_dir_path = structure_resp_process_request.inference_result_storage_path
        else:
            output_dir_path = os.path.join(
                config.CLARIDATA_DIR_PATH,
                str(structure_resp_process_request.organization_id),
                current_date_trim()
            )

        if file_extension == FileExtension.JSON:
            generated_filepath = file_creation.create_json_file(output_dir_path,output_filename,cleaned_response)
        else:
            generated_filepath = file_creation.create_text_file(output_dir_path,output_filename,cleaned_response)

        # Step 6: Response datamodel formation
        structure_resp_process_response = (StructureRespProcessResponseBuilder(structure_resp_process_request)
                    .with_inference_result_file_path(generated_filepath)
                    .with_status(Status.COMPLETED)
                    .with_message("File generated successfully")
                    .build()
        )

        return structure_resp_process_response


    async def temp_structure_resp_process(self, structure_resp_process_request: StructureRespProcessRequest) -> Any:
        structure_resp_process_request.desired_output_format=structure_resp_process_request.desired_output_format.lower()
        """Main orchestrator for structured response generation."""
        file_context = self._validate_and_load_file(structure_resp_process_request)
        schema_prompt_response = await self._generate_schema_prompt(structure_resp_process_request)
        entity_prompt_response = await self._generate_entity_prompt(structure_resp_process_request, schema_prompt_response)
        structure_input_data = self._build_structure_input(structure_resp_process_request, file_context)
        response = await self._invoke_rag_model(structure_input_data, entity_prompt_response)
        generated_filepath = self._generate_output_file(structure_resp_process_request, response)
        return self._build_final_response(structure_resp_process_request, generated_filepath)

    # ---------------- Helper Methods ---------------- #

    def _validate_and_load_file(self, req: Any) -> str:
        """Step 1: Validate and read OCR or text file content."""
        if not req.ocr_result_file_path:
            logger.error("temp_structure_resp_process - OCR result file path is empty")
            raise ValueError("temp_structure_resp_process - OCR result file path is empty")

        fetch_content = FetchContent()
        file_path = req.ocr_result_file_path.lower()
        if file_path.endswith(FileExtension.JSON):
            file_context = fetch_content.fetch_ocr_content(req.ocr_result_file_path)
        else:
            file_context = TextLoader().get_text_content(req.ocr_result_file_path)

        logger.info("Step 1 completed: File context loaded")
        return file_context

    async def _generate_schema_prompt(self, req: Any) -> Any:
        """Step 2 & 3: Build schema prompt request and generate prompt."""
        schema_prompt_request = SchemaPromptRequest(
            request_reference_id=req.request_reference_id or CommonPatterns.EMPTY_SPACE,
            organization_id=req.organization_id or Numerical.ZERO,
            domain_id=req.domain_id or CommonPatterns.EMPTY_SPACE,
            user_id=req.user_id or Numerical.ZERO,
            desired_output_mode=req.desired_output_mode or CommonPatterns.EMPTY_SPACE,
            desired_output_format=req.desired_output_format or CommonPatterns.EMPTY_SPACE,
            source_file_path=req.source_file_path or CommonPatterns.EMPTY_SPACE,
            source_lang=req.source_lang or CommonPatterns.EMPTY_SPACE,
            document_type=req.document_type or CommonPatterns.EMPTY_SPACE,
            output_filename=req.output_filename or CommonPatterns.EMPTY_SPACE,
            ocr_result_storage_path=req.ocr_result_storage_path or CommonPatterns.EMPTY_SPACE,
            inference_result_storage_path=req.inference_result_storage_path or CommonPatterns.EMPTY_SPACE,
            ocr_result_file_path=req.ocr_result_file_path or CommonPatterns.EMPTY_SPACE
        )
        schema_service = SchemaPromptService()
        schema_response = await schema_service.generate_schema_prompt(schema_prompt_request)
        logger.info(f"Step 3 completed: Schema template -> {schema_response.schema_template_filepath}")
        return schema_response

    async def _generate_entity_prompt(self, req: Any, schema_response: Any) -> Any:
        """Step 4 & 5: Build entity prompt and generate response prompt."""
        entity_request = EntityPromptRequest(
            request_reference_id=req.request_reference_id or CommonPatterns.EMPTY_SPACE,
            organization_id=req.organization_id or Numerical.ZERO,
            domain_id=req.domain_id or CommonPatterns.EMPTY_SPACE,
            user_id=req.user_id or Numerical.ZERO,
            desired_output_mode=req.desired_output_mode or CommonPatterns.EMPTY_SPACE,
            desired_output_format=req.desired_output_format or CommonPatterns.EMPTY_SPACE,
            source_file_path=req.source_file_path or CommonPatterns.EMPTY_SPACE,
            source_lang=req.source_lang or CommonPatterns.EMPTY_SPACE,
            document_type=req.document_type or CommonPatterns.EMPTY_SPACE,
            schema_template_filepath=schema_response.schema_template_filepath or CommonPatterns.EMPTY_SPACE
        )
        schema_service = SchemaPromptService()
        entity_response = await schema_service.generate_entity_prompt(entity_request)
        logger.info("Step 5 completed: Entity prompt generated")
        return entity_response

    def _build_structure_input(self, req: Any, file_context: str) -> Any:
        """Step 6: Build structured input data for inference."""
        return (
            StructureInputDataBuilder()
            .with_source_path(req.ocr_result_file_path)
            .with_context_data(file_context)
            .with_prompt_type(PromptType.RESP_PROMPT)
            .with_organization_id(req.organization_id)
            .with_domain_id(req.domain_id)
            .with_document_type(req.document_type)
            .with_user_id(req.user_id)
            .with_source_lang(req.source_lang)
            .with_output_mode(req.desired_output_mode)
            .with_output_format(req.desired_output_format)
            .build()
        )

    async def _invoke_rag_model(self, structure_input_data: Any, entity_response: Any) -> str:
        """Step 7: Call inference (RAG) model to generate structured output."""
        rag_process = RagGeneratorProcess()
        if config.ENABLE_HF_INFRENCE_YN == "Y":
            response = await rag_process.temp_hf_response(
                structure_input_data,
                entity_response.system_prompt,
                entity_response.user_prompt,
                entity_response.input_variables
            )
        else:
            response = await rag_process.temp_generate_structured_response(
                structure_input_data,
                entity_response.system_prompt,
                entity_response.user_prompt,
                entity_response.input_variables
            )
        logger.info("Step 7 completed: Inference generated successfully")
        return response

    def _generate_output_file(self, req: Any, response: str) -> str:
        """Step 8 & 9: Clean response, determine path, and write output file."""
        fetch_content = FetchContent()
        file_creation = FileCreation()

        final_format, file_extension = self._determine_output_format(req.desired_output_format)
        cleaned_response = fetch_content.fetch_schema_content(response, final_format)

        output_filename = file_creation.generate_file_name(
            req.output_filename,
            config.CLARIDATA_FILENAME,
            req.document_batch_id,
            req.request_reference_id,
            file_extension
        )

        output_dir_path = self._resolve_output_dir(req.inference_result_storage_path, req.organization_id)
        if file_extension == FileExtension.JSON:
            return file_creation.create_json_file(output_dir_path, output_filename, cleaned_response)
        return file_creation.create_text_file(output_dir_path, output_filename, cleaned_response)

    def _determine_output_format(self, desired_format: str) -> tuple:
        """Determine file format and extension."""
        if desired_format in [FileFormats.JSON, FileFormats.CSV, FileFormats.XLSX, FileFormats.PDF]:
            return FileFormats.JSON, FileExtension.JSON
        elif desired_format == FileFormats.CSV:
            return FileFormats.CSV, FileExtension.CSV
        return FileFormats.TEXT, FileExtension.TEXT

    def _resolve_output_dir(self, path: str, org_id: int) -> str:
        """Determine valid output directory."""
        path_validator = PathValidator()
        if path_validator.is_directory(path):
            return path
        return os.path.join(config.CLARIDATA_DIR_PATH, str(org_id), current_date_trim())

    def _build_final_response(self, req: Any, generated_filepath: str) -> Any:
        """Step 10: Build final structured response."""
        return (
            StructureRespProcessResponseBuilder(req)
            .with_inference_result_file_path(generated_filepath)
            .with_status(Status.COMPLETED)
            .with_message("File generated successfully")
            .build()
        )
