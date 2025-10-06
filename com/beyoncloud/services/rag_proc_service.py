import os
from datetime import datetime
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
    SchemaPromptResponse, 
    EntityPromptRequest, 
    EntityPromptResponse
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

logger = logging.getLogger(__name__)

async def rag_chat_process(ragReqDataModel: RagReqDataModel) -> RagRespDataModel:

    starttime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ragProcessImpl = RagProcessImpl()
    response = await ragProcessImpl.generateRAGResponse(ragReqDataModel)
    endtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
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
        print("Step 5 - completed")
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
        print("Step 5 - completed")

        # Step 6: Response datamodel formation
        structure_resp_process_response = (StructureRespProcessResponseBuilder(structure_resp_process_request)
                    .with_inference_result_file_path(generated_filepath)
                    .with_status(Status.COMPLETED)
                    .with_message("File generated successfully")
                    .build()
        )

        return structure_resp_process_response


    async def temp_structure_resp_process(
        self, 
        structure_resp_process_request: StructureRespProcessRequest
    ) -> StructureRespProcessResponse:

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
        print("Step 1 - completed")

        # Step 2: Form schema prompt request
        schema_prompt_request = SchemaPromptRequest(
            request_reference_id = structure_resp_process_request.request_reference_id or CommonPatterns.EMPTY_SPACE,
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
            ocr_result_file_path = structure_resp_process_request.ocr_result_file_path or CommonPatterns.EMPTY_SPACE
        )
        print("Step 2 - completed")

        # Step 3: Call schema prompt generation process
        schema_prompt_service = SchemaPromptService()
        schema_prompt_response = await schema_prompt_service.generate_schema_prompt(schema_prompt_request)
        print(f"Step 3 - completed --> {schema_prompt_response.schema_template_filepath}")

        # Step 4: Form entity prompt request
        entity_prompt_request = EntityPromptRequest(
            request_reference_id = structure_resp_process_request.request_reference_id or CommonPatterns.EMPTY_SPACE,
            organization_id = structure_resp_process_request.organization_id or Numerical.ZERO,
            domain_id = structure_resp_process_request.domain_id or CommonPatterns.EMPTY_SPACE,
            user_id = structure_resp_process_request.user_id or Numerical.ZERO,
            desired_output_mode = structure_resp_process_request.desired_output_mode or CommonPatterns.EMPTY_SPACE,
            desired_output_format = structure_resp_process_request.desired_output_format or CommonPatterns.EMPTY_SPACE,
            source_file_path = structure_resp_process_request.source_file_path or CommonPatterns.EMPTY_SPACE,
            source_lang = structure_resp_process_request.source_lang or CommonPatterns.EMPTY_SPACE,
            document_type = structure_resp_process_request.document_type or CommonPatterns.EMPTY_SPACE,
            schema_template_filepath = schema_prompt_response.schema_template_filepath or CommonPatterns.EMPTY_SPACE
        )
        print("Step 4 - completed")

        # Step 5: Call response prompt generation process
        entity_prompt_response = await schema_prompt_service.generate_entity_prompt(entity_prompt_request)
        print("Step 5 - completed")
        print(f"Step 5 - system prompt : {entity_prompt_response.system_prompt}")
        print(f"Step 5 - user prompt : {entity_prompt_response.user_prompt}")
        print(f"Step 5 - input variables : {entity_prompt_response.input_variables}")


        # Step 6: Build the entity request using the builder pattern, ensuring the builder returns expected shape.
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
        print("Step 6 - completed")

        # Step 7: Call inference model to get final response.
        rag_generator_process = RagGeneratorProcess()
        if config.ENABLE_HF_INFRENCE_YN == "Y":
            print("Step 7 - if")
            response = await rag_generator_process.temp_hf_response(
                structure_input_data,
                entity_prompt_response.system_prompt, 
                entity_prompt_response.user_prompt,
                entity_prompt_response.input_variables
            )
        else:
            print("Step 7 - else")
            #response = await rag_generator_process.generate_structured_response(structure_input_data)
            response= await rag_generator_process.temp_generate_structured_response(
                structure_input_data,
                entity_prompt_response.system_prompt, 
                entity_prompt_response.user_prompt,
                entity_prompt_response.input_variables
            )
        print("Step 7 - completed")

        # Step 8: Generate output file
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


        # Step 8: Clean response and extract specific file content
        cleaned_response = fetch_content.fetch_schema_content(response, final_output_format)
        print("Step 8.1 - completed")
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
        print("Step 9 - completed")

        # Step 10: Response datamodel formation
        structure_resp_process_response = (StructureRespProcessResponseBuilder(structure_resp_process_request)
                    .with_inference_result_file_path(generated_filepath)
                    .with_status(Status.COMPLETED)
                    .with_message("File generated successfully")
                    .build()
        )
        print("Step 10 - completed")

        return structure_resp_process_response
