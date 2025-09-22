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
from com.beyoncloud.services.database_service import DataBaseService
from com.beyoncloud.common.constants import PromptType, FileExtension, FileFormats, Status, CommonPatterns
from com.beyoncloud.utils.file_util import FetchContent, FileCreation, TextLoader
from com.beyoncloud.utils.date_utils import current_date_trim
import com.beyoncloud.config.settings.env_config as config

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
            logger.info(f"OCR result file path is empty")
            raise ValueError("OCR result file path is empty")

        fetch_content = FetchContent()
        file_context = CommonPatterns.EMPTY_SPACE
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
        starttime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        ragProcessImpl = RagProcessImpl()
        response = await ragProcessImpl.generate_structured_response(structure_input_data)
        endtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        logger.info("Process taking time from '"+starttime+"' to '"+endtime+"'")

        # Step 4: Clean response and extract specific file content
        cleaned_response = fetch_content.fetch_schema_content(response, structure_resp_process_request.desired_output_format)
        print(f"cleaned_response --> {cleaned_response}")

        # Step 5: Generate output file
        file_extension = FileExtension.TEXT
        if structure_resp_process_request.desired_output_format == FileFormats.JSON:
            file_extension = FileExtension.JSON
        elif structure_resp_process_request.desired_output_format == FileFormats.CSV:
            file_extension = FileExtension.CSV
        elif structure_resp_process_request.desired_output_format == FileFormats.XLSX:
            file_extension = FileExtension.XLSX

        file_creation = FileCreation()
        output_filename = file_creation.generate_file_name(
            structure_resp_process_request.output_filename,
            config.CLARIDATA_FILENAME,
            structure_resp_process_request.document_batch_id,
            structure_resp_process_request.request_reference_id,
            file_extension
        )

        output_dir_path = structure_resp_process_request.inference_result_storage_path
        if not output_dir_path:
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
