import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from com.beyoncloud.common.constants import CommonPatterns, Numerical, Status
from com.beyoncloud.config.settings.table_mapper_config import load_table_mapping
from com.beyoncloud.db.db_connection import sql_db_connection
from com.beyoncloud.schemas.prompt_gen_reqres_datamodel import (
    SchemaPromptRequest,
    SchemaPromptResponse, 
    EntityPromptRequest, 
    EntityPromptResponse
)

from com.beyoncloud.model.db_models import (
    InfSchemaPromptLogBuilder
)

logger = logging.getLogger(__name__)

class BuilderUtils:
    """
    Utility class for constructing data processing entities and response objects.

    Author:
        Balaji G.R (24-07-2025)

    Description:
        Provides builder-based helper methods for:
        - Constructing database models (`DataProcessingSource`, `WebPageExecution`).
        - Building schema models (`TextDetails`, `HeaderDetails`, `LinkDetails`, etc.).
        - Generating standard job responses for API calls.
    """

    table_mapper = load_table_mapping()

    @staticmethod
    def build_db_schema_prompt(schema_prompt_request: SchemaPromptRequest, oth_val: Dict[str, Any]) -> dict:
        """
        Builds a inf_schema_prompt_log record from DataSource payload.
        Maps logical fields → DB columns using schema.
        """
        builder = InfSchemaPromptLogBuilder()

        return (
            builder
            .with_org_id(schema_prompt_request.organization_id)
            .with_domain_id(schema_prompt_request.domain_id)
            .with_ref_id(schema_prompt_request.request_reference_id)
            .with_src_path(schema_prompt_request.source_file_path)
            .with_lang(schema_prompt_request.source_lang)
            .with_doc_typ(oth_val["document_type"] or schema_prompt_request.document_type)
            .with_out_mode(schema_prompt_request.desired_output_mode)
            .with_out_format(schema_prompt_request.desired_output_format)
            .with_out_dir(oth_val["schema_dir_path"] or schema_prompt_request.output_directory)
            .with_out_filename(oth_val["schema_filename"] or schema_prompt_request.output_filename)
            .with_sys_gen_schema_prop(oth_val["schema_json"])
            .with_status(oth_val["status"])
            .with_created_by(schema_prompt_request.user_id)
            .with_created_at(datetime.utcnow())
            .build()
        )

    @staticmethod
    def build_rag_request(source_type: str, task_type: str, file_path: str):
        """Builds `DpsToRagRequest` protobuf message."""
        return pb.DpsToRagRequest(
            source_type=source_type,
            task_type=task_type,
            file_path=file_path,
            params={},
        )
