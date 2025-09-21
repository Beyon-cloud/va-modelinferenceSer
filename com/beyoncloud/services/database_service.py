# com/beyoncloud/services/database_service.py

import logging
from typing import Dict, Any, List, Optional

#from com.beyoncloud.schemas.website_models import WebsiteData
from com.beyoncloud.repositories.postgres_repository import PostgreSqlRepository
from com.beyoncloud.utils.builder_utils import BuilderUtils
from com.beyoncloud.utils.filter_utils import FilterUtils
from com.beyoncloud.common.constants import Status, CommonPatterns,DBConstants
from com.beyoncloud.db.sql_db_connectivity import PostgreSqlConnectivity
from com.beyoncloud.config.settings.table_mapper_config import load_table_mapping
from com.beyoncloud.schemas.prompt_gen_reqres_datamodel import (
    SchemaPromptRequest,
    SchemaPromptResponse, 
    EntityPromptRequest, 
    EntityPromptResponse
)

pg_conn = PostgreSqlConnectivity()
mapper = load_table_mapping()
logger = logging.getLogger(__name__)


class DataBaseService:
    """
    Singleton service for handling database operations (upsert and update).
    Lazily initializes repositories after automap.
    Uses logical field names; repository handles schema mapping.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._inf_schema_promp_log_repo = None
            cls._instance._mt_bc_prompt_param_repo = None
        return cls._instance

    def _ensure_repos(self):
        """Initialize repos only when needed."""
        if self._inf_schema_promp_log_repo is None:
            model = self._get_db_model(CommonPatterns.SCHEMA1, DBConstants.INF_SCH_PROMPT)
            self._inf_schema_promp_log_repo = PostgreSqlRepository(model,DBConstants.INF_SCH_PROMPT)

        if self._mt_bc_prompt_param_repo is None:
            model = self._get_db_model(CommonPatterns.SCHEMA1, DBConstants.MST_PROMPT_PARAM)
            self._mt_bc_prompt_param_repo = PostgreSqlRepository(model,DBConstants.MST_PROMPT_PARAM)

    # ---------- MODEL & COLUMN RESOLUTION ----------
    def _get_db_model(self, schema: str, table_key: str):
        """Get automapped model class by logical table name."""
        try:
            table_name = mapper.get_db_table_name(schema, table_key).lower()
            return getattr(pg_conn.Base.classes, table_name)
        except (KeyError, AttributeError):
            available = list(pg_conn.Base.classes.keys())
            raise RuntimeError(
                f"Automap could not find class for table '{table_key}' "
                f"(mapped to '{table_name}'). Available: {available}"
            )

    # ---------- MAIN METHODS ----------

    async def save_or_update_schema_prompt_log(self, schema_prompt_request: SchemaPromptRequest,
        oth_val: Dict[str, Any]
    ) -> Optional[int]:
        """
        Upsert DataProcessingSource record.
        Conflicts on (org_id, domain_id, external_id, schema_id)
        """
        print("Inside -- save_or_update_schema_prompt_log")
        self._ensure_repos()
        print("Inside -- save_or_update_schema_prompt_log1")

        try:
            data_src = BuilderUtils.build_db_schema_prompt(schema_prompt_request, oth_val)
            print(f"data_src -- {data_src}")
            result = await self._inf_schema_promp_log_repo.upsert(
                data_obj=data_src,
                conflict_fields=["schema_id"],
                update_fields=["status"],
                return_field="uuid_id"
            )
            print(f"result -- {result}")
            return result

        except Exception as e:
            logger.error(
                "Failed to upsert dps_src for payload=%s, status=%s",
                schema_prompt_request, oth_val, exc_info=True
            )
            raise

    async def save_or_update_dp_src(self, payload: Dict[str, Any], status: str = Status.SUBMITTED) -> Optional[int]:
        """
        Upsert DataProcessingSource record.
        Conflicts on (org_id, domain_id, external_id, source_details)
        """
        self._ensure_repos()

        try:
            data_src = BuilderUtils.build_dp_src(payload, status)

            result = await self._inf_schema_promp_log_repo.upsert(
                data_obj=data_src,
                conflict_fields=["id"],
                update_fields=["status", "source_description"],
                return_field="external_id"
            )
            return result

        except Exception as e:
            logger.error(
                "Failed to upsert dps_src for payload=%s, status=%s",
                payload, status, exc_info=True
            )
            raise

    async def select_prompt_param(self, prompt_id: int):
        """
        Select Prompt param details.
        Returns the param key and value.
        """
        print("select_prompt_param")
        self._ensure_repos()

        filters = [
            FilterUtils.eq("prompt_id", prompt_id)
        ]
        print("select_prompt_param1")
        result = await self._mt_bc_prompt_param_repo.find_by_filters(
            filters=filters,
            order_by=None,
            limit=None
        )
        return result


    async def update_dps_by_src(self, src_details: str, **fields) -> int:
        """
        Update DataProcessingSource by source_details.
        Returns the updated record's ID.
        """
        self._ensure_repos()

        filters = [FilterUtils.eq("source_details", src_details)]

        result = await self._inf_schema_promp_log_repo.update_by_filters(
            filters=filters,
            update_fields=fields,
            return_field="id",
            return_all=False
        )
        return result
