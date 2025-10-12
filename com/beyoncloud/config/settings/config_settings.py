import logging
from typing import List, Dict, Any, Optional

from com.beyoncloud.config.templates.template_loader import TemplateLoader
from com.beyoncloud.common.constants import Numerical, CommonPatterns
from com.beyoncloud.config.settings import env_config

logger = logging.getLogger(__name__)


class ConfigSettings:
    """
    Provides convenient accessors for configuration templates
    (e.g., ID definitions, domain definitions, gender definitions).

    Author: Balaji G.R (01-09-2025)
    """

    def __init__(self) -> None:
        self._template_loader = TemplateLoader()

    async def get_id_definitions(self) -> List[Dict[str, Any]]:
        """
        Load all ID definition configuration templates.
        """
        template_name = env_config.MT_ID_DFN_CONFIG_TEMPLATE
        logger.debug("Fetching ID definition config: %s", template_name)
        return await self._template_loader.get_template(template_name)

    async def get_id_definitions_by_type(
        self, dfn_type: str, org_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get ID definitions filtered by dfn_type and org_id.
        """
        records = await self.get_id_definitions()
        filtered = [
            rec
            for rec in records
            if rec.get("dfn_type") == dfn_type and (rec.get("org_id") == org_id or rec.get("org_id") == Numerical.ZERO)
        ]
        logger.info("Found %d records for type '%s'", len(filtered), dfn_type)
        return filtered

    async def get_value_by_type_and_id(
        self, dfn_type: str, dfn_id: str, org_id: int
    ) -> Optional[str]:
        """
        Get the `dfn_value` for a given dfn_type + dfn_id + org_id.
        """
        records = await self.get_id_definitions_by_type(dfn_type, org_id)
        for rec in records:
            if rec.get("dfn_id") == dfn_id:
                return rec.get("dfn_value")

        logger.warning(
            "No matching dfn_value found for dfn_type=%s, dfn_id=%s, org_id=%s",
            dfn_type,
            dfn_id,
            org_id,
        )
        return None

    async def get_template(self, template_name: str) -> List[Dict[str, Any]]:
        """
        Generic template fetcher by name.
        """
        logger.debug("Fetching template: %s", template_name)
        return await self._template_loader.get_template(template_name)

    async def get_mt_org_prompt(self) -> List[Dict[str, Any]]:
        """
        Load all ID definition configuration templates.
        """
        template_name = env_config.MT_ORG_PROMPT_CONFIG_TEMPLATE
        logger.debug("Fetching ID definition config: %s", template_name)
        return await self._template_loader.get_template(template_name)

    async def get_prompt_template(
        self,
        domain_id: str, 
        document_typ: str, 
        org_id: int, 
        prompt_typ: str, 
        out_typ: str = None 
    ) -> Dict[str, Any]:

        records = await self.get_mt_org_prompt()

        filtered = [
            rec
            for rec in records
            if rec.get("prompt_typ") == prompt_typ 
            and (rec.get("org_id") == org_id or rec.get("org_id") == Numerical.ZERO)
            and rec.get("domain_id") == domain_id 
            and rec.get("document_typ") == document_typ 
            and rec.get("out_typ") == out_typ
        ]
        
        if not filtered:
            filtered = [
                rec
                for rec in records
                if rec.get("prompt_typ") == prompt_typ 
                and (rec.get("org_id") == org_id or rec.get("org_id") == Numerical.ZERO)
                and rec.get("domain_id") == CommonPatterns.ASTRICK 
                and rec.get("document_typ") == CommonPatterns.ASTRICK 
                and rec.get("out_typ") == out_typ
            ]
        logger.info("Found %d records for type '%s'", len(filtered), prompt_typ)
        
        prompt_tmpl_config = {}
        if filtered:
            for rec in filtered:
                print(f"prompt_id --> {rec.get("prompt_id")}")
                print(f"sys_tpl --> {rec.get("sys_tpl")}")
                print(f"usr_tpl --> {rec.get("usr_tpl")}")
                print(f"mt_org_prompt_param --> {rec.get("mt_org_prompt_param")}")
                
                lst_param = rec.get("mt_org_prompt_param")
                prompt_param = self._get_prompt_param(lst_param)

                prompt_tmpl_config = {
                    "prompt_id": rec.get("prompt_id"),
                    "input_variables": rec.get("input_variables"),
                    "system_prompt_template": rec.get("sys_tpl"),
                    "user_prompt_template": rec.get("usr_tpl"),
                    "template": rec.get("template"),
                    "prompt_param": prompt_param
                }
                break

        if not prompt_tmpl_config:
            logger.warning(
                "No matching prompt found for prompt_typ=%s, org_id=%s, domain_id=%s, document_typ=%s, out_typ=%s",
                prompt_typ,
                org_id,
                domain_id,
                document_typ,
                out_typ
            )

        return prompt_tmpl_config

    def _get_prompt_param(self, prompt_param_list: Any) -> Dict[str,str]:

        prompt_param = {
                    p["param_key"]: p.get("param_value")
                    for p in prompt_param_list or []
                    if p.get("param_key") is not None
                }
        return prompt_param
