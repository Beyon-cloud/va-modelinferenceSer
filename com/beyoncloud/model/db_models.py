from datetime import datetime, timezone
from typing import Optional, Dict
from com.beyoncloud.common.constants import Status, CommonConstants


#####################################
# BUILDER CLASSES
#####################################

class InfSchemaPromptLogBuilder:
    def __init__(self) -> None:
        self._obj = {}

    def with_schema_id(self, schema_id: int):
        self._obj["schema_id"] = schema_id
        return self

    def with_org_id(self, org_id: int):
        if org_id is None:
            raise ValueError("organization_id cannot be None")
        self._obj["org_id"] = org_id
        return self

    def with_domain_id(self, domain_id: str):
        if domain_id is None:
            raise ValueError("domain_id cannot be None")
        self._obj["domain_id"] = domain_id
        return self

    def with_external_id(self, external_id: Optional[str]):
        self._obj["uuid_id"] = external_id
        return self

    def with_ref_id(self, ref_id: Optional[str]):
        self._obj["ref_id"] = ref_id
        return self

    def with_src_path(self, src_path: Optional[str]):
        self._obj["src_path"] = src_path
        return self

    def with_lang(self, lang: Optional[str]):
        self._obj["lang"] = lang or CommonConstants.DFLT_LANG
        return self

    def with_doc_typ(self, doc_typ: Optional[str]):
        self._obj["doc_typ"] = doc_typ
        return self

    def with_out_mode(self, out_mode: Optional[str]):
        self._obj["out_mode"] = out_mode
        return self

    def with_out_format(self, out_format: Optional[str]):
        self._obj["out_format"] = out_format
        return self

    def with_out_dir(self, out_dir: Optional[str]):
        self._obj["out_dir"] = out_dir
        return self

    def with_out_filename(self, out_filename: Optional[str]):
        self._obj["out_filename"] = out_filename
        return self

    def with_sys_gen_schema_prop(self, sys_gen_schema_prop: Optional[str]):
        self._obj["sys_gen_schema_prop"] = sys_gen_schema_prop
        return self

    def with_res_sys_prompt(self, res_sys_prompt: Optional[str]):
        self._obj["res_sys_prompt"] = res_sys_prompt
        return self

    def with_res_usr_prompt(self, res_usr_prompt: Optional[str]):
        self._obj["res_usr_prompt"] = res_usr_prompt
        return self

    def with_res_inp_vars(self, res_inp_vars: Optional[str]):
        self._obj["res_inp_vars"] = res_inp_vars
        return self

    def with_status(self, status: Optional[str]):
        self._obj["status"] = status
        return self

    def with_created_by(self, user: Optional[str]):
        self._obj["created_by"] = user or CommonConstants.SYSTEM
        return self

    def with_created_at(self, dt: Optional[datetime] = None):
        aware_dt = dt or datetime.now(timezone.utc)
        self._obj["created_at"] = aware_dt.replace(tzinfo=None)
        return self

    def build(self) -> dict:
        """Finalize and validate the object"""
        if not self._obj.get("org_id"):
            raise ValueError("org_id is required")
        return self._obj

class RagProcessingSourceBuilder:
    def __init__(self) -> None:
        self._obj = {}

    def with_src_id(self, src_id: int):
        self._obj["src_id"] = src_id
        return self

    def with_org_id(self, org_id: int):
        if org_id is None:
            raise ValueError("organization_id cannot be None")
        self._obj["org_id"] = org_id
        return self

    def with_domain_id(self, domain_id: str):
        if domain_id is None:
            raise ValueError("domain_id cannot be None")
        self._obj["domain_id"] = domain_id
        return self

    def with_external_id(self, external_id: Optional[str]):
        self._obj["external_id"] = external_id
        return self

    def with_source_type(self, src_type: str):
        if not src_type:
            raise ValueError("source_type is required")
        self._obj["src_typ"] = src_type
        return self

    def with_source_details(self, src_detail: str):
        if not src_detail:
            raise ValueError("source_details is required")
        self._obj["src_dtls"] = src_detail
        return self

    def with_lang(self, lang: str):
        self._obj["lang"] = lang or CommonConstants.DFLT_LANG
        return self

    def with_title(self, title: Optional[str]):
        self._obj["title"] = title
        return self

    def with_status(self, status: str):
        self._obj["status"] = status or Status.WIP
        return self

    def with_metadata(self, metadata: Optional[Dict]):
        self._obj["metadata"] = metadata
        return self

    def with_published_at(self, dt: Optional[datetime] = None):
        aware_dt = dt or datetime.now(timezone.utc)
        self._obj["published_at"] = aware_dt.replace(tzinfo=None)
        return self

    def with_created_by(self, user: Optional[str]):
        self._obj["created_by"] = user or "Admin"
        return self

    def with_created_at(self, dt: Optional[datetime] = None):
        aware_dt = dt or datetime.now(timezone.utc)
        self._obj["created_at"] = aware_dt.replace(tzinfo=None)
        return self

    def build(self) -> dict:
        """Finalize and validate the object"""
        if not self._obj.get("org_id"):
            raise ValueError("org_id is required")
        return self._obj

