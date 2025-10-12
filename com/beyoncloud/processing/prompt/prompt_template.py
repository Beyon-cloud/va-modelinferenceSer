import os
import asyncio
from pydantic import BaseModel
from typing import List, Dict, Any
import json
from com.beyoncloud.db.postgresql_impl import PostgresSqlImpl
from langchain.prompts import PromptTemplate
import com.beyoncloud.config.settings.env_config as config
from com.beyoncloud.config.settings.config_settings import ConfigSettings
from com.beyoncloud.utils.file_util import JsonLoader
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from com.beyoncloud.common.constants import CommonPatterns, PromptType
from com.beyoncloud.services.database_service import DataBaseService


class TaskDetail(BaseModel):
    template: str
    input_variables: List[str]
    sys_template: str
    usr_template: str
    out_typ: str
    prompt_id: int


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"   # keep placeholder as-is


# ---------- Helper Functions for generate_prompt_template_json ----------

def _parse_input_variables(row: Dict[str, Any]) -> List[str]:
    if row.get("input_variables"):
        return [v.strip() for v in row["input_variables"].split(",")]
    return []


def _ensure_nested_keys(final_output: Dict, org_id: str, prompt_typ: str, domain: str) -> None:
    final_output.setdefault(org_id, {}).setdefault(prompt_typ, {}).setdefault(domain, {})


def _convert_to_json(final_output: Dict[str, Any]) -> str:
    return json.dumps(
        {
            org: {
                ptyp: {
                    dom: {tid: detail.dict() for tid, detail in dom_map.items()}
                    for dom, dom_map in ptyp_map.items()
                }
                for ptyp, ptyp_map in org_map.items()
            }
            for org, org_map in final_output.items()
        },
        indent=2,
    )


async def generate_prompt_template_json():
    postgres_sql_impl = PostgresSqlImpl()
    lst_prompt_record = await postgres_sql_impl.query_select("prompt_query")

    final_output: Dict[str, Dict[str, Dict[str, Dict[str, TaskDetail]]]] = {}

    for row in lst_prompt_record:
        org_id = row.get("org_id") or ""
        prompt_typ = row.get("prompt_typ") or ""
        domain = row.get("domain_id") or ""
        task_id = row.get("task_id") or ""

        _ensure_nested_keys(final_output, org_id, prompt_typ, domain)

        final_output[org_id][prompt_typ][domain][task_id] = TaskDetail(
            template=row.get("template") or "",
            input_variables=_parse_input_variables(row),
            sys_template=row.get("sys_tpl") or "",
            usr_template=row.get("usr_tpl") or "",
            out_typ=row.get("out_typ") or "",
            prompt_id=row.get("prompt_id"),
        )

    json_output = _convert_to_json(final_output)
    print(f"json_output -----> {json_output}")

    os.makedirs(config.PROMPT_FOLDER_PATH, exist_ok=True)
    prompt_filepath = os.path.join(config.PROMPT_FOLDER_PATH, config.PROMPT_FILENAME)

    def write_sync(path, content):
        with open(path, "w") as f:
            f.write(content)

    await asyncio.get_event_loop().run_in_executor(None, write_sync, prompt_filepath, json_output)
    return prompt_filepath


# ---------- Helper Functions for get_prompt_template ----------

def _get_prompt_data(org_id: int, prompt_typ: str, domain_id: str, task: str, json_loader: JsonLoader) -> Dict[str, Any]:
    prompt_file_path = os.path.join(config.PROMPT_FOLDER_PATH, config.PROMPT_FILENAME)
    print(f"prompt_file_path --> {prompt_file_path}")
    prompt_data = json_loader.get_json_object(prompt_file_path)
    print(f"promptInputVariables --> {org_id} - {prompt_typ} - {domain_id} - {task}")

    prompt_base_json = (
        prompt_data.get(str(org_id), {}).get(prompt_typ, {}).get(domain_id, {}).get(task)
        or prompt_data.get(str(org_id), {}).get(prompt_typ, {}).get(CommonPatterns.ASTRICK, {}).get(CommonPatterns.ASTRICK, {})
    )
    return prompt_base_json


def _find_schema_prompt(prompt_base_json: Dict[str, Any], out_typ: str) -> Dict[str, Any]:
    if not (prompt_base_json and out_typ):
        return {}

    for v in prompt_base_json.values():
        if isinstance(v, dict) and v.get("out_typ") == out_typ:
            return v
    return {}


def _build_prompt_object(prompt_json: Dict[str, Any]) -> Dict[str, Any]:
    prompt_template = (prompt_json.get("template") or "").replace("\\n", "\n")
    system_prompt_template = (prompt_json.get("sys_template") or "").replace("\\n", "\n")
    user_prompt_template = (prompt_json.get("usr_template") or "").replace("\\n", "\n")
    input_vars = prompt_json.get("input_variables")
    prompt_id = prompt_json.get("prompt_id")

    if system_prompt_template and user_prompt_template:
        custom_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt_template),
            HumanMessagePromptTemplate.from_template(user_prompt_template)
        ])
    else:
        custom_prompt = PromptTemplate(
            template=prompt_template,
            input_variables=input_vars,
        )

    return {
        "prompt_id": prompt_id,
        "prompt": custom_prompt,
        "input_variables": input_vars,
        "system_prompt_template": system_prompt_template,
        "user_prompt_template": user_prompt_template,
        "template": prompt_template,
    }


def get_prompt_template_old(domain_id: str, task: str, org_id: int, prompt_typ: str, out_typ: str = None) -> Dict[str, Any]:
    json_loader = JsonLoader()
    prompt_base_json = _get_prompt_data(domain_id=domain_id, task=task, org_id=org_id, prompt_typ=prompt_typ, json_loader=json_loader)

    prompt_json = {}
    if prompt_typ == PromptType.SCHEMA_PROMPT and out_typ:
        prompt_json = _find_schema_prompt(prompt_base_json, out_typ)

    if not prompt_json:
        prompt_json = prompt_base_json or {}

    if not prompt_json.get("template"):
        prompt_data = json_loader.get_json_object(os.path.join(config.PROMPT_FOLDER_PATH, config.PROMPT_FILENAME))
        prompt_json["template"] = (
            prompt_data.get(org_id, {}).get(prompt_typ, {}).get("general", {}).get("instruction", {}).get("template")
        )

    print(f"promptInputVariables --> {prompt_json.get('input_variables')}")
    print(f"system_prompt_template --> {prompt_json.get('sys_template')}")
    print(f"user_prompt_template --> {prompt_json.get('usr_template')}")

    return _build_prompt_object(prompt_json)

async def get_prompt_template(
    domain_id: str, 
    document_typ: str, 
    org_id: int, 
    prompt_typ: str, 
    out_typ: str = None 
) -> Dict[str, Any]:

    config_settings = ConfigSettings()
    prompt_temp = await config_settings.get_prompt_template(domain_id,document_typ,org_id,prompt_typ,out_typ)
    print(f"prompt_temp --> {prompt_temp}")
    return prompt_temp

def get_temp_prompt_template(sys_prompt_tpl: str, usr_prompt_tpl: str, input_variables: str) -> Dict[str, Any]:

    if sys_prompt_tpl:
        system_prompt_template = sys_prompt_tpl.replace("\\n", "\n")

    if usr_prompt_tpl:
        user_prompt_template = usr_prompt_tpl.replace("\\n", "\n")

    input_vars_list = []
    if input_variables:
        input_vars_list = [v.strip() for v in input_variables.split(",")]

    if system_prompt_template and user_prompt_template:
        custom_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt_template),
            HumanMessagePromptTemplate.from_template(user_prompt_template)
        ])

    propmpt_output = {
        "prompt": custom_prompt,
        "system_prompt_template": system_prompt_template,
        "user_prompt_template": user_prompt_template,
        "input_variables": input_vars_list
    }
    return propmpt_output


async def get_prompt_param(prompt_id: int):
    print(f"get_prompt_param --> {prompt_id}")
    db_service = DataBaseService()
    param_result = await db_service.select_prompt_param(prompt_id)
    return param_result

def get_prompt_input_variables(domain_id: str, task: str) -> List[str]:
    
    json_loader = JsonLoader()
    prompt_file_path = os.path.join(config.PROMPT_FOLDER_PATH, config.PROMPT_FILENAME)
    prompt_data = json_loader.get_json_object(prompt_file_path)
    prompt_input_variables = prompt_data.get(domain_id, {}).get(task, {}).get("input_variables")
    if not prompt_input_variables:
        prompt_input_variables = prompt_data.get("general", {}).get("instruction", {}).get("input_variables")
    return prompt_input_variables

def build_prompt(context: str, query: str, is_instruct: bool, tokenizer=None) -> str:
    if is_instruct and tokenizer:
        return tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a highly knowledgeable and precise assistant. "
                        "Your goal is to provide accurate, concise, and contextually relevant answers. "
                        "Use the provided context to answer the user's query clearly and factually."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuery: {query}"
                }
            ],
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        return (
            f"You are a helpful AI assistant."
            f"You will be given retrieved documents and a user question."
            f"Use only the information from the retrieved documents to answer."
            f"If the documents do not contain the answer, say:"
            f"The retrieved documents do not contain the answer.\n\n"
            f"Retrieved Documents:"
            f"{context}\n\n"
            f"Question:"
            f"{query}\n\n"
            f"Answer:"
        )

def get_schema(prompts: Dict[str, str]) -> str:

    SYSTEM=prompts["system_prompt"]
    USER=prompts["user_prompt"]

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM),
        HumanMessagePromptTemplate.from_template(USER),
    ])

    return chat_prompt