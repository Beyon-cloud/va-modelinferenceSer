import os
from pydantic import BaseModel
from typing import List, Dict, Any
import json
from com.beyoncloud.db.postgresql_impl import PostgresSqlImpl
from langchain.prompts import PromptTemplate
import com.beyoncloud.config.settings.env_config as config
from com.beyoncloud.utils.file_util import JsonLoader
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from textwrap import dedent
from com.beyoncloud.common.constants import CommonPatterns
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

async def generatePromptJson():
    # Fetch records and build JSON structure
    postgresSqlImpl = PostgresSqlImpl()
    lstPromptRecord = await postgresSqlImpl.query_select("prompt_query")

    # Hierarchy: org_id → prompt_typ → domain → task_id → TaskDetail
    final_output: Dict[str, Dict[str, Dict[str, Dict[str, TaskDetail]]]] = {}

    for row in lstPromptRecord:
        prompt_id = row["prompt_id"]
        org_id = row["org_id"] or ""
        prompt_typ = row["prompt_typ"] or ""
        domain = row["domain_id"] or ""
        task_id = row["task_id"] or ""
        template = row["template"] or ""
        sys_tpl = row["sys_tpl"] or ""
        usr_tpl = row["usr_tpl"] or ""
        out_typ = row["out_typ"] or ""

        input_vars_list = []
        if row["input_variables"]:
            input_vars_list = [v.strip() for v in row["input_variables"].split(",")]

        # Ensure org_id level
        if org_id not in final_output:
            final_output[org_id] = {}

        # Ensure prompt_typ level
        if prompt_typ not in final_output[org_id]:
            final_output[org_id][prompt_typ] = {}

        # Ensure domain level
        if domain not in final_output[org_id][prompt_typ]:
            final_output[org_id][prompt_typ][domain] = {}

        # Assign TaskDetail
        final_output[org_id][prompt_typ][domain][task_id] = TaskDetail(
            template=template,
            input_variables=input_vars_list,
            sys_template=sys_tpl,
            usr_template=usr_tpl,
            out_typ=out_typ,
            prompt_id = prompt_id
        )

    # Convert Pydantic objects to JSON
    jsonOutput = json.dumps(
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

    print(jsonOutput)

    # Save to json file
    os.makedirs(config.PROMPT_FOLDER_PATH, exist_ok=True)
    promptFilepath = os.path.join(config.PROMPT_FOLDER_PATH, config.PROMPT_FILENAME)
    with open(promptFilepath, "w") as f:
        f.write(jsonOutput)

    return promptFilepath

def get_prompt_template(domainCatg: str, task: str, org_id: int, prompt_typ: str ) -> Dict[str, Any]:

    json_loader = JsonLoader()
    prompt_file_path = os.path.join(config.PROMPT_FOLDER_PATH, config.PROMPT_FILENAME)
    prompt_data = json_loader.get_json_object(prompt_file_path)
    print(f"promptInputVariables --> {org_id} - {prompt_typ} - {domainCatg} - {task}")
    prompt_base_json = prompt_data.get(str(org_id), {}).get(prompt_typ, {}).get(domainCatg, {}).get(task, {})
    if not prompt_base_json:
        prompt_base_json = prompt_data.get(str(org_id), {}).get(prompt_typ, {}).get(CommonPatterns.ASTRICK, {}).get(CommonPatterns.ASTRICK, {})
    
    prompt_id = prompt_base_json.get("prompt_id")
    prompt_template = prompt_base_json.get("template")
    system_prompt_template = prompt_base_json.get("sys_template")
    user_prompt_template = prompt_base_json.get("usr_template")
    prompt_input_variables = prompt_base_json.get("input_variables")
    print(f"promptInputVariables --> {prompt_input_variables}")

    if not prompt_template:
        prompt_template = prompt_data.get(org_id, {}).get(prompt_typ, {}).get("general", {}).get("instruction", {}).get("template")
        #promptInputVariables = prompt_data.get(org_id, {}).get(prompt_typ, {}).get("general", {}).get("instruction", {}).get("input_variables")

    if prompt_template:
        prompt_template = prompt_template.replace("\\n", "\n")
    
    if system_prompt_template:
        system_prompt_template = system_prompt_template.replace("\\n", "\n")

    if user_prompt_template:
        user_prompt_template = user_prompt_template.replace("\\n", "\n")

    if system_prompt_template and user_prompt_template:
        customPrompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt_template),
            HumanMessagePromptTemplate.from_template(user_prompt_template)
        ])
    else:
        customPrompt = PromptTemplate(
            template=prompt_template,
            input_variables=prompt_input_variables
        )

    propmpt_output = {
        "prompt_id": prompt_id,
        "prompt": customPrompt,
        "input_variables": prompt_input_variables,
        "system_prompt_template": system_prompt_template,
        "user_prompt_template": user_prompt_template,
        "template": prompt_template
    }
    return propmpt_output


async def get_prompt_param(prompt_id: int):
    print(f"get_prompt_param --> {prompt_id}")
    db_service = DataBaseService()
    param_result = await db_service.select_prompt_param(prompt_id)
    return param_result

"""
def get_prompt_template(domainCatg: str, task: str) -> PromptTemplate:

    json_loader = JsonLoader()
    prompt_file_path = os.path.join(config.PROMPT_FOLDER_PATH, config.PROMPT_FILENAME)
    prompt_data = json_loader.get_json_object(prompt_file_path)

    promptTemplate = prompt_data.get(domainCatg, {}).get(task, {}).get("template")
    promptInputVariables = prompt_data.get(domainCatg, {}).get(task, {}).get("input_variables")

    if not promptTemplate:
        promptTemplate = prompt_data.get("general", {}).get("instruction", {}).get("template")
        promptInputVariables = prompt_data.get("general", {}).get("instruction", {}).get("input_variables")

    if promptTemplate:
        promptTemplate = promptTemplate.replace("\\n", "\n")

    customPrompt = PromptTemplate(
        template=promptTemplate,
        input_variables=promptInputVariables
    )
    return customPrompt
"""

def get_prompt_input_variables(domainCatg: str, task: str) -> List[str]:
    
    json_loader = JsonLoader()
    prompt_file_path = os.path.join(config.PROMPT_FOLDER_PATH, config.PROMPT_FILENAME)
    prompt_data = json_loader.get_json_object(prompt_file_path)
    promptInputVariables = prompt_data.get(domainCatg, {}).get(task, {}).get("input_variables")
    if not promptInputVariables:
        promptInputVariables = prompt_data.get("general", {}).get("instruction", {}).get("input_variables")
    return promptInputVariables

def build_prompt(context: str, query: str, is_instruct: bool, tokenizer=None) -> str:
    if is_instruct and tokenizer:
        """
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are an intelligent assistant."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        """
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
        """
        return (
            f"Instruction: Use the following context to answer the user's question.\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        """
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

def build_universal_prompt(
    task="Answer the question", 
    query="", 
    context=[], 
    top_k=5, 
    answer_style="one short paragraph",
    language="English", 
    max_tokens=256, 
    today=""):

    print("Inside build_universal_prompt")

    SYSTEM = (
        "You are a precise retrieval-augmented assistant. "
        "Use ONLY the text in 'Retrieved Documents' as evidence. "
        "If the documents don’t contain the answer, reply exactly: Not Found. "
        "Prefer {language}. Be concise and factual. Do not invent or add outside facts. "
        "Cite minimal supporting spans with their doc_id(s)."
    )

    USER = (
        "Task: {task}\n"
        "Question: {query}\n\n"
        "Retrieved Documents (top-{top_k}):\n"
        "{context}\n"
        "# Each item should include: doc_id: <id>, text: <chunk>\n\n"
        "Constraints:\n"
        "- No external knowledge or speculation.\n"
        "- If evidence is missing or contradictory → \"Not Found\".\n"
        "- Keep within {max_tokens} tokens.\n\n"
        #"Output style: {answer_style}\n"
        "Output format (strictly follow this):\n"
        "Answer: <concise answer, with optional citations like (doc_id:x)>\n\n"
        "Date (optional): {today}\n"
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM),
        HumanMessagePromptTemplate.from_template(USER),
    ])

    output_context = ""
    if context:
        output_context = '"'
        for i, chunk in enumerate(context, start=1):
            output_context += f'doc_id: D{i}, text: {chunk}\\n'
        output_context = output_context.rstrip('\\n') + '"'

    # Example formatting
    filled = chat_prompt.format_messages(
        language=language,
        task=task,
        query=query,
        top_k=top_k,
        context=output_context,
        max_tokens=max_tokens,
        #answer_style=answer_style,
        today=today
    )

    return filled


def build_universal_prompt1(
    task="Answer the question", 
    query="", 
    context=[], 
    top_k=5, 
    answer_style="one short paragraph",
    language="English", 
    max_tokens=256, 
    today=""):

    

    # ----------------------
    # Prompt Template
    # ----------------------
    RAG_PROMPT_TEMPLATE = dedent("""
    You are a retrieval-augmented assistant. 
    Your task is to provide the most accurate response possible based strictly on the retrieved context.

    Guidelines:
    1. Use ONLY the information present in the "Retrieved Context". Do not invent or add outside facts.
    2. If the context does not contain the answer, reply exactly: "Not Found".
    3. Ensure the response directly addresses the user’s question.
    4. Present the answer in a clear, concise, and structured way (bullets, numbered lists, or short paragraphs).
    5. If multiple possible answers exist, include them all with supporting evidence.
    6. When document IDs or sources are available, cite them minimally (e.g., doc1, doc2).
    7. Never output raw context — always synthesize it into a clean answer.

    ---

    User Question:
    {query}

    Retrieved Context:
    {context}

    ---

    Answer:
    Provide the most accurate and context-grounded answer here.
    If no relevant information is found, output: "Not Found".
    """)

    # ----------------------
    # Example Parameters
    # ----------------------
    output_context = ""
    if context:
        #output_context = '"'
        #for i, chunk in enumerate(context, start=1):
        #    output_context += f'doc{i}: {chunk}\\n'
        #output_context = output_context.rstrip('\\n') + '"'
        output_context = '"'
        for i, chunk in enumerate(context, start=1):
            output_context += f'{chunk}\\n'
        output_context = output_context.rstrip('\\n') + '"'

    # ----------------------
    # Fill the template
    # ----------------------
    final_prompt = RAG_PROMPT_TEMPLATE.format(
        query=query,
        context=output_context
    )

    #print(f" build_universal_prompt1 --> final_prompt - {final_prompt}")
    return final_prompt

def get_json_extraction_prompt(document_text: str, prompt_type: str) -> str:

    prompt = ""
    if prompt_type=="prop_doc":
        prompt = get_json_propdoc_prompt(document_text)
    elif prompt_type=="cert_doc":
        prompt = get_json_certificate_prompt(document_text)
    elif prompt_type=="rent_doc":
        prompt = get_lease_agreement_prompt(document_text)
    elif prompt_type=="SCHEMA":
        prompt = get_schema_prompt()
    else:
        prompt = get_json_certificate_prompt(document_text)

    return prompt

def get_json_propdoc_prompt(document_text: str):

    template = """
    You are an AI assistant that extracts structured data from legal documents.  
    Your task is to carefully analyze the provided text and return only a valid JSON object following the schema below.  
    Ensure all values are extracted exactly as they appear, normalized when necessary (e.g., numbers as integers, dates in DD/MM/YYYY).  
    If a field is missing in the text, set it to null.

    Schema:
    {schema}

    Text:
    <<<{context}>>>

    Output:
    Return only the JSON object, no explanation or extra text.
    """

    schema = """
    {{
      "document_type": "string",
      "document_details": {{
        "value": "  ",
        "currency": "string",
        "amount": "number",
        "amount_words": "string",
        "serial_no": "string",
        "date": "string",
        "state": "string"
      }},
      "applicant": {{
        "name": "string",
        "father_name": "string",
        "designation": "string",
        "organization": "string",
        "age": "number",
        "address": {{
          "village": "string",
          "post_office": "string",
          "district": "string",
          "pin": "string",
          "state": "string"
        }}
      }},
      "application_purpose": {{
        "authority": "string",
        "location": "string",
        "purpose": "string",
        "course": "string",
        "intake": "number"
      }},
      "land_details": {{
        "total_area_sqm": "number",
        "address": {{
          "village": "string",
          "post_office": "string",
          "district": "string",
          "pin": "string"
        }},
        "plot_no": ["string"],
        "khasra_no": "string",
        "ownership": "string",
        "boundaries": {{
          "north": "string",
          "south": "string",
          "east": "string",
          "west": "string"
        }}
      }},
      "registration": {{
        "office": "string",
        "district": "string",
        "date": "string"
      }},
      "verification": {{
        "notary_name": "string",
        "notary_regn_no": "string",
        "designation": "string",
        "court": "string",
        "room_no": "string",
        "location": "string",
        "signature_present": "boolean",
        "date": "string",
        "seal_present": "boolean"
      }}
    }}
    """
    return PromptTemplate(
        input_variables=["context"],
        template=template,
        partial_variables={"schema": schema.strip()}
    )

def get_json_propdoc_prompt1(document_text: str) -> str:

    prompt = f"""
    You are an AI assistant that extracts structured data from legal documents.  
    Your task is to carefully analyze the provided text and return only a valid JSON object following the schema below.  
    Ensure all values are extracted exactly as they appear, normalized when necessary (e.g., numbers as integers, dates in DD/MM/YYYY).  
    If a field is missing in the text, set it to null.

    Schema:
    {{
      "document_type": "string",
      "document_details": {{
        "value": "  ",
        "currency": "string",
        "amount": "number",
        "amount_words": "string",
        "serial_no": "string",
        "date": "string",
        "state": "string"
      }},
      "applicant": {{
        "name": "string",
        "father_name": "string",
        "designation": "string",
        "organization": "string",
        "age": "number",
        "address": {{
          "village": "string",
          "post_office": "string",
          "district": "string",
          "pin": "string",
          "state": "string"
        }}
      }},
      "application_purpose": {{
        "authority": "string",
        "location": "string",
        "purpose": "string",
        "course": "string",
        "intake": "number"
      }},
      "land_details": {{
        "total_area_sqm": "number",
        "address": {{
          "village": "string",
          "post_office": "string",
          "district": "string",
          "pin": "string"
        }},
        "plot_no": ["string"],
        "khasra_no": "string",
        "ownership": "string",
        "boundaries": {{
          "north": "string",
          "south": "string",
          "east": "string",
          "west": "string"
        }}
      }},
      "registration": {{
        "office": "string",
        "district": "string",
        "date": "string"
      }},
      "verification": {{
        "notary_name": "string",
        "notary_regn_no": "string",
        "designation": "string",
        "court": "string",
        "room_no": "string",
        "location": "string",
        "signature_present": "boolean",
        "date": "string",
        "seal_present": "boolean"
      }}
    }}

    Text:
    <<<{document_text}>>>

    Output:
    Return only the JSON object, no explanation or extra text.
    """
    return prompt

def get_json_certificate_prompt(document_text: str) -> str:

    prompt = f"""
    You are an AI assistant that extracts structured data from academic certificates.  
    Return only a valid JSON object following the schema below.  
    If a field is missing, set it to null. Dates must be in DD/MM/YYYY format.

    Schema:
    {{
      "certificate_type": "string", 
      "certificate_title": "string", 
      "recipient": {{
        "name": "string"
      }},
      "degree_or_course": {{
        "degree": "string",
        "field_of_study": "string",
        "program_type": "string", 
        "class_or_grade": "string",
        "duration_or_hours": "string"
      }},
      "issuing_institution": {{
        "university_or_institute": "string",
        "faculty_or_department": "string",
        "location": "string"
      }},
      "certificate_details": {{
        "certificate_id": "string",
        "registration_number": "string",
        "university_seat_number": "string",
        "date_of_issue": "string"
      }},
      "signatories": [
        {{
          "name": "string",
          "designation": "string"
        }}
      ],
      "seal_or_logo_present": "boolean"
    }}

    Text:
    <<<{document_text}>>>

    Output:
    Return only the JSON object, no explanation or extra text.
    """
    return prompt


def get_lease_agreement_prompt(document_text: str):
    """
    Prompt for extracting structured JSON fields from Lease/Rental Deed Agreement text
    """

    return ChatPromptTemplate.from_messages([
        ("system", 
         "You are a precise contract extraction assistant. "
         "Your task is to carefully read the provided Lease/Rental Deed Agreement text "
         "and extract structured information in JSON format. "
         "Always return well-formed JSON. Do not add extra commentary."),
        
        ("human", 
         """Extract the following fields from the lease/rental deed agreement text:

        Required JSON schema:
        {{
          "document_type": "string",
          "document_details": {{
            "value": "  ",
            "currency": "string",
            "amount": "number",
            "amount_words": "string",
            "serial_no": "string",
            "date": "string",
            "state": "string"
          }},
          "lease_details": {{
              "agreement_type": "string (Lease / Rental Deed)",
              "agreement_date": "string (YYYY-MM-DD if available, else raw text)",
              "registration_number": "string",
              "jurisdiction": "string (city/court/registration office)",
              "start_date": "string",
              "end_date": "string",
              "duration_months": "integer",
              "renewal_terms": "string"
          }},
          "landlord": {{
              "name": "string",
              "address": "string",
              "contact": "string",
              "id_proof": "string",
              "pan_or_tax_id": "string",
              "bank_account_details": "string"
          }},
          "tenant": {{
              "name": "string",
              "address": "string",
              "contact": "string",
              "id_proof": "string",
              "pan_or_tax_id": "string",
              "emergency_contact": "string"
          }},
          "property": {{
              "address": "string",
              "description": "string (size, type, boundaries, etc.)",
              "furnishings": "string (furnished/semi/unfurnished, list of items)",
              "utilities_included": ["list of utilities included"],
              "restrictions": ["list of restrictions if any (pets, subletting, etc.)"]
          }},
          "rent_terms": {{
              "monthly_rent": "string",
              "rent_increment_policy": "string",
              "security_deposit": "string",
              "deposit_refund_terms": "string",
              "payment_due_date": "string",
              "payment_method": "string",
              "late_fee_policy": "string"
          }},
          "termination_clauses": {{
              "notice_period": "string",
              "early_termination_penalty": "string",
              "default_conditions": "string"
          }},
          "other_clauses": [
              "list of additional clauses (dispute resolution, arbitration, force majeure, etc.)"
          ],
          "signatures": {{
              "landlord_signature": "string (yes/no or name)",
              "tenant_signature": "string (yes/no or name)",
              "witnesses": [
                  "list of witness names if available"
              ],
              "notary_signature": "string"
          }}
        }}

        Agreement Text:
        \"\"\"{context}\"\"\" 

        Return only valid JSON following the schema.
      """)
    ])


def get_schema_prompt6() -> ChatPromptTemplate:
    """
    Generates a ChatPromptTemplate for directly extracting all available
    entities from a document and formatting them as a JSON object.
    """
    # The system prompt is designed to directly extract and format the data.
    SYSTEM = (
        "You are an expert at extracting structured information from any given document."
        "Your task is to read the document and extract all available entities into a valid JSON object."
        "The output must be a valid JSON object with the extracted data. Do not include a schema or any additional text, explanations, or code fences—return only the JSON."
    )

    # The user prompt provides the context of the document from which to extract data.
    USER = (
        "Extract information from the following document:"
        "\"\"\"{{context}}\"\"\""
    )
    
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM),
        HumanMessagePromptTemplate.from_template(USER),
    ])
    
    return chat_prompt

def get_schema_prompt5() -> ChatPromptTemplate:
    """
    Generates a ChatPromptTemplate for creating a structured schema
    for data extraction from any type of document.
    """

    document_type = "Rental or Lease Document"
    output_type = "JSON"

    # The system prompt is now generic and accepts the document and output types as parameters.
    SYSTEM = (
        f"You are an AI that generates structured {output_type} schemas from {document_type} text."
        f"Your task is to identify all available entities in the document and represent them in the schema."
        f"Return only a valid {output_type} schema with a \"title\", \"type\", and \"properties\" key. "
        f"Do not include any additional explanation, only the {output_type}."
    )

    # The user prompt is also generic to accept any document context.
    USER = (
        "Generate a schema for the following document:"
        "\"\"\"{{context}}\"\"\""
    )
    
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM),
        HumanMessagePromptTemplate.from_template(USER),
    ])
    
    return chat_prompt

def get_schema_prompt() -> str:

    SYSTEM = (
        "You are an AI that generates JSON Schema templates from legal or official document text. "
        "The input may be a Rental Agreement, Lease Agreement, Property Deed, Non-Judicial Stamp Paper, "
        "Affidavit, Educational Certificate, or other official document. "
        "Your task is to carefully analyze the input and design a JSON Schema that captures ALL available "
        "entities mentioned in the document. "
        "Always return only a valid JSON Schema with 'title', 'type', and 'properties'. "
        "No explanation, only JSON."
    )

    # Enhanced USER prompt:
    USER = (
        "Generate a JSON Schema for the following document text. "
        "Infer an appropriate 'title' from the document type or content (e.g., 'RentalAgreement', 'PropertyDeed'). "
        "Use standard JSON Schema draft-07/draft-2019-09 conventions: include 'title'and 'type'. "
        "For each property, provide an appropriate 'type' (string, integer, number, boolean, object, array). "
        "When a field can repeat (e.g., multiple tenants, witnesses), model it as an 'array' with an 'items' schema. "
        "Mark fields as 'required' when they are explicitly present in the document. "
        #"Prefer helpful 'format' or 'pattern' annotations where applicable (e.g., 'date' for dates, 'email', 'uri', or a 'pattern' for identifiers and amounts). "
        "Do not invent values or add sample values—only define the schema shape and constraints inferred from the text. "
        "If a field's type is ambiguous, choose the most general appropriate type and note optional formats via 'format' or 'pattern'. "
        "Return strictly valid JSON (no comments, no extra text and no revised version). "
        "\n\nDocument Text:\n\"\"\"{context}\"\"\""
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM),
        HumanMessagePromptTemplate.from_template(USER),
    ])

    return chat_prompt


def get_schema_prompt2() -> str:

    document_type="Rental or Lease Document"
    output_typr="JSON"
    SYSTEM = (
        f"You are an AI that generates {output_typr} Schema templates from {document_type} text."
        f"Return only a valid {output_typr} schema with \"title\", \"type\", and \"properties\". No explanation, only {output_typr}."
    )

    USER = (
        "Generate a schema for the following document:"
        "\"\"\"{{context}}\"\"\""
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM),
        HumanMessagePromptTemplate.from_template(USER),
    ])

    return chat_prompt

def get_schema_prompt1() -> str:

    template = """
    You are an expert in creating document information extraction prompts.

    Task:
    1. Read the provided document text carefully.
    2. Identify the type of document (lease agreement, rental agreement, property deed, certificate, affidavit, etc.).
    3. Design a **system + human prompt template** for an LLM that will be used to extract structured data from documents of this type.
    4. The output must strictly follow the format below:

    (
      "system",
      "You are a precise [document type] extraction assistant. "
      "Your task is to carefully read the provided text and extract structured information in JSON format. "
      "Always return well-formed JSON. Do not add extra commentary."
    ),

    (
      "human",
      \"\"\"Extract the following fields from the [document type] text:

      Required JSON schema:
      {{
         ... (define fields relevant to this document type here) ...
      }}

      Document Text:
      \"\"\"{{input_text}}\"\"\"

      Return only valid JSON following the schema.
    \"\"\"
    )

    Rules:
    - Adapt the schema fields to match the entities present in the input document.
    - Use nested JSON objects for logical grouping (e.g., parties, property, registration, notary).
    - Include only fields relevant to this document type.
    - Do not fill values, only design the schema.

    INPUT DOCUMENT:
    \"\"\"
    {context}
    \"\"\"

    OUTPUT:
    Return only the system + human prompt template as shown above.
    """

    return PromptTemplate(
        input_variables=["context"],
        template=template
    )

def get_schema(prompts: Dict[str, str]) -> str:

    SYSTEM=prompts["system_prompt"]
    USER=prompts["user_prompt"]

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM),
        HumanMessagePromptTemplate.from_template(USER),
    ])

    return chat_prompt