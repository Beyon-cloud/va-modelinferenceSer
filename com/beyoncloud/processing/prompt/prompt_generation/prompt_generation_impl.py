import logging
from typing import Dict, Any
from com.beyoncloud.processing.prompt.prompt_template import (
    get_prompt_template, 
    get_prompt_param,
    SafeDict
)
from com.beyoncloud.schemas.rag_reqres_data_model import StructureInputData


logger = logging.getLogger(__name__)

class PromptGenerationImpl:
    """Prompt generation implementation"""
    
    def __init__(self):
        # Intentionally empty for now.
        # Reason: This class does not require instance state at construction
        # and will initialize attributes lazily when the analysis runs.
        # If future attributes are needed, initialize them here.
        pass

    async def generate_extraction_prompts(
        self, 
        schema_template,
        structure_input_data: StructureInputData
    ) -> Dict[str, str]:
        """Generate extraction prompts"""
        print("generate_extraction_prompts inside")
        prompt_output = get_prompt_template(
            structure_input_data.domain_id, structure_input_data.document_type, 
            structure_input_data.organization_id, structure_input_data.prompt_type 
        )

        system_prompt_template = prompt_output["system_prompt_template"]
        user_prompt_template = prompt_output["user_prompt_template"]
        input_variables = prompt_output["input_variables"]
        prompt_id = prompt_output["prompt_id"]

        print(f"generate_extraction_prompts -->{prompt_id} -  {input_variables}")
        param_result = await get_prompt_param(prompt_id)

        variable_map = {
            "schema_template": schema_template
        }
        for result in param_result:
            param_key = getattr(result, "param_key", None)
            param_value = getattr(result, "param_value", None)
            if param_key is not None:
                variable_map[param_key] = param_value

        system_prompt = system_prompt_template.format_map(SafeDict(variable_map))
        user_prompt = user_prompt_template.format_map(SafeDict(variable_map))

        print(f"system_prompt ---########################-> {system_prompt}")
        print(f"user_prompt -###########################-> {user_prompt}")
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "input_variables": "context"
        }