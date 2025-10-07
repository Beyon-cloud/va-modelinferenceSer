import logging
from com.beyoncloud.processing.prompt.prompt_template import generate_prompt_template_json

logger = logging.getLogger(__name__)

async def generate_prompt_json():
    prompt_filepath = await generate_prompt_template_json()
    print(prompt_filepath)
    return f"Prompt JSON created successfully in : {prompt_filepath}"