import logging
from com.beyoncloud.processing.prompt.prompt_template import generate_prompt_template_json

logger = logging.getLogger(__name__)

async def generate_prompt_json():
    promptFilepath = await generate_prompt_template_json()
    print(promptFilepath)
    return f"Prompt JSON created successfully in : {promptFilepath}"