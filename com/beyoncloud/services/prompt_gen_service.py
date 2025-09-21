import logging
from com.beyoncloud.processing.prompt.prompt_template import generatePromptJson

logger = logging.getLogger(__name__)

async def generate_prompt_json():
    promptFilepath = await generatePromptJson()
    print(promptFilepath)
    return f"Prompt JSON created successfully in : " + promptFilepath