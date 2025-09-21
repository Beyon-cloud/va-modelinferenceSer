import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# Hugging Face Llama3 Client Implementation ( HuggingFaceLlama3Client Class 1)
class HuggingFaceLlama3Client:
    """Client for interacting with Llama3 via Hugging Face router"""
    
    def __init__(self, api_key: str, model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        self.model_name = str(model_name)
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )
        logger.info(f"Initialized Hugging Face Llama3 client with model: {model_name}")
    
    def test_connection(self) -> bool:
        """Test the Llama 3 connection with a simple query"""
        try:
            messages = [
                {"role": "user", "content": "What is the capital of France?"}
            ]
            payload = {
                "model": self.model_name,
                "messages": messages
            }
            logger.debug(f"[HF-Llama3] Sending payload: {payload} (types: model={type(self.model_name)}, messages={type(messages)})")

            completion = self.client.chat.completions.create(**payload)

            logger.info("Llama 3 Connection Test:")
            logger.info(completion.choices[0].message.content)
            return True
        except Exception as e:
            logger.error(f"Llama 3 connection failed: {e}")
            return False


    def generate_sync(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Synchronous generation using Hugging Face router"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.1),
                "max_tokens": kwargs.get("max_tokens", 2048),
                "timeout": kwargs.get("timeout", 30),
            }
            logger.debug(f"[HF-Llama3] Sending payload: {payload} (types: model={type(self.model_name)}, messages={type(messages)})")

            completion = self.client.chat.completions.create(**payload)

            response_content = completion.choices[0].message.content
            logger.info(f"Generated response length: {len(response_content)} characters")
            return response_content

        except Exception as e:
            logger.error(f"Llama3 generation failed: {e}")
            raise
