import logging
from functools import lru_cache
from typing import Dict, List

from com.beyoncloud.common.constants import Numerical
from com.beyoncloud.config.settings import env_config

logger = logging.getLogger(__name__)

class ModelSettings:
    """
    Loads and provides access to Model configuration from YAML.
    Matches the modern YAML schema with:
        - models: []
    """

    def __init__(self):
        try:
            self._model_config = env_config.MODEL_CONFIG
            logger.info("✅ Model config loaded successfully")
        except Exception as e:
            logger.exception("❌ Failed to load model config")
            raise RuntimeError(f"Failed to load config: {e}")

    @property
    def model_config(self) -> dict:
        """Return root cnn_config section"""
        return self._model_config.get("cnn_config", {})

    @property
    def environments(self) -> dict:
        """Return all environments (dev, prod, etc.)"""
        return self.model_config.get("environments", {})

    @property
    def current_env_settings(self) -> dict:
        """Return settings for current APP_ENV"""
        env = env_config.APP_ENV
        if env not in self.environments:
            raise ValueError(f"Environment '{env}' not found in cnn_config")
        return self.environments[env]

    # ---------------- TASK CONFIGS ---------------- #

    @property
    def tasks(self) -> dict:
        """Return all tasks (caption, vqa, generate)"""
        return self.current_env_settings.get("tasks", {})

    def get_task_config(self, task_name: str) -> dict:
        """Get config for a specific task"""
        task = self.tasks.get(task_name)
        if not task:
            raise ValueError(f"Task '{task_name}' not found in config")
        return task

    def get_task_default_model(self, task_name: str) -> str:
        """Get default model alias for a task"""
        return self.get_task_config(task_name)["default_model"]

    def get_task_default_prompt_key(self, task_name: str) -> str:
        """Get default prompt key for a task (e.g., 'describe')"""
        return self.get_task_config(task_name).get("default_prompt_key", "default")

    def get_prompt(self, task_name: str, prompt_key: str = None) -> str:
        """
        Get the actual prompt text for a task and key.
        If prompt_key is None, uses default_prompt_key from config.
        """
        if prompt_key is None:
            prompt_key = self.get_task_default_prompt_key(task_name)

        prompts = self.get_current_environment().get("prompts", {})
        task_prompts = prompts.get(task_name)

        if not task_prompts:
            raise ValueError(f"No prompts defined for task '{task_name}'")

        if prompt_key not in task_prompts:
            raise KeyError(f"Prompt key '{prompt_key}' not found for task '{task_name}'")

        return task_prompts[prompt_key]
    
    @property
    def get_caption_prompt(self, prompt_key: str = None) -> str:
        """
        Get captioning prompt.
        Uses `prompt_key` if given, else uses default from config.
        """
        return self.get_prompt("caption", prompt_key)

    def get_api_rate_limit(self, task_name: str) -> str:
        """Get default model alias for a task"""
        return self.get_task_config(task_name)["api_rate_limit"]

    @property
    def get_caption_rate_limit(self) -> str:
        return self.get_api_rate_limit("caption")

    @property
    def get_vqa_rate_limit(self) -> str:
        return self.get_api_rate_limit("vqa")

    
    def allowed_mime_types(self, task_name: str) -> list:
        """Return all allowed_mime_types"""
        return self.get_task_config(task_name)["allowed_mime_types"]

    @property
    def caption_allowed_mime_types(self) -> list:
        """Return all allowed_mime_types"""
        return self.allowed_mime_types("caption")

    def max_images(self, task_name: str) -> int:
        """Return all max_images"""
        return int(self.get_task_config(task_name)["max_images"],5)

 
    @property
    def max_caption_images(self) -> int:
        """Return all max_images"""
        return self.max_images("caption")


    # ---------------- MODEL REGISTRY ---------------- #

    @property
    def models(self) -> List[Dict]:
        """Return all available models in current environment"""
        return self.current_env_settings.get("models", [])

    def get_model_by_alias(self, alias: str) -> dict:
        """Fetch model config by alias"""
        for model in self.models:
            if model.get("model_alias") == alias:
                # Ensure model_type defaults to "text" if not set
                model.setdefault("model_type", "text")
                return model
        raise ValueError(f"❌ Model alias '{alias}' not found in config")

    def get_model_max_new_tokens(self, alias: str) -> int:
        """Get max_new_tokens for a model"""
        model = self.get_model_by_alias(alias)
        return model.get("max_new_tokens", 128)

    # ---------------- PROMPTS ---------------- #

    @property
    def prompts(self) -> dict:
        """Return all prompts by task and key"""
        return self.current_env_settings.get("prompts", {})

    def get_prompt(self, task: str, prompt_key: str = None) -> str:
        """
        Get prompt text for a task and key.
        Falls back to default_prompt_key if prompt_key is None.
        """
        if task not in self.prompts:
            raise ValueError(f"No prompts defined for task: {task}")

        task_prompts = self.prompts[task]
        if prompt_key is None:
            prompt_key = self.get_task_default_prompt_key(task)

        if prompt_key not in task_prompts:
            raise ValueError(f"Prompt key '{prompt_key}' not found for task '{task}'")

        return task_prompts[prompt_key]

@lru_cache(maxsize=Numerical.ONE)
def get_config() -> CnnSettings:
    """Return cached singleton instance of CnnSettings."""
    return CnnSettings()