from com.beyoncloud.config.settings import env_config as env_config
from com.beyoncloud.config.templates.template_cache import TemplateCache
import logging

logger = logging.getLogger(__name__)

class TemplateLoader:
    def __init__(self):
        self.config_template_cache = TemplateCache(env_config.BASE_PATH, preload=env_config.PRELOAD_TEMPLATES,load_all=env_config.LOAD_ALL_TEMPLATES)

    async def load_all_templates(self):
        logger.info("Loading all templates...")
        await self.config_template_cache.load_templates()
    
    async def refresh_templates(self):
        logger.info("Refreshing all templates...")
        await self.config_template_cache.refresh_template()

    async def get_template(self, template_name: str):
        records = await self.config_template_cache.get_template(template_name)
        logger.info(f"Loaded {len(records)} records for {template_name}")
        if records:
            logger.info(f"{template_name}-->  {records[0]}")
        return records
