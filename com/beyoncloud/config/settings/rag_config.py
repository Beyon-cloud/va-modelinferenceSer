import logging

from functools import lru_cache
from com.beyoncloud.common.constants import WebScrap, Numerical, CommonPatterns
from com.beyoncloud.config.settings import env_config

logger = logging.getLogger(__name__)

class RagProcessingSettings:
    """
    Loads and provides access to rag processing configuration from YAML.
    Provides strongly-typed property access for paths, scraping, and Selenium settings.
    """

    def __init__(self):
        try:
            self._config = env_config.RAG_CONFIG
            logger.info("✅ Data processing config cached successfully")
        except Exception as e:
            logger.exception("❌ Failed to cache data processing config", exc_info=True)
            raise RuntimeError(f"Failed to cache config: {e}")

    @property
    def rag_settings(self) -> dict:
        return self._config.get("rag_process_settings", {})

    # ---------------- ENVIRONMENT SETTINGS ----------------

    @property
    def environments(self) -> dict:
        return self.rag_settings.get("environments", {})

    @property
    def get_env_settings(self) -> dict:
        return self.environments.get(env_config.APP_ENV, {})

    @property
    def get_page_crawl_limit(self) -> dict:
        """Return page crawl limit config for the given environment."""
        return self.get_env_settings.get("page_crawl_limit", {})


    @property
    def is_crawl_limit_enabled(self) -> bool:
        """Check if crawl limit is enabled for the environment."""
        return self.get_page_crawl_limit.get("enabled", False)

    @property
    def get_max_pages(self) -> int:
        """Get max page crawl limit for the environment."""
        return self.get_page_crawl_limit.get("max_pages", Numerical.ZERO)

    # ---------------- PATH SETTINGS ----------------
    @property
    def paths(self) -> dict:
        return self.rag_settings.get("paths", {})

    @property
    def base_directory(self) -> str:
        base_dir = self.paths.get("base_directory", env_config.DPS_BASE_DIR)
        if "${DPS_BASE_DIR}" in base_dir:
            base_dir = base_dir.replace("${DPS_BASE_DIR}", env_config.DPS_BASE_DIR)
        return base_dir

    @property
    def folder_pattern(self) -> str:
        return self.rag_settings.get("paths", {}).get("folder_pattern", CommonPatterns.SPACE)

    @property
    def output_structure(self) -> str:
        return self.paths.get("output_structure", {})

    @property
    def file_name_max_length(self) -> int:
        return self.rag_settings.get("file_name_max_length", Numerical.SIXTY)

    # ---------------- WEB SCRAPING ----------------
    @property
    def web_scraping(self) -> dict:
        return self.rag_settings.get("web_scraping", {})

    @property
    def parser(self) -> str:
        return self.web_scraping.get("parser", "lxml")

    @property
    def scraper(self) -> str:
        return self.web_scraping.get("scraper", WebScrap.SELENIUM)

    @property
    def request_headers(self) -> dict:
        raw_headers = self.web_scraping.get("request_headers", {})
        if raw_headers.get("user_agent") == "random":
            bot_identifier = raw_headers.get("bot_identifier", "")
            raw_headers["user_agent"] = user_agent.strip()
            logger.info(f"raw_headers for the request {raw_headers}")
        return {k.title(): v for k, v in raw_headers.items()}

    @property
    def force_remove_tags(self) -> list:
        return self.web_scraping.get("force_remove_tags", [])

    @property
    def conditional_remove_tags(self) -> list:        
        return self.web_scraping.get("conditional_remove_tags", [])

    @property
    def default_threshold(self) -> int:
        return self.web_scraping.get("default_threshold", 50)

    @property
    def links_to_ignore(self) -> dict:
        return self.web_scraping.get("link_ignore_prefixes", [])

    @property
    def web_crawler_concurrent_limit(self) -> dict:
        return self.web_scraping.get("web_crawler_concurrent_limit", Numerical.FIVE)
    @property
    def httpx_timeout(self) -> int:
        return self.web_scraping.get("httpx_timeout", Numerical.TEN)

    @property
    def retry_delay(self) -> int:
        return self.web_scraping.get("retry_delay", Numerical.TWO)

    @property
    def verify_ssl(self) -> int:
        return self.web_scraping.get("verify_ssl", True)

    @property
    def max_retries(self) -> int:
        return self.web_scraping.get("max_retries", Numerical.THREE)

    @property
    def follow_redirect(self) -> bool:
        return self.web_scraping.get("follow_redirect", True)

    # ---------------- DYNAMIC DETECTION ----------------
    @property
    def dynamic_detection(self) -> dict:
        return self.web_scraping.get("dynamic_detection", {})

    @property
    def js_frameworks(self) -> list[str]:
        return self.dynamic_detection.get("js_frameworks", [])

    @property
    def dynamic_content_min_length(self) -> int:
        return self.dynamic_detection.get("min_text_length", Numerical.HUNDRED)

    # ---------------- EXTENSIONS ----------------
    @property
    def allowed_extensions(self) -> dict:
        return self.web_scraping.get("allowed_extensions", {})

    @property
    def user_agent(self) -> str :
        return self.web_scraping.get("user_agent", CommonPatterns.ASTRICK)

    @property
    def audio_extensions(self) -> list[str]:
        return self.allowed_extensions.get("audio", [])

    @property
    def video_extensions(self) -> list[str]:
        return self.allowed_extensions.get("video", [])

    @property
    def image_extensions(self) -> list[str]:
        return self.allowed_extensions.get("image", [])

    @property
    def document_extensions(self) -> list[str]:
        return self.allowed_extensions.get("document", [])

    @property
    def media_extensions(self) -> list[str]:
        return self.audio_extensions + self.video_extensions + self.image_extensions

    # ---------------- SELENIUM ----------------
    @property
    def selenium_settings(self) -> dict:
        return self.web_scraping.get("selenium_settings", {})

    @property
    def browser_selection(self) -> str:
        return self.selenium_settings.get("browser_selection", WebScrap.CHROME)

    @property
    def selenium_chrome_options(self) -> list[str]:
        options = self.selenium_settings.get("chrome", {}).get("chrome_options", [])
        user_agent_option = next((option for option in options if option.startswith("--user-agent")), None)
        if user_agent_option:
            options.remove(user_agent_option)
        
        options.append(f"--user-agent={self.request_headers}")
        return options

    @property
    def selenium_implicit_wait(self) -> int:
        return self.selenium_settings.get("implicit_wait_seconds", Numerical.TEN)

    @property
    def selenium_timeout(self) -> int:
        return self.selenium_settings.get("selenium_timeout", Numerical.SIXTY)

    @property
    def selenium_scroll_iterations(self) -> int:
        return self.selenium_settings.get("scroll_iterations", Numerical.THREE)

    @property
    def selenium_scroll_delay(self) -> int:
        return self.selenium_settings.get("scroll_delay_seconds", Numerical.TWO)

    @property
    def selenium_max_instances(self) -> int:
        return self.selenium_settings.get("max_chrome_instances", Numerical.ONE)


@lru_cache(maxsize=Numerical.ONE)
def get_config() -> DataProcessingSettings:
    """Return a cached singleton instance of DataProcessingSettings."""
    return DataProcessingSettings()
