import os
import logging
from typing import Dict
from com.beyoncloud.config.settings.dps_config import get_config
from com.beyoncloud.common.constants import FileTypes

logger = logging.getLogger(__name__)
settings = get_config()


class DirectoryUtils:
    """
    Utility class to manage directory structures for extracted and processed data.
    
    Author:
        Balaji G.R (24-07-2025)
    
    This class:
    - Builds dynamic output paths based on organization ID and source type.
    - Creates folder structures for `raw` and `processed` data as defined in YAML settings.
    - Provides helper methods to retrieve specific subdirectories (texts, json, videos, etc.)
      under both raw and processed categories.

    The folder structure is defined in the YAML under `paths.output_structure`, for example:

    ```yaml
    output_structure:
      raw: [texts, json, images, videos, documents]
      processed: [json]
    ```

    Example folder layout for org_id=1 and source_type="website":
    ```
    /base/org_1/website/raw/texts
    /base/org_1/website/raw/json
    /base/org_1/website/processed/json
    ```

    Usage:
    -------
    ```python
    DirectoryUtils.create_output_structure_for_source(1, "website")
    text_dir = DirectoryUtils.get_raw_text_directory(1, "website")
    ```
    """

    @staticmethod
    def _build_base_output_path(org_id: int, source_type: str) -> str:
        """Build the base output path using dynamic folder format."""
        dynamic_folder = settings.folder_pattern.format(
            org_id=org_id, source_type=source_type
        )
        return os.path.join(settings.base_directory, dynamic_folder)

    @staticmethod
    def get_output_structure_paths(org_id: int, source_type: str) -> Dict[str, Dict[str, str]]:
        """Return a mapping of category → subfolder → full path."""
        base_path = DirectoryUtils._build_base_output_path(org_id, source_type)
        paths_config = settings.output_structure

        return {
            category: {
                subfolder: os.path.join(base_path, category, subfolder)
                for subfolder in subfolders
            }
            for category, subfolders in paths_config.items()
        }

    @staticmethod
    def create_output_structure_for_source(org_id: int, source_type: str) -> None:
        """Create the entire folder structure for the given org_id and source type."""
        structure = DirectoryUtils.get_output_structure_paths(org_id, source_type)

        for category, folders in structure.items():
            for subfolder, path in folders.items():
                try:
                    os.makedirs(path, exist_ok=True)
                    logger.debug(f"📂 Created directory: {path}")
                except Exception as e:
                    logger.exception(f"❌ Failed to create path {path}: {e}")

    @staticmethod
    def _get_resource_directory(org_id: int, source: str, category: str, folder_name: str) -> str:
        """Safely retrieve a specific directory path from the structure."""
        structure = DirectoryUtils.get_output_structure_paths(org_id, source)
        try:
            return structure[category][folder_name]
        except KeyError:
            logger.error(f"❌ Invalid path requested: {category}/{folder_name}")
            raise ValueError(f"Invalid directory or folder name: {category}/{folder_name}")

    # ---------- RAW DIRECTORIES ----------
    @staticmethod
    def get_raw_text_directory(org_id: int, source: str) -> str:
        return DirectoryUtils._get_resource_directory(org_id, source, FileTypes.RAW, FileTypes.TEXTS)

    @staticmethod
    def get_raw_web_page_directory(org_id: int, source: str) -> str:
        return DirectoryUtils._get_resource_directory(org_id, source, FileTypes.RAW, FileTypes.WEB_PAGE)

    @staticmethod
    def get_raw_video_directory(org_id: int, source: str) -> str:
        return DirectoryUtils._get_resource_directory(org_id, source, FileTypes.RAW, FileTypes.VIDEOS)

    @staticmethod
    def get_raw_images_directory(org_id: int, source: str) -> str:
        return DirectoryUtils._get_resource_directory(org_id, source, FileTypes.RAW, FileTypes.IMAGES)

    @staticmethod
    def get_raw_documents_directory(org_id: int, source: str) -> str:
        return DirectoryUtils._get_resource_directory(org_id, source, FileTypes.RAW, FileTypes.DOCUMENTS)

    # ---------- PROCESSED DIRECTORIES ----------
    @staticmethod
    def get_prc_web_page_directory(org_id: int, source: str) -> str:
        return DirectoryUtils._get_resource_directory(org_id, source, FileTypes.PROCESSED, FileTypes.WEB_PAGE)
