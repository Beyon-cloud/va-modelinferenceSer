import logging
import yaml
from functools import lru_cache
from pathlib import Path
import com.beyoncloud.config.settings.env_config as config
 
logger = logging.getLogger(__name__)
 
class TableSettings:
    """
    Loads table configurations from YAML (supports multiple schemas).
    Provides helper methods to get table and column mappings.
    """
 
    def __init__(self):
        logger.info(f"Table mapping initilizing...")
        #self._config = YamlConfigLoader(CONFIG_PATH).load_yaml_config()
        #self.schemas = self._config.get("schemas", {})
        self.schemas = config.TABLE_MAPPER.get("schemas", {})
 
    def get_table_config(self, schema_name: str, table_name: str) -> dict:
        print(self.schemas)
        return (
            self.schemas.get(schema_name, {})
            .get("tables", {})
            .get(table_name, {})
        )
 
    def get_column_mappings(self, schema_name: str, table_name: str) -> dict:
        table_cfg = self.get_table_config(schema_name, table_name)
        return table_cfg.get("columns", {})
 
    def get_db_table_name(self, schema_name: str, table_name: str) -> str:
        table_cfg = self.get_table_config(schema_name, table_name)
        return table_cfg.get("table_name", "")
 
    def get_db_column_name(self, schema_name: str, table_name: str, column_key: str) -> str:
        """Return DB column name for a given logical column key."""
        return self.get_column_mappings(schema_name, table_name).get(column_key, "")
 
    def all_tables(self) -> dict:
        return self.schemas
 
 
@lru_cache(maxsize=1)
def load_table_mapping() -> TableSettings:
    """Cached singleton instance."""
    return TableSettings()

#table_settings = get_table_config()
#print("Table Name:", table_settings.get_db_column_name("schema1", "DATA_SRC","last_updated_at"))