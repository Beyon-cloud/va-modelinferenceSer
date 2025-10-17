import os
from dotenv import load_dotenv
from pathlib import Path
import yaml
import logging
from com.beyoncloud.common.constants import LogConfig, EnvKeys, CommonPatterns
from com.beyoncloud.utils.file_util import YamlLoader

logger = logging.getLogger(__name__)

# Load base .env
load_dotenv(dotenv_path=".env", encoding="utf-8-sig",override=False)

# Read APP_ENV
APP_ENV = os.getenv(EnvKeys.APP_ENV, "").strip()
if APP_ENV:
    logger.info("Loaded base environment: .env")
else:
    logger.warning("Could not load base .env")

# Load environment-specific .env
env_file = f".env.{APP_ENV}"
load_dotenv(dotenv_path=env_file, encoding="utf-8-sig",override=True)
logger.info(f"✅ Loaded environment override: {env_file}")

# PostgreSQL DB Connectivity
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_MINCONN=os.getenv("POSTGRES_MINCONN")
POSTGRES_MAXCONN=os.getenv("POSTGRES_MAXCONN")
POSTGRES_CMD_TIMEOUT=os.getenv("POSTGRES_CMD_TIMEOUT")

POSTGRES_URI = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

SQLALCHEMY_ECHO = os.getenv("SQLALCHEMY_ECHO", "false").lower() == "true"

BC_ROOT_PATH=os.getenv(EnvKeys.BC_ROOT_PATH, CommonPatterns.SPACE)

# Uvicorn Details
HOST = os.getenv(EnvKeys.UVICORN_HOST, "0.0.0.0")
PORT = int(os.getenv(EnvKeys.UVICORN_PORT, 5009))

# Looger configuration
log_level_str = os.getenv(LogConfig.LEVEL, LogConfig.WARNING).upper()
log_level = getattr(logging, log_level_str, logging.WARNING)

# Application Root Path
app_root = os.environ.get("APP_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

# Entity Filepath
ENTITY_FOLDER_PATH = os.path.join(app_root, "com", "beyoncloud", "data", "entity")

# Faiss Details
FAISS_EMBED_FOLDER_PATH = os.path.join(app_root, "com", "beyoncloud", "data", "embeddings")
FAISS_INDEX_FILENAME=os.getenv("FAISS_INDEX_FILENAME")

# Prompt Details
PROMPT_FOLDER_PATH = os.path.join(app_root, "com", "beyoncloud", "data", "prompt")
PROMPT_FILENAME=os.getenv("PROMPT_FILENAME")

SCHEMA_PROMPT_DIR_PATH = os.path.join(BC_ROOT_PATH, os.getenv("SCHEMA_PROMPT_DIR_PATH"))
SCHEMA_PROMPT_FILENAME=os.getenv("SCHEMA_PROMPT_FILENAME")

CLARIDATA_DIR_PATH = os.path.join(BC_ROOT_PATH, os.getenv("CLARIDATA_DIR_PATH"))
CLARIDATA_FILENAME=os.getenv("CLARIDATA_FILENAME")

FAISS_DIM=os.getenv("FAISS_DIM")

# Temp code for testing purpose
ENABLE_HF_INFRENCE_YN = os.getenv("ENABLE_HF_INFRENCE_YN")
HF_KEY = os.getenv("HF_KEY")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME")
TEMP_FLOW_YN = os.getenv("TEMP_FLOW_YN")

LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH")

# ------------------  Config ------------------

CONFIG_FILE_PATH=os.getenv(EnvKeys.CONFIG_FILE_PATH, CommonPatterns.SPACE)
BASE_PATH=f"{BC_ROOT_PATH}{CONFIG_FILE_PATH}"

preload_temp = os.getenv(EnvKeys.PRELOAD_TEMPLATES, CommonPatterns.SPACE).strip()
PRELOAD_TEMPLATES = [t.strip() for t in preload_temp.split(",") if t.strip()]
LOAD_ALL_TEMPLATES=os.getenv(EnvKeys.LOAD_ALL_TEMPLATES, CommonPatterns.SPACE)

MT_ID_DFN_CONFIG_TEMPLATE = os.getenv(EnvKeys.MT_ID_DFN_CONFIG_TEMPLATE, CommonPatterns.SPACE)
MT_ORG_PROMPT_CONFIG_TEMPLATE = os.getenv(EnvKeys.MT_ORG_PROMPT_CONFIG_TEMPLATE, CommonPatterns.SPACE)

# Load from yaml file
script_dir = Path(__file__).parent

yaml_loader = YamlLoader()

model_config_file_path = os.path.join(script_dir, "yaml", "model_config.yaml")
MODEL_CONFIG = yaml_loader.get_yaml_object(model_config_file_path)

logging_config_file_path = os.path.join(script_dir, "yaml", "logging_config.yaml")
LOGGING_CONFIG = yaml_loader.get_yaml_object(logging_config_file_path)

query_config_file_path = os.path.join(script_dir, "yaml", "queries.yaml")
QUERY_CONFIG = yaml_loader.get_yaml_object(query_config_file_path)

common_config_file_path = os.path.join(script_dir, "yaml", "common_config.yaml")
COMMON_CONFIG = yaml_loader.get_yaml_object(common_config_file_path)

tables_mapper_file_path = os.path.join(script_dir, "yaml", "tables_mapper.yaml")
TABLE_MAPPER = yaml_loader.get_yaml_object(tables_mapper_file_path)

# ------------------ gRPC Server Default Config ------------------
GRPC_SERVER_HOST=os.getenv(EnvKeys.GRPC_SERVER_HOST, "localhost").lower()
GRPC_SERVER_PORT=int(os.getenv(EnvKeys.GRPC_SERVER_PORT, "50051"))

# ------------------ Consul Config ------------------
MODEL_INFERENCE_SERVICE=os.getenv(EnvKeys.MODEL_INFERENCE_SERVICE, "modelinference-service").lower()

# ---------- Consul watcher settings (for grpc client) ----------
CONSUL_USE_HTTPS=os.getenv(EnvKeys.CONSUL_USE_HTTPS, "N").upper()
CONSUL_SERVICE_ENABLED_YN=os.getenv(EnvKeys.CONSUL_SERVICE_ENABLED_YN, "N").upper()
CONSUL_HOST=os.getenv(EnvKeys.CONSUL_HOST, "consul")
CONSUL_PORT=int(os.getenv(EnvKeys.CONSUL_PORT, "8500"))
CONSUL_SERVICE_WATCHER_TIMEOUT = int(os.getenv(EnvKeys.CONSUL_SERVICE_WATCHER_TIMEOUT, 600))
CONSUL_SERVICE_WATCHER_INTERVAL = int(os.getenv(EnvKeys.CONSUL_SERVICE_WATCHER_INTERVAL, 2))
CONSUL_SERVICE_GRPC_PREFER=os.getenv(EnvKeys.CONSUL_SERVICE_GRPC_PREFER, "grpc_port")
