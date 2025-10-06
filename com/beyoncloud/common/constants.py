import re


class Status:
    """Pipeline and job statuses."""
    SUBMITTED = "SUBMITTED"
    STARTED = "STARTED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"

    RAG_SENT = "RAG_SENT"
    RAG_FAIL = "RAG_FAIL"
    RAG_SUCC = "RAG_SUCC"
    WIP = "WIP"


class DataSourceConstants:
    """Supported data source types."""
    WEBSITE = "website"
    API = "api"
    DB = "db"


class LogConfig:
    """Logging related constants."""
    LEVEL = "LOG_LEVEL"
    WARNING = "WARNING"
    FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


class EnvKeys:
    """Environment variable keys."""
    APP_ENV = "APP_ENV"
    POSTGRES_HOST = "POSTGRES_HOST"
    POSTGRES_PORT = "POSTGRES_PORT"
    POSTGRES_USER = "POSTGRES_USER"
    POSTGRES_PASSWORD = "POSTGRES_PASSWORD"
    POSTGRES_DB = "POSTGRES_DB"
    POSTGRES_MINCONN = "POSTGRES_MINCONN"
    POSTGRES_MAXCONN = "POSTGRES_MAXCONN"
    POSTGRES_CMD_TIMEOUT = "POSTGRES_CMD_TIMEOUT"

    UVICORN_PORT = "UVICORN_PORT"
    UVICORN_HOST = "UVICORN_HOST"

    SQLALCHEMY_ECHO = "SQLALCHEMY_ECHO"

    DPS_BASE_DIR = "DPS_BASE_DIR"
    DPS_CONFIG_FILE = "DPS_CONFIG_FILE"

    GRPC_ENV = "GRPC_ENV"
    GRPC_CLIENT_SRV_RGTY = "GRPC_CLIENT_SRV_RGTY"
    RAG_GRPC_DEV_HOST = "RAG_GRPC_DEV_HOST"
    RAG_GRPC_DEV_PORT = "RAG_GRPC_DEV_PORT"
    RAG_GRPC_TIMEOUT = "RAG_GRPC_TIMEOUT"
    RAG_GRPC_ENABLED = "RAG_GRPC_ENABLED"

    BC_ROOT_PATH="BC_ROOT_PATH"
    CONFIG_FILE_PATH="CONFIG_FILE_PATH"
    PRELOAD_TEMPLATES="PRELOAD_TEMPLATES"
    LOAD_ALL_TEMPLATES="LOAD_ALL_TEMPLATES"
    MT_ID_DFN_CONFIG_TEMPLATE="MT_ID_DFN_CONFIG_TEMPLATE"
    MT_ORG_PROMPT_CONFIG_TEMPLATE = "MT_ORG_PROMPT_CONFIG_TEMPLATE"


class FileTypes:
    """File type constants."""
    TEXTS = "texts"
    VIDEOS = "videos"
    JSON = "json"
    DOCUMENTS = "documents"
    IMAGES = "images"


class CommonPatterns:
    """Regex ,separators and common constants."""
    
    YOUTUBE_DOMAIN = "youtube"
    URL = "url"
    SOUP = "soup"
    PARAGRAPH_SEPARATOR = "\n\n"
    WHITESPACE = r"\s+"
    SPACE = " "
    EMPTY_SPACE =""
    TRAILING_SLASH = "/"
    ASTRICK = "*"
    HASH = "#"
    DOLLAR = "$"
    COLON =":"
    SCHEMA1 = "schema1"


class Numerical:
    """Numeric constants for reuse."""
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    SIXTY = 60
    HUNDRED = 100
    HUNDRED_TWENTY_EIGHT = 128


class RepositoryOps:
    """Database operation constants."""
    EQUALS = "="
    NOT_EQUALS = "!="
    LESS_THAN = "<"
    LESS_THAN_EQUALS = "<="
    GREATER_THAN = ">"
    GREATER_THAN_EQUALS = ">="
    IN = "IN"
    LIKE = "LIKE"
    ILIKE = "ILIKE"
    IS = "IS"
    ORDER_BY_WITH_PT = "{order_by}"


class RepositoryKeys:
    """Common repository key constants."""
    ID = "id"
    OR = "OR"
    AND = "AND"
    LOGIC = "LOGIC"
    CONDITIONS = "CONDITIONS"
    FIELD = "FIELD"
    OPERATION = "OP"
    VALUE = "VALUE"
    DIRECTION = "DIRECTION"
    DESC = "DESC"

class HTTPStatusCodes:
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204

    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    CONFLICT = 409

    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503

class DBConstants:
    TXN_RAG_SOURCE = "RagSrc"
    TXN_TEXT_EMBED = "RagTxtEmbed"
    TXN_LINKS = "RagLinks"
    TXN_IMG_EMBED = "RagImgEmbed"
    TXN_VIDEO_EMBED = "RagVideoEmbed"
    TXN_QRY_LOG = "RagQryLogs"
    MST_PROMPT = "MtRagPrompt"
    INF_SCH_PROMPT = "InfSchemaPromptLog"
    MST_PROMPT_PARAM = "MtBcPromptParam"

class Delimiter:
    JSON = "```"
    CSV = "```"
    XLSX = "```"

class FileExtension:
    TEXT = ".txt"
    JSON = ".json"
    CSV = ".csv"
    XLSX = ".xlsx"
    PDF = ".pdf"

class FileFormats:
    TEXT = "txt"
    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"
    PDF = "pdf"

class CommonConstants:
    DFLT_LANG = "en"
    SYSTEM = "SYSTEM"

class PromptType:
    RESP_PROMPT = "RESP_PROMPT"
    SCHEMA_PROMPT = "SCHEMA_PROMPT"
    GEN_PROMPT = "GENERATION_PROMPT"