from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import logging
import logging.config
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager


class LogLevel(Enum):
    """Enumeration for log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """Enumeration for log formats"""
    JSON = "json"
    DETAILED = "detailed"
    SIMPLE = "simple"
    CUSTOM = "custom"


class HandlerType(Enum):
    """Enumeration for handler types"""
    CONSOLE = "console"
    FILE = "file"
    ROTATING_FILE = "rotating_file"
    TIMED_ROTATING_FILE = "timed_rotating_file"
    SYSLOG = "syslog"
    HTTP = "http"
    SMTP = "smtp"
    CUSTOM = "custom"


@dataclass
class LoggerConfig:
    """Configuration for a specific logger"""
    name: str
    level: LogLevel = LogLevel.INFO
    handlers: List[str] = field(default_factory=list)
    propagate: bool = False
    extra_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandlerConfig:
    """Configuration for a log handler"""
    name: str
    handler_type: HandlerType
    level: LogLevel = LogLevel.INFO
    formatter: str = "json"
    target: Optional[str] = None  # file path, host:port, etc.
    max_bytes: Optional[int] = None
    backup_count: Optional[int] = None
    encoding: str = "utf8"
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormatterConfig:
    """Configuration for a log formatter"""
    name: str
    format_type: LogFormat
    format_string: Optional[str] = None
    date_format: Optional[str] = None
    custom_class: Optional[str] = None


class ILoggingFormatter(ABC):
    """Interface for custom logging formatters"""
    
    @abstractmethod
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record"""
        pass


class ILoggingHandler(ABC):
    """Interface for custom logging handlers"""
    
    @abstractmethod
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the handler"""
        pass


class ILoggingFilter(ABC):
    """Interface for logging filters"""
    
    @abstractmethod
    def filter(self, record: logging.LogRecord) -> bool:
        """Determine if the record should be logged"""
        pass


class ILoggingContext(ABC):
    """Interface for logging context managers"""
    
    @abstractmethod
    def add_context(self, **kwargs) -> None:
        """Add context to log records"""
        pass
    
    @abstractmethod
    def remove_context(self, *keys) -> None:
        """Remove context from log records"""
        pass


class ILoggingManager(ABC):
    """Interface for logging manager"""
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure logging system"""
        pass
    
    @abstractmethod
    def get_logger(self, name: str) -> logging.Logger:
        """Get a configured logger"""
        pass
    
    @abstractmethod
    def add_handler(self, handler_config: HandlerConfig) -> None:
        """Add a new handler"""
        pass
    
    @abstractmethod
    def add_formatter(self, formatter_config: FormatterConfig) -> None:
        """Add a new formatter"""
        pass
    
    @abstractmethod
    def reload_config(self) -> None:
        """Reload logging configuration"""
        pass


class JSONFormatter(ILoggingFormatter):
    """JSON formatter implementation"""
    
    def __init__(self, format_string: Optional[str] = None):
        self.format_string = format_string or '%(asctime)s %(name)s %(levelname)s %(message)s'
        self.formatter = logging.Formatter(self.format_string)
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': self.formatter.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process,
        }
        
        # Add extra context if present
        if hasattr(record, 'context'):
            log_entry.update(record.context)
        
        # Add exception info if present
        if record.exc_info:
            formatter = logging.Formatter()
            log_entry['exception'] = formatter.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class LoggingContext(ILoggingContext):
    """Context manager for adding context to logs"""
    
    def __init__(self):
        self._context: Dict[str, Any] = {}
        self._old_factory = None
    
    def add_context(self, **kwargs) -> None:
        """Add context that will be included in all log records"""
        self._context.update(kwargs)
    
    def remove_context(self, *keys) -> None:
        """Remove context keys"""
        for key in keys:
            self._context.pop(key, None)
    
    def clear_context(self) -> None:
        """Clear all context"""
        self._context.clear()
    
    def __enter__(self):
        self._old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            record.context = self._context.copy()
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)


class ConfigurationManager:
    """Manages logging configuration"""
    
    def __init__(self):
        self.formatters: Dict[str, FormatterConfig] = {}
        self.handlers: Dict[str, HandlerConfig] = {}
        self.loggers: Dict[str, LoggerConfig] = {}
        self.root_config: Optional[LoggerConfig] = None
    
    def add_formatter(self, config: FormatterConfig) -> 'ConfigurationManager':
        """Add formatter configuration"""
        self.formatters[config.name] = config
        return self
    
    def add_handler(self, config: HandlerConfig) -> 'ConfigurationManager':
        """Add handler configuration"""
        self.handlers[config.name] = config
        return self
    
    def add_logger(self, config: LoggerConfig) -> 'ConfigurationManager':
        """Add logger configuration"""
        if config.name == '':
            self.root_config = config
        else:
            self.loggers[config.name] = config
        return self
    
    def build_config(self) -> Dict[str, Any]:
        """Build the complete logging configuration dictionary"""
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {},
            'handlers': {},
            'loggers': {}
        }
        
        # Build formatters
        for formatter_config in self.formatters.values():
            config['formatters'][formatter_config.name] = self._build_formatter_config(formatter_config)
        
        # Build handlers
        for handler_config in self.handlers.values():
            config['handlers'][handler_config.name] = self._build_handler_config(handler_config)
        
        # Build loggers
        for logger_config in self.loggers.values():
            config['loggers'][logger_config.name] = self._build_logger_config(logger_config)
        
        # Add root logger if configured
        if self.root_config:
            config['root'] = self._build_logger_config(self.root_config)
        
        return config
    
    def _build_formatter_config(self, formatter_config: FormatterConfig) -> Dict[str, Any]:
        """Build formatter configuration"""
        if formatter_config.format_type == LogFormat.JSON:
            return {
                '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                'format': formatter_config.format_string or '%(asctime)s %(name)s %(levelname)s %(message)s'
            }
        elif formatter_config.format_type == LogFormat.CUSTOM and formatter_config.custom_class:
            return {'()': formatter_config.custom_class}
        else:
            config = {
                'format': formatter_config.format_string or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
            if formatter_config.date_format:
                config['datefmt'] = formatter_config.date_format
            return config
    
    def _build_handler_config(self, handler_config: HandlerConfig) -> Dict[str, Any]:
        """Build handler configuration"""
        base_config = {
            'level': handler_config.level.value,
            'formatter': handler_config.formatter
        }
        
        if handler_config.handler_type == HandlerType.CONSOLE:
            base_config.update({
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout'
            })
        elif handler_config.handler_type == HandlerType.FILE:
            base_config.update({
                'class': 'logging.FileHandler',
                'filename': handler_config.target,
                'encoding': handler_config.encoding
            })
        elif handler_config.handler_type == HandlerType.ROTATING_FILE:
            base_config.update({
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': handler_config.target,
                'maxBytes': handler_config.max_bytes or 10485760,
                'backupCount': handler_config.backup_count or 5,
                'encoding': handler_config.encoding
            })
        elif handler_config.handler_type == HandlerType.SYSLOG:
            base_config.update({
                'class': 'logging.handlers.SysLogHandler',
                'address': handler_config.target or ('localhost', 514)
            })
        
        # Add extra parameters
        base_config.update(handler_config.extra_params)
        return base_config
    
    def _build_logger_config(self, logger_config: LoggerConfig) -> Dict[str, Any]:
        """Build logger configuration"""
        return {
            'handlers': logger_config.handlers,
            'level': logger_config.level.value,
            'propagate': logger_config.propagate
        }


class CentralizedLoggingManager(ILoggingManager):
    """Main implementation of centralized logging manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config_manager = ConfigurationManager()
        self.context = LoggingContext()
        self._is_configured = False
        self._setup_default_configuration()
    
    def _setup_default_configuration(self) -> None:
        """Setup default logging configuration"""
        # Default formatters
        self.config_manager.add_formatter(
            FormatterConfig(name="json", format_type=LogFormat.JSON)
        ).add_formatter(
            FormatterConfig(name="detailed", format_type=LogFormat.DETAILED)
        ).add_formatter(
            FormatterConfig(name="simple", format_type=LogFormat.SIMPLE, format_string="%(levelname)s - %(message)s")
        )
        
        # Default handlers
        self.config_manager.add_handler(
            HandlerConfig(name="console", handler_type=HandlerType.CONSOLE, formatter="json")
        ).add_handler(
            HandlerConfig(
                name="file", 
                handler_type=HandlerType.ROTATING_FILE, 
                target="logs/app.log",
                formatter="json",
                max_bytes=10485760,
                backup_count=5
            )
        ).add_handler(
            HandlerConfig(
                name="error_file", 
                handler_type=HandlerType.ROTATING_FILE, 
                target="logs/errors.log",
                level=LogLevel.ERROR,
                formatter="detailed",
                max_bytes=10485760,
                backup_count=5
            )
        )
        
        # Default root logger
        self.config_manager.add_logger(
            LoggerConfig(name="", level=LogLevel.INFO, handlers=["console", "file", "error_file"])
        )
    
    def configure(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Configure the logging system"""
        if config:
            logging.config.dictConfig(config)
        else:
            # Use default configuration
            self._ensure_log_directory()
            logging.config.dictConfig(self.config_manager.build_config())
        
        self._is_configured = True
    
    def _ensure_log_directory(self) -> None:
        """Ensure log directory exists"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a configured logger instance"""
        if not self._is_configured:
            self.configure()
        return logging.getLogger(name)
    
    def add_handler(self, handler_config: HandlerConfig) -> None:
        """Add a new handler configuration"""
        self.config_manager.add_handler(handler_config)
        if self._is_configured:
            self.reload_config()
    
    def add_formatter(self, formatter_config: FormatterConfig) -> None:
        """Add a new formatter configuration"""
        self.config_manager.add_formatter(formatter_config)
        if self._is_configured:
            self.reload_config()
    
    def add_logger_config(self, logger_config: LoggerConfig) -> None:
        """Add a new logger configuration"""
        self.config_manager.add_logger(logger_config)
        if self._is_configured:
            self.reload_config()
    
    def reload_config(self) -> None:
        """Reload the logging configuration"""
        self.configure()
    
    @contextmanager
    def logging_context(self, **kwargs):
        """Context manager for adding context to logs"""
        self.context.add_context(**kwargs)
        with self.context:
            yield
        self.context.clear_context()
    
    def set_environment_overrides(self) -> None:
        """Apply environment variable overrides"""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        # Below Changes done by ramana for future usecase, Commenting now due to Sonar issue
        #log_format = os.getenv('LOG_FORMAT', 'json')
        environment = os.getenv('ENVIRONMENT', 'development')
        
        # Update root logger level
        if hasattr(LogLevel, log_level):
            if self.config_manager.root_config:
                self.config_manager.root_config.level = LogLevel[log_level]
        
        # Add production-specific handlers
        if environment == 'production':
            self.config_manager.add_handler(
                HandlerConfig(
                    name="syslog",
                    handler_type=HandlerType.SYSLOG,
                    target=('localhost', 514),
                    formatter="json"
                )
            )
            # Add syslog to root logger handlers
            if self.config_manager.root_config:
                self.config_manager.root_config.handlers.append("syslog")


# Factory class for easy setup
class LoggingFactory:
    """Factory for creating logging configurations"""
    
    @staticmethod
    def create_development_config() -> CentralizedLoggingManager:
        """Create configuration for development environment"""
        manager = CentralizedLoggingManager()
        manager.config_manager.add_handler(
            HandlerConfig(name="debug_console", handler_type=HandlerType.CONSOLE, level=LogLevel.DEBUG, formatter="detailed")
        )
        return manager
    
    @staticmethod
    def create_production_config() -> CentralizedLoggingManager:
        """Create configuration for production environment"""
        manager = CentralizedLoggingManager()
        
        # Add production-specific handlers
        manager.config_manager.add_handler(
            HandlerConfig(
                name="production_file",
                handler_type=HandlerType.ROTATING_FILE,
                target="/var/log/app/production.log",
                level=LogLevel.INFO,
                formatter="json",
                max_bytes=52428800,  # 50MB
                backup_count=10
            )
        ).add_handler(
            HandlerConfig(
                name="syslog",
                handler_type=HandlerType.SYSLOG,
                target=('rsyslog-server', 514),
                formatter="json"
            )
        )
        
        # Configure production logger
        manager.config_manager.add_logger(
            LoggerConfig(
                name="",
                level=LogLevel.INFO,
                handlers=["production_file", "syslog", "error_file"]
            )
        )
        
        return manager
    
    @staticmethod
    def create_testing_config() -> CentralizedLoggingManager:
        """Create configuration for testing environment"""
        manager = CentralizedLoggingManager()
        
        # Minimal logging for tests
        manager.config_manager.add_logger(
            LoggerConfig(
                name="",
                level=LogLevel.WARNING,
                handlers=["console"]
            )
        )
        
        return manager

