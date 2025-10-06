import logging
from com.beyoncloud.logger.centralized_logging import LoggingFactory

def initialize_logging():

    logging_manager = LoggingFactory.create_development_config()
    
    # Apply environment overrides
    logging_manager.set_environment_overrides()
    
    # Configure logging
    logging_manager.configure()
    
    # Get loggers
    app_logger = logging_manager.get_logger('myapp')
    db_logger = logging_manager.get_logger('myapp.database')
    api_logger = logging_manager.get_logger('myapp.api')
    
    # Regular logging
    app_logger.info("Application started successfully")
    app_logger.error("Failed to connect to external service", exc_info=True)
    db_logger.debug("Executing database query")
    api_logger.warning("Rate limit approaching")
    
