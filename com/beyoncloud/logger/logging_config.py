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
    
    """
    # Use context for request tracing
    with logging_manager.logging_context(request_id="req-123", user_id="user-456"):
        app_logger.info("Processing user request")
        db_logger.debug("Executing database query")
        api_logger.warning("Rate limit approaching")
    """
    # Regular logging
    app_logger.info("Application started successfully")
    app_logger.error("Failed to connect to external service", exc_info=True)
    
