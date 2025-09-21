"""
This module initializes the FastAPI application's lifecycle events,
setting up database connections and loading machine learning models.

Key Components:
- Sets the log level based on configuration.
- Initializes SQL database connection on application startup.
- Loads all registered models into memory using ModelRegistry.
- Closes SQL database connection on application shutdown.

Attributes:
    logger (logging.Logger): Logger instance for application lifecycle events.

Functions:
    lifespan(app: FastAPI): Async context manager that handles startup and shutdown events
                            for the FastAPI application.

Usage:
    Used in FastAPI as:
        app = FastAPI(lifespan=lifespan)

Author: Jenson
Date: 03-June-2025 
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from com.beyoncloud.config.settings import env_config as env_config
from com.beyoncloud.db.db_connection import sql_db_connection as sqlDbConn
#from com.beyoncloud.grpc.config.grpc_client_loader import GrpcClient
#from com.beyoncloud.grpc.config.grpc_all_servers_loader import grpc_servers
from com.beyoncloud.config.settings.grpc_config import grpc_servers

logger = logging.getLogger(__name__)

#grpc_client = GrpcClient()
grpc_server = None
grpc_processes = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager for FastAPI application lifespan events.

    This function is invoked when the FastAPI application starts and stops.
    It performs the following tasks:
        - Initializes SQL database connections.
        - Loads all ML models into memory via ModelRegistry.
        - Cleans up resources on shutdown, such as closing DB connections.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None
    """

    global grpc_server, grpc_client, grpc_processes
    try:
        logger.info("Application DB startup: Initializing ...")
        #await noSqlDbConn.create_db_indexes()
        await sqlDbConn.initialize()

        #logger.info("Initializing application and loading models...")
        #ModelRegistry.load_all_models()
        #model_singleton.modelServiceLoader = ModelServiceLoader() 
        #logger.info("Models loaded successfully!")

        logger.info("Starting gRPC all microservices")
        #grpc_processes = run_all_servers()
        await grpc_servers.start()
        logger.info("gRPC all microservices started successfully.")

        #logger.info("Creating gRPC client instance")
        #await grpc_client.start()
        #grpc_client = GrpcClient()
        #await grpc_client.init_stub("rag")
        #logger.info("gRPC client instance created successfully.")

        #logger.info("Creating gRPC server instance")
        #grpc_server = GrpcServer()
        #logger.info("gRPC server instance created successfully.")

        yield
    finally:
        print("Life span close connection calling.....")
        #noSqlDbConn.close_connection()
        await sqlDbConn.close_connection()

        # gRPC client shutdown
        logger.info("Shutdown gRPC client instance")
        await grpc_servers.shutdown()
        #await grpc_client.shutdown()
        # Terminate gRPC microservices cleanly
        #for proc in grpc_processes:
        #    proc.terminate()
        #    proc.join()
