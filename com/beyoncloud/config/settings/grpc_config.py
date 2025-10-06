import os
import re
import logging
import asyncio
import importlib
from grpc.aio import Channel
from grpclib.server import Server
from com.beyoncloud.grpc.server.infrence_model.inf_server import RAGInfService
from com.beyoncloud.grpc.server.dps_ocr.dps_ocr_server import DpsOcrInfService

from com.beyoncloud.models.model_service import ModelServiceLoader
from com.beyoncloud.models import model_singleton
from com.beyoncloud.db.db_connection import sql_db_connection as sqlDbConn
from com.beyoncloud.config.settings.table_mapper_config import load_table_mapping
import com.beyoncloud.config.settings.env_config as env_config

logger = logging.getLogger(__name__)

class GrpcServers:
    """
    Manages both DMS and DPS gRPC servers together.
    """
    def __init__(self):
        self.inf_service = None
        self.dps_ocr_service = None
        self._task = None

    async def start(self):

        logger.info("Application DB startup Initializing ...")
        await sqlDbConn.initialize()

        logger.info("Table mapper Initializing ...")
        load_table_mapping()

        logger.info("Application related models Initializing ...")
        if model_singleton.modelServiceLoader is None:
            model_singleton.modelServiceLoader = ModelServiceLoader()

        self.inf_service = BaseGrpcServer(
            service_name="inf_service",
            servicer_instance=RAGInfService()
        )

        self.dps_ocr_service = BaseGrpcServer(
            service_name="dps_ocr_service",
            servicer_instance=DpsOcrInfService()
        )

        # gather servers to run concurrently
        async def run_servers():
            await asyncio.gather(
                self.inf_service.start(),
                self.dps_ocr_service.start()
            )

        self._task = asyncio.create_task(run_servers())

    async def shutdown(self):
        # if you have stop() methods implemented on BaseGrpcServer
        if self.inf_service:
            await self.inf_service.stop(timeout=5.0)
        if self.dps_ocr_service:
            await self.dps_ocr_service.stop(timeout=5.0)
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.info("Server tasks cancelled during shutdown.")
                raise

# create a singleton so you can import it easily
grpc_servers = GrpcServers()

class BaseGrpcServer:
    def __init__(self, service_name: str, servicer_instance):
        self.service_name = service_name
        self.servicer_instance = servicer_instance

        self.registry = GrpcServerConfigLoader()
        self.host = self.registry.get_host(service_name)
        self.port = self.registry.get_port(service_name)

        self.server = Server([self.servicer_instance])  # moved to attribute

        self._server_task = None

    async def start(self):
        bind_address = f"{self.host}:{self.port}"
        logger.info(f"[{self.service_name}] Starting async gRPC server on {bind_address}")
        await self.server.start(self.host, self.port)
        self._server_task = asyncio.create_task(self.server.wait_closed())

    async def stop(self, timeout: float = 5.0):
        logger.info(f"[{self.service_name}] Stopping async gRPC server on {self.host}:{self.port}")
        self.server.close()
        try:
            if self._server_task:
                await asyncio.wait_for(self._server_task, timeout)
            logger.info(f"[{self.service_name}] Shutdown complete.")
        except asyncio.TimeoutError:
            logger.warning(f"[{self.service_name}] Shutdown timeout exceeded. Forcing shutdown.")
        except asyncio.CancelledError:
            logger.info(f"[{self.service_name}] wait_closed task cancelled.")
            raise


class GrpcServerConfigLoader:
    def __init__(self):
        env = env_config.APP_ENV

        server_config = self.get_server
        env_services = server_config.get(env)
        if not env_services:
            raise ValueError(f"No gRPC service config found for environment: {env}")

        # Substitute env vars in the loaded config subtree
        self.services = self._substitute_env_vars(env_services)

    @property
    def get_server(self) -> dict:
        return env_config.GRPC_CONFIG.get("server", {})

    def _substitute_env_vars(self, data):
        """Recursively substitute ${VAR} with environment variable values."""

        pattern = re.compile(r"\$\{(\w+)\}")

        if isinstance(data, dict):
            return {k: self._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(i) for i in data]
        elif isinstance(data, str):
            def replace(match):
                var_name = match.group(1)
                return os.getenv(var_name, match.group(0))  # fallback: keep ${VAR} if not set
            return pattern.sub(replace, data)
        else:
            return data

    def get_host(self, service_name: str, default: str = "0.0.0.0") -> str:
        service = self.services.get(service_name)
        if not service or "host" not in service:
            env_host = os.getenv(f"{service_name.upper()}_HOST")
            return env_host or default
        return service["host"]

    def get_port(self, service_name: str, default: int = None) -> int:
        service = self.services.get(service_name)
        if not service or "port" not in service:
            env_port = os.getenv(f"{service_name.upper()}_PORT")
            if env_port:
                return int(env_port)
            if default is not None:
                return default
            raise ValueError(f"Port not defined for service: {service_name}")
        return int(service["port"])

grpc_stub_store = {}  # GLOBAL dictionary to hold stubs

class GrpcClient:
    def __init__(self):
        env = env_config.APP_ENV
        client_config = self.get_client
        raw_config = client_config.get(env)

        if not raw_config:
            raise ValueError(f"Environment '{env}' not found in config")

        self.config = self._substitute_env_vars(raw_config)
        self.channels = {}

    def _substitute_env_vars(self,obj):
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(i) for i in obj]
        elif isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj

    @property
    def get_client(self) -> dict:
        return env_config.GRPC_CONFIG.get("client", {})

    async def init_stub(self, service_name):
        if service_name in grpc_stub_store:
            return grpc_stub_store[service_name]

        service = self.config.get(service_name)
        if not service:
            raise ValueError(f"Service '{service_name}' not found in config")

        host = service["host"]
        port_str = service["port"]

        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port number '{port_str}' for service '{service_name}'")

        module = importlib.import_module(service["module"])
        stub = await module.get_stub(host, port)

        if hasattr(stub, "_channel") and isinstance(stub._channel, Channel):
            self.channels[service_name] = stub._channel

        grpc_stub_store[service_name] = stub
        return stub

    def get_stub(self, service_name):
        return grpc_stub_store.get(service_name)

    async def shutdown(self):
        for name, channel in self.channels.items():
            try:
                await channel.close()
                print(f"gRPC channel closed for service: {name}")
            except Exception as e:
                print(f"Failed to close channel for {name}: {e}")