import asyncio
import logging
from grpclib.server import Server
import com.beyoncloud.config.settings.env_config as env_config
from com.beyoncloud.config.settings.table_mapper_config import load_table_mapping
from com.beyoncloud.utils.consul_watcher import ConsulServiceWatcher
from com.beyoncloud.utils.health_service import HealthService
from com.beyoncloud.grpc.server.infrence_model.inf_server import RAGInfService
from com.beyoncloud.grpc.server.dps_ocr.dps_ocr_server import DpsOcrInfService
from com.beyoncloud.models.model_service import ModelServiceLoader
from com.beyoncloud.models import model_singleton
from com.beyoncloud.db.db_connection import sql_db_connection as sqlDbConn

logger = logging.getLogger(__name__)

class GrpcServer:
    """
    Manages all gRPC servers (e.g., OCR, Inference, etc.)
    """
    def __init__(self):
        self.all_inference_server = None
        self._task = None

    async def start(self):
        try:
            logger.info("Application DB startup Initializing ...")
            await sqlDbConn.initialize()

            logger.info("Table mapper Initializing ...")
            load_table_mapping()

            logger.info("Application related models Initializing ...")
            if model_singleton.modelServiceLoader is None:
                model_singleton.modelServiceLoader = ModelServiceLoader()

            # Example: Multi-servicer gRPC server (can serve multiple services)
            self.all_inference_server = BaseGrpcServer(
                service_name=env_config.MODEL_INFERENCE_SERVICE,
                servicer_instances=[
                    RAGInfService(),
                    DpsOcrInfService()
                ]
            )

            async def run_servers():
                await asyncio.gather(
                    self.all_inference_server.start()
                )

            self._task = asyncio.create_task(run_servers())
            logger.info("✅ gRPC servers started successfully.")

        except Exception as e:
            logger.error(f"❌ Failed to start gRPC servers: {e}", exc_info=True)
            raise
  
    async def shutdown(self):
        # if you have stop() methods implemented on BaseGrpcServer
        if self.all_inference_server:
            await self.all_inference_server.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.info("Server tasks cancelled during shutdown.")
                raise

grpc_servers = GrpcServer()


# -------------------- BASE SERVER --------------------
class BaseGrpcServer:
    def __init__(self, service_name: str, servicer_instances: list, prefer="grpc_port"):
        self.service_name = service_name
        self.servicer_instances = servicer_instances
        self.prefer = prefer
        self.server = None
        self.health_service = None
        self.watcher = None

    def start(self):
        try:
            # ------------------------------------------------------------------
            # CONSUL MODE
            # ------------------------------------------------------------------
            if env_config.CONSUL_SERVICE_ENABLED_YN == "Y":
                logger.info(f"[{self.service_name}] Consul discovery enabled — starting watcher...")
                self.watcher = ConsulServiceWatcher(self.service_name, prefer=self.prefer)
                self.watcher.start()

                while not self.watcher.get_address():
                    logger.info(f"[{self.service_name}] waiting for Consul service discovery...")
                    asyncio.sleep(1)

                _, port = self.watcher.get_address().split(":")
                bind_host = "0.0.0.0"

                # Health service for Consul gRPC checks
                self.health_service = HealthService()
                self.health_service.set_status(self.service_name, "SERVING")

                all_services = self.servicer_instances + [self.health_service]
                self.server = Server(all_services)

                logger.info(f"[{self.service_name}] Starting via Consul on {bind_host}:{port}")

            # ------------------------------------------------------------------
            # STATIC CONFIG MODE (no Consul)
            # ------------------------------------------------------------------
            else:
                logger.info(f"[{self.service_name}] Consul disabled — using grpc_config.yml")

                host = env_config.GRPC_SERVER_HOST
                port = env_config.GRPC_SERVER_PORT
                bind_host = host or "0.0.0.0"

                # Only application servicers in static mode
                self.server = Server(self.servicer_instances)

                logger.info(f"[{self.service_name}] Starting static gRPC server on {bind_host}:{port}")

            # ------------------------------------------------------------------
            # START SERVER
            # ------------------------------------------------------------------
            self.server.start(bind_host, int(port))
            self.server.wait_closed()

        except Exception as e:
            logger.error(f"❌ Error while starting {self.service_name} server: {e}", exc_info=True)
            raise

    async def stop(self):
        try:
            if not self.server:
                logger.warning(f"[{self.service_name}] Server was never started.")
                return

            logger.info(f"[{self.service_name}] Stopping gRPC server...")
            if self.health_service:
                self.health_service.set_status(self.service_name, "NOT_SERVING")

            self.server.close()
            await asyncio.wait_for(self.server.wait_closed())
            logger.info(f"[{self.service_name}] Shutdown complete.")

        except asyncio.TimeoutError:
            logger.warning(f"[{self.service_name}] Shutdown timeout exceeded.")
        except Exception as e:
            logger.error(f"[{self.service_name}] Unexpected shutdown error: {e}", exc_info=True)
