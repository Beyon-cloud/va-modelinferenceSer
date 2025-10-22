import asyncio
import logging
import contextlib
from grpclib.server import Server
import com.beyoncloud.config.settings.env_config as env_config
from com.beyoncloud.config.settings.table_mapper_config import load_table_mapping
from com.beyoncloud.utils.grpc_health_service import HealthService
from com.beyoncloud.utils.grpc_address_resolver import GrpcAddressResolver
from com.beyoncloud.grpc.server.infrence_model.inf_server import RAGInfService
from com.beyoncloud.grpc.server.dps_ocr.dps_ocr_server import DpsOcrInfService
from com.beyoncloud.models.model_service import ModelServiceLoader
from com.beyoncloud.models import model_singleton
from com.beyoncloud.db.db_connection import sql_db_connection as sqlDbConn

logger = logging.getLogger(__name__)


class GrpcServer:
    """Manages all gRPC servers (e.g., OCR, Inference, etc.)"""
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

            # ✅ Create inference server (multi-servicer)
            self.all_inference_server = BaseGrpcServer(
                service_name=env_config.MODEL_INFERENCE_SERVICE,
                servicer_instances=[RAGInfService(), DpsOcrInfService()],
                prefer=env_config.CONSUL_SERVICE_GRPC_PREFER
            )

            async def run_servers():
                await asyncio.gather(self.all_inference_server.start())

            self._task = asyncio.create_task(run_servers())
            logger.info("✅ gRPC servers started successfully.")

        except Exception as e:
            logger.error(f"❌ Failed to start gRPC servers: {e}", exc_info=True)
            raise
  
    async def shutdown(self):
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
    def __init__(self, service_name: str, servicer_instances: list,prefer="grpc_port"):
        self.service_name = service_name
        self.servicer_instances = servicer_instances
        self.server = None
        self.prefer = prefer
        self.health_service = None
        self._server_task = None  # ✅ background server task

    async def start(self):
        try:
            # ✅ Get gRPC server address using unified resolver
            _, port, bind_host = await GrpcAddressResolver.get_address(
                service_name=self.service_name, prefer=self.prefer,server_mode=True
            )

            # ✅ Create HealthService for Consul checks
            self.health_service = HealthService()
            self.health_service.set_status(self.service_name, "SERVING")

            all_services = self.servicer_instances + [self.health_service]
            self.server = Server(all_services)

            logger.info(f"[{self.service_name}] Starting gRPC server on {bind_host}:{port}")
            await self.server.start(bind_host, int(port))
            
            # ✅ Run wait_closed in the background so start() returns immediately
            self._server_task = asyncio.create_task(self._serve_forever())

        except Exception as e:
            logger.error(f"❌ Error while starting {self.service_name} server: {e}", exc_info=True)
            raise

    async def _serve_forever(self):
        """Keep server alive until closed, handle graceful cancellation."""
        try:
            await self.server.wait_closed()
        except asyncio.CancelledError:
            # ✅ Expected during shutdown
            logger.info(f"[{self.service_name}] Server task cancelled during shutdown.")
            raise
        except Exception as e:
            logger.error(f"[{self.service_name}] Server error during serving: {e}", exc_info=True)
            raise

    async def stop(self):
        """Gracefully stop gRPC server and background task."""
        try:
            if not self.server:
                logger.warning(f"[{self.service_name}] Server was never started.")
                return

            logger.info(f"[{self.service_name}] Stopping gRPC server...")
            if self.health_service:
                self.health_service.set_status(self.service_name, "NOT_SERVING")

            # ✅ Close server gracefully
            self.server.close()
            
            # ✅ Wait for shutdown but handle cancellation cleanly
            try:
                await asyncio.wait_for(self.server.wait_closed(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning(f"[{self.service_name}] Shutdown timeout exceeded.")
            except asyncio.CancelledError:
                logger.info(f"[{self.service_name}] Server shutdown cancelled cleanly.")
                raise

            # ✅ Cancel background task if still running
            if self._server_task and not self._server_task.done():
                self._server_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._server_task
            logger.info(f"[{self.service_name}] Shutdown complete.")
        except asyncio.CancelledError:
            # ✅ Ensure final cancellation is propagated
            logger.info(f"[{self.service_name}] Shutdown cancelled during cleanup.")
            raise
        except Exception as e:
            logger.error(f"[{self.service_name}] Unexpected shutdown error: {e}", exc_info=True)
            raise
