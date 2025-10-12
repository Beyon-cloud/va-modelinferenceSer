import asyncio
import logging
from com.beyoncloud.utils.consul_watcher import ConsulServiceWatcher
import com.beyoncloud.config.settings.env_config as env_config

logger = logging.getLogger(__name__)

class GrpcAddressResolver:
    """
    Unified resolver for both client and server gRPC host/port resolution.
    Supports Consul-based dynamic lookup or static YAML-based config.
    """

    @staticmethod
    async def get_address(service_name: str, service=None, prefer="grpc_port", server_mode=False):
        """
        Returns (host, port) for client or (host, port, bind_host) for server.
        Automatically decides based on CONSUL_SERVICE_ENABLED_YN flag.
        """
        if env_config.CONSUL_SERVICE_ENABLED_YN == "Y":
            return await GrpcAddressResolver._resolve_via_consul(service_name, prefer, server_mode)
        return GrpcAddressResolver._resolve_static(service_name, service, server_mode)

    # ----------------------------------------------------------------------
    # 🔍 Consul mode
    # ----------------------------------------------------------------------
    @staticmethod
    async def _resolve_via_consul(service_name: str, prefer="grpc_port", server_mode=False):
        """Resolve address via Consul watcher."""
        logger.info(f"[{service_name}] 🔍 Using Consul for address resolution...")

        watcher = ConsulServiceWatcher(service_name, prefer=prefer)
        watcher.start()

        timeout = env_config.CONSUL_SERVICE_WATCHER_TIMEOUT
        interval = env_config.CONSUL_SERVICE_WATCHER_INTERVAL
        waited = 0

        # Wait until watcher gets a valid address
        while not watcher.get_address():
            if waited >= timeout:
                raise RuntimeError(f"No address resolved for {service_name} after {timeout}s")
            logger.info(f"[{service_name}] Waiting for Consul registration...")
            await asyncio.sleep(interval)
            waited += interval

        # Extract host and port
        addr = watcher.get_address()
        host, port = addr.split(":")
        bind_host = "0.0.0.0" if server_mode else host

        # Server mode → (host, port, bind_host)
        if server_mode:
            logger.info(f"[{service_name}] ✅ Consul resolved: bind={bind_host}, port={port}")
            return host, port, bind_host

        # Client mode → (host, port)
        logger.info(f"[{service_name}] ✅ Consul resolved: {host}:{port}")
        return host, port

    # ----------------------------------------------------------------------
    # ⚙️ Static mode
    # ----------------------------------------------------------------------
    @staticmethod
    def _resolve_static(service_name: str, service=None, server_mode=False):
        """Resolve host/port from static grpc_config.yml."""
        if server_mode:
            host = env_config.GRPC_SERVER_HOST or "0.0.0.0"
            port = env_config.GRPC_SERVER_PORT
            bind_host = host or "0.0.0.0"
            logger.info(f"[{service_name}] ⚙️ Static mode → bind={bind_host}:{port}")
            return host, port, bind_host

        if not service:
            raise ValueError(f"Static config not provided for client service: {service_name}")

        host = service.get("host")
        port = service.get("port")

        if not host or not port:
            raise ValueError(f"Missing host/port for client service '{service_name}'")

        logger.info(f"[{service_name}] ⚙️ Static mode → {host}:{port}")
        return host, port
