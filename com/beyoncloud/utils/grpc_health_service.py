import logging
from com.beyoncloud.grpc.protos.health.health_grpc import HealthBase
from com.beyoncloud.grpc.protos.health.health_pb2 import HealthCheckResponse

logger = logging.getLogger(__name__)

# -----------------------------
# HealthService: gRPC Health Service
# -----------------------------
class HealthService(HealthBase):
    """gRPC Health service - async implementation."""

    def __init__(self):
        self._status_map = {}

    async def Check(self, stream):
        """Check RPC - returns the status of a service"""
        request = await stream.recv_message()
        service = request.service
        status = self._status_map.get(service, HealthCheckResponse.SERVING)
        await stream.send_message(HealthCheckResponse(status=status))
        logger.debug(f"[HealthService] {service} → {status}")

    async def Watch(self, stream):
        """Watch RPC - simple implementation for clients like Consul"""
        await stream.recv_message()
        await stream.send_message(
            HealthCheckResponse(status=HealthCheckResponse.SERVING)
        )

    def set_status(self, service_name: str, status: str):
        """Set the health status of a service"""
        self._status_map[service_name] = getattr(
            HealthCheckResponse, status, HealthCheckResponse.UNKNOWN
        )
        logger.info(f"[HealthService] {service_name} = {status}")