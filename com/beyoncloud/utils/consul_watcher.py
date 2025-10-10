import aiohttp
import asyncio
import logging
import com.beyoncloud.config.settings.env_config as env_config

logger = logging.getLogger(__name__)

CONSUL_HOST = env_config.CONSUL_HOST
CONSUL_PORT = env_config.CONSUL_PORT
CONSUL_USE_HTTPS = env_config.CONSUL_USE_HTTPS


class ConsulServiceWatcher:
    def __init__(self, service_name: str, prefer="grpc_port"):
        self.service_name = service_name
        self.prefer = prefer
        self.current_addr = None
        self.last_index = None
        self.running = False
        self._task = None

    # ----------------------------------------------------------------------
    # 🔹 Helper functions (split logic to reduce complexity)
    # ----------------------------------------------------------------------
    async def _fetch_service_data(self, session, url, params):
        """Fetch service data from Consul with error handling."""
        async with session.get(url, params=params, timeout=310) as resp:
            if resp.status != 200:
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info,
                    history=resp.history,
                    status=resp.status,
                    message=f"Consul returned unexpected status {resp.status}",
                    headers=resp.headers,
                )
            data = await resp.json()
            return data, resp.headers.get("X-Consul-Index")

    def _extract_service_address(self, svc):
        """Extract address and port from Consul service metadata."""
        service_port = svc.get("ServicePort")
        meta = svc.get("ServiceMeta", {})
        if self.prefer in meta:
            service_port = meta[self.prefer]
        return f"{self.service_name}:{service_port}"

    async def _handle_error(self, error_message, delay=5):
        """Generic error handler with delay."""
        logger.error(f"[{self.service_name}] {error_message}")
        await asyncio.sleep(delay)

    async def _process_service_update(self, data, new_index):
        """Process new service data and update current address."""
        if not data:
            logger.warning(f"No instances found for {self.service_name}")
            await asyncio.sleep(2)
            return

        self.last_index = new_index
        new_addr = self._extract_service_address(data[0])
        if new_addr != self.current_addr:
            self.current_addr = new_addr
            logger.info(f"[WATCH] {self.service_name} updated → {self.current_addr}")

    # ----------------------------------------------------------------------
    # 🔹 Main watcher loop
    # ----------------------------------------------------------------------
    async def _watch_loop(self):
        """Main watcher loop — monitors Consul for service address updates."""
        scheme = "https" if CONSUL_USE_HTTPS else "http"
        url = f"{scheme}://{CONSUL_HOST}:{CONSUL_PORT}/v1/catalog/service/{self.service_name}"
        logger.info(f"Starting Consul watcher for {self.service_name} at {url}")
        self.running = True

        while self.running:
            try:
                params = {"index": self.last_index, "wait": "300s"} if self.last_index else {}
                async with aiohttp.ClientSession() as session:
                    data, new_index = await self._fetch_service_data(session, url, params)
                    await self._process_service_update(data, new_index)

            except aiohttp.ClientResponseError as e:
                await self._handle_error(f"HTTP error: {e.status} - {e.message}")
            except aiohttp.ClientConnectionError as e:
                await self._handle_error(f"Connection error: {e}")
            except asyncio.TimeoutError:
                await self._handle_error("Timeout waiting for Consul response")
            except Exception as e:
                await self._handle_error(f"Unexpected error: {e}")

    # ----------------------------------------------------------------------
    # 🔹 Public controls
    # ----------------------------------------------------------------------
    def start(self):
        """Start the watcher safely and keep a task reference."""
        if self._task and not self._task.done():
            logger.info(f"Watcher already running for {self.service_name}")
            return
        self._task = asyncio.create_task(self._watch_loop())

    def get_address(self):
        """Return the latest discovered address."""
        return self.current_addr

    def stop(self):
        """Stop watcher safely."""
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info(f"Watcher stopped for {self.service_name}")
