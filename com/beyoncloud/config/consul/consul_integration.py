# consul_integration.py
from fastapi import FastAPI
from consul import Consul
import socket
from contextlib import asynccontextmanager

# Initialize the FastAPI app
#app = FastAPI()

# Initialize Consul client
consul = Consul()

# Use lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event: Register the service with Consul
    service_id = f"vis-nlpengineser-{socket.gethostname()}"
    service_name = "vis-nlpengineser"
    consul.agent.service.register(
        service_name,
        service_id=service_id,
        port=5107,  # Port your FastAPI app is running on
        tags=["vis-nlpengineser", "web"],
    )
    print(f"vis-nlpengineser registered with Consul: {service_name}")
    yield
    # Shutdown event: Deregister the service from Consul
    consul.agent.service.deregister(service_id)
    print(f"vis-nlpengineser deregistered from Consul: {service_id}")

# Apply the lifespan context manager to the FastAPI app
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return {"message": "Hello, vis-nlpengineser integrated with Consul!"}
