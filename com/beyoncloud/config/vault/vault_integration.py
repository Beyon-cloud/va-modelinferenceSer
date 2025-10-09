# vault_integration.py
import hvac
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Initialize FastAPI app
app = FastAPI()

# Initialize Vault client
client = hvac.Client(url="http://127.0.0.1:8200")

# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Fetch secrets from Vault during startup
        secret = client.secrets.kv.v2.read_secret_version(path="my-secrets/fastapi")
        secret_value = secret['data']['data']['secret_key']
        print(f"Fetched secret: {secret_value}")
        
        # Yield control to FastAPI (app running)
        yield
    except Exception as e:
        print(f"Error fetching secrets from Vault: {e}")

# Apply the lifespan context manager to the FastAPI app
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return {"message": "FastAPI integrated with Vault!"}
