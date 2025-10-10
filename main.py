import uvicorn
import logging
from fastapi import FastAPI
import sys
import os
import com.beyoncloud.config.settings.env_config as config

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from com.beyoncloud.api.v1.routes import v1_router

from com.beyoncloud.config.consul.consul_integration import app as consul_app

from com.beyoncloud.config.vault.vault_integration import app as vault_app
from app_lifespan import lifespan
from com.beyoncloud.logger.logging_config import initialize_logging

logger = logging.getLogger(__name__)

# Configure logging once here
initialize_logging()

app = FastAPI(title="Virtual Assistant API NLP Engine",lifespan=lifespan)

app.mount("/consul", consul_app)
app.mount("/vault", vault_app)

# Register API routes
app.include_router(v1_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Welcome to the Virtual Assistant Service !"}
    
@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    
    uvcPort = config.PORT
    uvcHost = config.HOST
    uvicorn.run(app, host=uvcHost, port=uvcPort)