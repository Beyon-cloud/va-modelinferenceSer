from typing import Optional
from com.beyoncloud.models.model_service import ModelServiceLoader

# This instance will be set at FastAPI startup
modelServiceLoader: Optional[ModelServiceLoader] = None