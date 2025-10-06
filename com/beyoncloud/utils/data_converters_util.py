import json
import asyncio
from typing import Any, Type, Dict
from pydantic import BaseModel, ValidationError
from com.beyoncloud.utils.file_util import YamlLoader, JsonLoader

class ModelConverter:

    def __init__(self):
        pass

    async def convert_json_to_basemodel(self, json_fpath: str, base_model: Type[BaseModel]) -> BaseModel:
        try:
            json_loader = JsonLoader()
            jsondata = json_loader.get_json_object(json_fpath)

            # Validate and convert to BaseModel
            basemodel_instance = await self.convert_dict_to_basemodel(jsondata, base_model)
            return basemodel_instance

        except FileNotFoundError:
            print(f"File not found: {json_fpath}")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON format in file '{json_fpath}': {e}")
        except Exception as e:
            print(f"Unexpected error in convert_json_to_basemodel: {e}")

        # Fallback: return a default/empty instance to satisfy return type
        try:
            return base_model()  # type: ignore[call-arg]
        except Exception as e:
            # If even constructing a default instance fails, raise a clear error
            raise RuntimeError("Failed to construct a default BaseModel instance") from e

    async def convert_dict_to_basemodel(
        self, dict_data: Dict[str, Any], base_model: Type[BaseModel]
    ) -> BaseModel:
        # Run the synchronous construction in the default thread pool
        loop = asyncio.get_event_loop()

        def _construct():
            try:
                return base_model(**dict_data)
            except ValidationError:
                # Re-raise to be handled by outer except blocks if needed
                raise
            except Exception as e:
                # Wrap other exceptions to be handled by caller
                print(f"Unexpected error: {e}")
                raise

        try:
            basemodel_instance = await loop.run_in_executor(None, _construct)
            return basemodel_instance
        except ValidationError as ve:
            print(f"Validation failed while creating Pydantic model:\n{ve}")
        except TypeError as te:
            print(f"Type error: {te}")
        except Exception as e:
            print(f"Unexpected error in convert_dict_to_basemodel: {e}")

        # Fallback: return a default instance to satisfy return type
        try:
            return base_model()
        except Exception as e:
            raise RuntimeError("Failed to construct a default BaseModel instance") from e