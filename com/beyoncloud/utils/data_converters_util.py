import json
from typing import Any, Type, Dict
from pydantic import BaseModel, ValidationError
from com.beyoncloud.utils.file_util import YamlLoader, JsonLoader

class ModelConverter:

    def __init__(self):
        pass

    async def convert_json_to_basemodel(self, json_fpath: str, baseModel: Type[BaseModel]) -> BaseModel:
        try:
            json_loader = JsonLoader()
            jsondata = json_loader.get_json_object(json_fpath)

            # Validate and convert to BaseModel
            basemodel_instance = await self.convert_dict_to_basemodel(jsondata, baseModel)
            return basemodel_instance

        except FileNotFoundError:
            print(f"File not found: {json_fpath}")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON format in file '{json_fpath}': {e}")
        except Exception as e:
            print(f"Unexpected error in convert_json_to_basemodel: {e}")
        return None

    async def convert_dict_to_basemodel(self, dictData: Dict[str, Any], baseModel: Type[BaseModel]) -> BaseModel:
        try:
            basemodel_instance = baseModel(**dictData)
            return basemodel_instance

        except ValidationError as ve:
            print(f"Validation failed while creating Pydantic model:\n{ve}")
        except TypeError as te:
            print(f"Type error: {te}")
        except Exception as e:
            print(f"Unexpected error in convert_dict_to_basemodel: {e}")
        return None