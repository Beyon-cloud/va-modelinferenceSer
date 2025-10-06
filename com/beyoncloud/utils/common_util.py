import json
from typing import Any, Type, Dict, List
from dacite import from_dict
from PIL import Image
from pathlib import Path
from dataclasses import asdict, is_dataclass
from langchain.schema import Document
from pydantic import BaseModel

@staticmethod
def convert_json_to_dataclass(data_class: Type[Any], data_payload: dict) -> Any:
    """
    Converts a dictionary (JSON payload) into an instance of the given dataclass.

    Args:
        data_class (Type[Any]): The dataclass type to instantiate.
        data_payload (dict): JSON data as a dictionary.

    Returns:
        Any: Instance of the specified dataclass populated from the dict.
    """
    return from_dict(data_class=data_class, data=data_payload)

@staticmethod
def convert_dataclass_to_dict(data_instance: Any) -> Dict[str, Any]:
    """
    Converts a dataclass instance to a dictionary.

    Args:
        data_instance (Any): An instance of a dataclass to be converted.

    Returns:
        Dict[str, Any]: A dictionary representation of the dataclass,
                        where keys are field names and values are field values.

    Raises:
        TypeError: If the input is not a dataclass instance.
    """
    if not is_dataclass(data_instance):
        raise TypeError("Provided input is not a dataclass instance.")

    return asdict(data_instance)

@staticmethod
def convert_dataclass_to_json(data_instance: Any) -> str:
    """
    Converts a dataclass instance to a JSON-formatted string.

    Args:
        data_instance (Any): An instance of a dataclass to be converted.

    Returns:
        str: A JSON string representation of the dataclass.

    Raises:
        TypeError: If the input is not a dataclass instance.
    """
    if not is_dataclass(data_instance):
        raise TypeError("Provided input is not a dataclass instance.")
    
    return json.dumps(asdict(data_instance), indent=2, default=str)

@staticmethod
def convertPathToImageMulti(lstFilepath: List[str]):
    return [Image.open(Path(path)) for path in lstFilepath]

@staticmethod
def convert_path_to_image_single(filepath: str):
    return [Image.open(Path(filepath))]

@staticmethod
def convert_text_to_document(text: str) -> Document:
    return [Document(page_content=text)]

@staticmethod
def convert_mulit_text_to_document(textList: List[str]) -> List[Document]:
    documentList = [Document(page_content=text) for text in textList]
    return documentList

@staticmethod
def convert_basemodel_to_dict(bsaeModel: BaseModel) -> Dict[str, Any]:
    dictOutput = bsaeModel.model_dump()
    return dictOutput

@staticmethod
def convert_dict_to_basemodel(dictData, baseModel: Type[BaseModel]) -> BaseModel:
    baseModelInstance = None
    try:
        baseModelInstance = baseModel(**dictData)
        
    except Exception as e:
        print(f"Error creating Pydantic model: {e}")
        
    return baseModelInstance

