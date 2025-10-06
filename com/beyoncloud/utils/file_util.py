import logging
import yaml
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from com.beyoncloud.common.constants import Delimiter, FileTypes, CommonPatterns, FileFormats
import com.beyoncloud.config.settings.env_config as config

logger = logging.getLogger(__name__)

class YamlLoader:
    def __init__(self):
        pass

    def get_yaml_object(self, file_path: str):
        """
        # Example usage:
        # fetcher = YamlLoader()
        # data = fetcher.get_yaml_object('your_yaml_file.yaml')
        """
        try:
            with open(file_path, 'r', encoding="utf-8") as file:
                yamldata = yaml.safe_load(file)
            return yamldata
        except FileNotFoundError:
            logger.error(f"Error: The file '{file_path}' was not found.")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

    def get_dict_data(self, file_path: str, key: str):
        data = self.get_yaml_object(file_path)
        return data.get(key, {})

class JsonLoader:
    def __init__(self):
        pass

    def get_json_object(self, file_path: str):
        """
        # Example usage:
        # fetcher = JsonLoader()
        # data = fetcher.get_json_object('your_yaml_file.yaml')
        """
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                jsonData = json.load(f)
            return jsonData
        except FileNotFoundError:
            logger.error(f"Error: The file '{file_path}' was not found.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")


class TextLoader:
    """
    Example usage:
    loader = TextLoader()
    file_content = loader.get_text_content('your_text_file.txt')
    if file_content:
        print(file_content)
    """
    def __init__(self):
        pass

    def get_text_content(self, file_path: str):
        """
        Loads the content of a text file and returns it as a single string.

        Args:
            file_path (str): The path to the text file.

        Returns:
            str: The content of the file, or None if an error occurs.
        """
        try:
            # Check if the file exists before attempting to open it
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

            with open(file_path, "r", encoding='utf-8') as f:
                text_data = f.read()

            return text_data
        except FileNotFoundError as e:
            logger.error(f"Error: The file '{file_path}' was not found.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return None


class FileCreation:

    def __init__(self):
        pass

    def is_directory(self, dir_path: str) -> bool:
        dir_path = Path(dir_path)

        # Check directory
        if dir_path.is_dir():
            return True
        else:
            return False

    def is_file(self, file_path: str) -> bool:
        file_path = Path(file_path)

        # Check directory
        if file_path.is_file():
            return True
        else:
            return False

    def create_text_file(self, dir_path: str, filename: str, text_content: str) -> str:
        """
        Create a text file with the given content.

        Args:
            dir_path (str): Target directory path. If it doesn't exist, it will be created.
            filename (str): Name of the file to create (should include extension, e.g., "output.txt").
            text_content (str): Text content to write into the file.

        Returns:
            str: The full path to the created output file.

        Raises:
            OSError: If the file cannot be created or written.
            ValueError: If filename is empty.
        """
        # Validate inputs
        if not filename:
            raise ValueError("filename must be a non-empty string")

        # Ensure the directory exists
        os.makedirs(dir_path, exist_ok=True)

        # Build the full path
        output_filepath = os.path.join(dir_path, filename)

        # Write content to file with UTF-8 encoding
        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(text_content)
                f.flush()  # ensure content is written to disk
        except OSError as e:
            # Re-raise with additional context if desired
            raise OSError(f"Failed to write text file at {output_filepath}: {e}") from e

        return output_filepath

    def create_json_file(self, dir_path: str, filename: str, json_content: Dict[str, Any]) -> str:
        """
        Creates a JSON file with the given content.

        Args:
            dir_path (str): Directory path where file will be stored.
            filename (str): Name of the file (should end with .json).
            json_content (Dict[str, Any]): Data to save as JSON.

        Returns:
            str: Full path of the created file.
        """
        os.makedirs(dir_path, exist_ok=True)
        output_filepath = os.path.join(dir_path, filename)

        # Ensure .json extension
        if not filename.lower().endswith(".json"):
            output_filepath += ".json"

        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(json_content, f, indent=4, ensure_ascii=False)

        return output_filepath

    @staticmethod
    def generate_file_name(
        output_filename=None, 
        flex_name1=None, 
        flex_name2=None, 
        flex_name3=None, 
        file_extension=".txt"
    ) -> str:
        """
        Generates a unique output file name for OCR results.

        Args:
            output_filename (str): Optional explicit output filename.
            flex_name1 (str): Flexible name field 1 (e.g., ref_name).
            flex_name2 (str): Flexible name field 2 (e.g., batch_id).
            flex_name3 (str): Flexible name field 3 (e.g., reference_id).
            file_extension (str): File extension (e.g., .json, .txt, .pdf).

        Returns:
            str: Generated file name.
        """
        try:
            if output_filename and output_filename.strip():
                return output_filename.strip()

            # Collect available parts (ignore None/empty)
            name_parts = [
                part.strip() for part in [flex_name1, flex_name2, flex_name3] if part and part.strip()
            ]

            if not name_parts:
                raise ValueError("No valid name parts provided for filename generation")

            generated_name = "_".join(name_parts) + file_extension
            return generated_name
        except Exception as e:
            logger.error(f"Error generating file name: {e}")
            raise

class FetchContent:

    def __init__(self):
        pass

    def fetch_schema_content(self, input_data: str, file_format: str):
        """
        Fetch and parse content based on filetype.
        Returns JSON object (dict or list) if output_filetype is JSON,
        else returns string.
        """
        extracted_data = None

        if input_data:
            delimiter = Delimiter.JSON
            if FileFormats.XLSX == file_format:
                delimiter = Delimiter.XLSX
            elif FileFormats.CSV == file_format:
                delimiter = Delimiter.CSV

            try:
                if FileFormats.CSV == file_format:
                    output_data = self._clean_csv_response(input_data, delimiter)
                    data_obj = output_data
                else:
                    output_data = self._clean_json_response(input_data, delimiter)
                    if isinstance(output_data, (dict, list)):
                        data_obj = output_data
                    else:
                        data_obj = json.loads(output_data)

                print(f"output_data ------> {output_data}")
                print(f"json_obj ---> {data_obj}")


                extracted_data = data_obj   # return structured object
            except json.JSONDecodeError as e:
                print("JSONDecodeError:")
                print(f"  Error: {e.msg}")          # error message
                print(f"  Location: line {getattr(e, 'lineno', '?')}, column {getattr(e, 'colno', '?')}")
                print(f"  Input type: {type(output_data)!r}")
                if isinstance(output_data, str):
                    # fallback to raw cleaned string
                    extracted_data = {"raw_text": output_data}
                else:
                    raise ValueError("JSON Decoding error")

        return extracted_data

    def _clean_json_response(self, input_data: str, delimiter: str) -> str:
        """Clean the response to extract valid JSON"""
        
        # Remove any text before the first { and after the last }
        input_content = input_data.strip()
        if delimiter in input_content:
            input_content = self._extract_structure_content_only(input_data.strip(), delimiter)
            
        print(f"input_content -----------> {input_content}")

        # Find first { and last }
        start_idx = input_content.find('{')
        end_idx = input_content.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = input_content[start_idx:end_idx + 1]
            return json_str
        
        # If no clear JSON structure, return the response as is
        return input_content

    def _clean_csv_response(self, input_data: str, delimiter: str) -> str:
        """Clean the response to extract valid JSON"""
        
        # Remove any text before the first { and after the last }
        input_content = input_data.strip()
        if delimiter in input_content:
            input_content = self._extract_structure_content_only(input_data.strip(), delimiter)
            
        print(f"CSV input_content -----------> {input_content}")

        # Find first { and last }
        start_idx = input_content.find('{')
        end_idx = input_content.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = input_content[start_idx:end_idx + 1]
            return json_str
        
        # If no clear JSON structure, return the response as is
        return input_content

    def _extract_structure_content_only(self, input_data: str, delimiter: str) -> str:

        """
        Extracts the first number from a string that is enclosed by '```' symbols.

        Args:
            input_string (str): The string to process.

        Returns:
            str: The extracted number as a string, or None if no match is found.
        """
        # The regex pattern looks for a sequence of digits (\d+) that
        # is preceded and followed by '```' symbols.
        match = re.search(fr'{delimiter}(.*?){delimiter}', input_data, re.DOTALL)
        if match:
            return match.group(1)
        return CommonPatterns.EMPTY_SPACE


    def fetch_ocr_content(self, filepath: str) -> str:

        if not filepath:
            logger.info("The given filepath is empty")
            return CommonPatterns.EMPTY_SPACE

        json_loader = JsonLoader()
        json_data = json_loader.get_json_object(filepath)

        if not json_data:
            logger.info(f"The given file have empty data. Filepath is {filepath} ")
            return CommonPatterns.EMPTY_SPACE

        full_text = "".join(page.get("text", "") for page in json_data.get("results", []))
        
        return full_text


class PathValidator:

    def __init__(self):
        pass

    def is_directory(self, dir_path: str) -> bool:
        """
        Check if the given path is a valid directory.
        Returns False if path is invalid or inaccessible.
        """

        if not dir_path:
            return False

        try:
            dir_path = Path(dir_path)
            return dir_path.is_dir()
        except Exception as e:
            logger.error(f"Error validating directory path '{dir_path}': {e}")
            return False

    def is_file(self, file_path: str) -> bool:
        """
        Check if the given path is a valid file.
        Returns False if path is invalid or inaccessible.
        """
        try:
            file_path = Path(file_path)
            return file_path.is_file()
        except Exception as e:
            logger.error(f"Error validating file path '{file_path}': {e}")
            return False