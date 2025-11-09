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
        # Intentionally empty for now.
        # Reason: This class does not require instance state at construction
        # and will initialize attributes lazily when the analysis runs.
        # If future attributes are needed, initialize them here.
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
        # Intentionally empty for now.
        # Reason: This class does not require instance state at construction
        # and will initialize attributes lazily when the analysis runs.
        # If future attributes are needed, initialize them here.
        pass

    def get_json_object(self, file_path: str):
        """
        # Example usage:
        # fetcher = JsonLoader()
        # data = fetcher.get_json_object('your_yaml_file.yaml')
        """
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                json_data = json.load(f)
            return json_data
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
        # Intentionally empty for now.
        # Reason: This class does not require instance state at construction
        # and will initialize attributes lazily when the analysis runs.
        # If future attributes are needed, initialize them here.
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
        # Intentionally empty for now.
        # Reason: This class does not require instance state at construction
        # and will initialize attributes lazily when the analysis runs.
        # If future attributes are needed, initialize them here.
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
        """No instance attributes required at initialization."""
        pass

    def fetch_schema_content(self, input_data: str, file_format: str):
        """
        Fetch and parse content based on filetype.
        Returns JSON object (dict or list) if output_filetype is JSON,
        else returns string.
        """
        if not input_data:
            return None

        delimiter = self._get_delimiter(file_format)
        try:
            output_data = self._clean_content(input_data, file_format, delimiter)
            extracted_data = self._parse_output(output_data)
        except json.JSONDecodeError as e:
            logger.error(
                f"JSONDecodeError: {e.msg}, line={getattr(e, 'lineno', '?')}, col={getattr(e, 'colno', '?')}"
            )
            extracted_data = self._handle_json_error(output_data)

        return extracted_data

    # -----------------------------
    # Helper methods
    # -----------------------------
    def _get_delimiter(self, file_format: str) -> str:
        """Map file format to corresponding delimiter."""
        return {
            FileFormats.XLSX: Delimiter.XLSX,
            FileFormats.CSV: Delimiter.CSV,
        }.get(file_format, Delimiter.JSON)

    def _clean_content(self, input_data: str, file_format: str, delimiter: str):
        """Dispatch content cleaning by file type."""
        clean_method = (
            self._clean_csv_response
            if file_format == FileFormats.CSV
            else self._clean_json_response
        )
        return clean_method(input_data, delimiter)

    def _parse_output(self, output_data: str):
        """Convert cleaned string into a structured JSON object if possible."""

        # If already a Python dict or list, return directly
        if isinstance(output_data, (dict, list)):
            return output_data

        try:
            # Step 1: Ensure it's a string
            if not isinstance(output_data, str):
                output_data = str(output_data)

            # Step 2: Remove JavaScript-style comments (// ...)
            cleaned_output_data = self.remove_json_comments(output_data)

            # Step 3: Optionally strip leading/trailing spaces
            cleaned_output_data = cleaned_output_data.strip()

            print(f"_parse_output cleaned_output_data -----> : {cleaned_output_data}")

            # Step 5: Try parsing as JSON
            return json.loads(cleaned_output_data)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in '_parse_output': {e}")
            # Optionally print a small snippet for debugging
            snippet = cleaned_output_data[:200].replace("\n", "\\n")
            logger.debug(f"Offending JSON snippet: {snippet}")
            return {"raw_text": output_data}

        except Exception as e:
            logger.error(f"Error load JSON data '_parse_output': {e}")
            return {"raw_text": output_data}

    def remove_json_comments(self, text: str) -> str:
        result = []
        in_string = False
        escaped = False
        i = 0
        length = len(text)

        while i < length:
            char = text[i]

            # Handle quote toggling
            if char == '"' and not escaped:
                in_string = not in_string
                result.append(char)
                i += 1
                continue

            # Handle escaping (e.g. \" inside strings)
            if char == "\\" and not escaped:
                escaped = True
                result.append(char)
                i += 1
                continue
            else:
                escaped = False

            # Detect comment start only when NOT in string
            if not in_string and char == '/' and i + 1 < length and text[i + 1] == '/':
                # Skip everything till newline
                i += 2
                while i < length and text[i] != '\n':
                    i += 1
                continue

            # Default: append character
            result.append(char)
            i += 1

        return ''.join(result)


    def _handle_json_error(self, output_data: str):
        """Handle JSON parsing errors gracefully."""
        if isinstance(output_data, str):
            logger.error(f"Error load JSON data '_handle_json_error' : {e}")
            return {"raw_text": output_data}
        raise ValueError("JSON Decoding error")

    def _clean_json_response(self, input_data: str, delimiter: str) -> str:
        """Clean the response to extract valid JSON."""
        return self._clean_delimited_block(input_data, delimiter, "JSON")

    def _clean_csv_response(self, input_data: str, delimiter: str) -> str:
        """Clean the CSV-like response to extract valid JSON."""
        return self._clean_delimited_block(input_data, delimiter, "CSV")

    def _clean_delimited_block(self, input_data: str, delimiter: str, log_label: str) -> str:
        """Extract content between delimiters and isolate valid JSON."""
        content = input_data.strip()
        print(f"Before clean ---> {content}")
        if delimiter in content:
            content = self._extract_structure_content_only(content, delimiter)

        print(f"After clean ---> {content}")
        logger.debug(f"{log_label} cleaned content -----------> {content}")
        start_idx, end_idx = content.find("{"), content.rfind("}")
        return content[start_idx:end_idx + 1] if start_idx != -1 < end_idx else content

    def _extract_structure_content_only(self, input_data: str, delimiter: str) -> str:
        """Extract text enclosed within ```json ... ``` block."""
        delimiter_escaped = re.escape(delimiter)

        # Match ```json\n ... ```
        pattern = fr"{delimiter_escaped}json\s*\n(.*?){delimiter_escaped}"
        match = re.search(pattern, input_data, re.DOTALL)

        if match:
            return match.group(1).strip()

        # Fallback: try plain ```...```
        pattern_no_lang = fr"{delimiter_escaped}(.*?){delimiter_escaped}"
        match = re.search(pattern_no_lang, input_data, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Final fallback: return everything
        return input_data.strip()


    def fetch_ocr_content(self, filepath: str) -> str:
        """Extracts text from OCR-generated JSON file."""
        if not filepath:
            logger.info("The given filepath is empty")
            return CommonPatterns.EMPTY_SPACE

        json_loader = JsonLoader()
        json_data = json_loader.get_json_object(filepath)
        if not json_data:
            logger.info(f"The given file has empty data. Filepath: {filepath}")
            return CommonPatterns.EMPTY_SPACE

        return "".join(page.get("text", "") for page in json_data.get("results", []))

class PathValidator:

    def __init__(self):
        # Intentionally empty for now.
        # Reason: This class does not require instance state at construction
        # and will initialize attributes lazily when the analysis runs.
        # If future attributes are needed, initialize them here.
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