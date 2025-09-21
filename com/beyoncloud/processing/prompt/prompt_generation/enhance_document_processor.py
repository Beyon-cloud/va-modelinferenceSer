import logging
from pathlib import Path
import re
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from com.beyoncloud.processing.prompt.prompt_generation.huggingface_llama_connect import HuggingFaceLlama3Client
from com.beyoncloud.schemas.prompt_gen_reqres_datamodel import (
    DocumentType,FieldType, SchemaField,DocumentSchema,ExtractionResult,ProcessingConfig,
    SchemaPromptRequest, EntityPromptRequest
) 
from com.beyoncloud.processing.prompt.prompt_generation.template_mapping  import PropertySchemaTemplate
from com.beyoncloud.utils.file_util import TextLoader
from com.beyoncloud.models.model_service import ModelServiceLoader
from com.beyoncloud.models import model_singleton
from com.beyoncloud.processing.prompt.prompt_template import (
    get_schema, get_prompt_template, get_prompt_param,
    SafeDict
)
from com.beyoncloud.common.constants import Delimiter, PromptType
from com.beyoncloud.processing.rag_process_impl import RagProcessImpl
from com.beyoncloud.schemas.rag_reqres_data_model import StructureInputData, EntityResponse, StructureInputDataBuilder


logger = logging.getLogger(__name__)

class EnhancedDocumentProcessor:
    """Complete document processor with Llama3 integration"""
    
    def __init__(self):
        hf_api_key: str = ""
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct:novita",
        config_type: str = "balanced"

        # Initialize Hugging Face Llama3 client
        self.llama3_client = HuggingFaceLlama3Client(
            api_key=hf_api_key,
            model_name=model_name
        )

        self.config = ProcessingConfig()
        self.schema_templates = {
            DocumentType.PROPERTY: PropertySchemaTemplate()
        }
        
        # Create output directory if specified
        if self.config.output_directory:
            Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
    
    async def extract_from_document(
        self, 
        file_path: str, 
        doc_type: Optional[str] = None,
        detection_hints: Optional[List[str]] = None,
        custom_requirements: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        Complete document extraction workflow
        """
        
        result = ExtractionResult(success=False)
        processing_start_time = datetime.now()
        
        try:
            # Step 1: Load document content
            logger.info(f"Loading document: {file_path}")
            content =""
            file_metadata = {}
            if file_path:
                text_loader = TextLoader()
                content = text_loader.get_text_content(file_path)

            print(f"fullContext --> {content}")
            
            if not content:
                raise ValueError("Document content is empty or could not be extracted")
            
            # Step 2: Detect document type
            if doc_type:
                document_type = DocumentType(doc_type)
                logger.info(f"Using provided document type: {doc_type}")
            else:
                document_type = self.detect_document_type(content, detection_hints)
                logger.info(f"Detected document type: {document_type.value}")
            
            # Step 3: Generate schema
            schema = self.generate_schema(document_type, custom_requirements)

            schema_json = await self.generate_schema_template(content, file_path)
            print(f"AFTER ##################################################################### {schema_json}")
            # Step 4: Generate prompts
            prompts = self.generate_extraction_prompts(schema, content, document_type, schema_json)
            
            # Step 5: Extract with Llama3
            if self.llama3_client:
                extracted_data = self._extract_with_llama3(prompts, result)
                if extracted_data:
                    result.extracted_data = extracted_data
                    result.success = True
            else:
                logger.warning("No Llama3 client provided, returning prompts only")
                result.success = True
                result.extracted_data = {"prompts_generated": True}
            
            # Step 6: Validate extraction
            if self.config.enable_validation and result.extracted_data and "prompts_generated" not in result.extracted_data:
                validation_result = self._validate_extraction(result.extracted_data, schema)
                result.validation_result = validation_result
                
                # Update confidence score based on validation
                if validation_result:
                    result.confidence_score = validation_result.get("completeness_score", 0.0)
            
            # Step 7: Post-process results
            if self.config.enable_post_processing and result.extracted_data:
                result = self._post_process_results(result, schema)
            
            # Step 8: Save intermediate results
            if self.config.save_intermediate_results:
                self._save_intermediate_results(file_path, result, prompts)
            
            # Update result metadata
            result.schema_used = schema
            result.processing_metadata = {
                "file_metadata": file_metadata,
                "document_type": document_type.value,
                "processing_time_seconds": (datetime.now() - processing_start_time).total_seconds(),
                "schema_field_count": len(schema.fields),
                "content_length": len(content),
                "prompts_used": {
                    "system_prompt_length": len(prompts["system_prompt"]),
                    "user_prompt_length": len(prompts["user_prompt"])
                }
            }
            
            if result.success:
                logger.info(f"Document extraction completed successfully for {file_path}")
            else:
                logger.warning(f"Document extraction completed with issues for {file_path}")
            
        except Exception as e:
            logger.error(f"Document extraction failed for {file_path}: {e}")
            result.errors.append(str(e))
            result.success = False
        
        return result
    
    def detect_document_type(self, content: str, hints: Optional[List[str]] = None) -> DocumentType:
        """Detect document type based on content analysis"""
        content_lower = content.lower()
        
        # Property keywords
        property_keywords = ["property", "real estate", "deed", "mortgage", "listing", "bedrooms", "bathrooms", "square feet", "mls"]
        property_score = sum(2 for keyword in property_keywords if keyword in content_lower)
        
        # Insurance keywords  
        insurance_keywords = ["policy", "premium", "coverage", "deductible", "claim", "beneficiary", "policyholder"]
        insurance_score = sum(2 for keyword in insurance_keywords if keyword in content_lower)
        
        # Education keywords
        education_keywords = ["transcript", "diploma", "gpa", "course", "credit", "degree", "graduation", "student"]
        education_score = sum(2 for keyword in education_keywords if keyword in content_lower)
        
        # Apply hints
        if hints:
            for hint in hints:
                hint_lower = hint.lower()
                if any(word in hint_lower for word in ["property", "real estate"]):
                    property_score += 5
                elif any(word in hint_lower for word in ["insurance", "policy"]):
                    insurance_score += 5
                elif any(word in hint_lower for word in ["education", "school", "university"]):
                    education_score += 5
        
        # Return highest scoring type
        scores = {
            DocumentType.PROPERTY: property_score,
            DocumentType.INSURANCE: insurance_score,
            DocumentType.EDUCATION: education_score
        }
        
        detected_type = max(scores, key=scores.get)
        logger.info(f"Document type detection scores: {scores}")
        
        return detected_type
    
    def generate_schema(self, document_type: DocumentType, custom_requirements: Optional[Dict[str, Any]]) -> DocumentSchema:
        """Generate schema for document type"""
        if document_type not in self.schema_templates:
            raise ValueError(f"Schema template not available for document type: {document_type.value}")
        
        schema = self.schema_templates[document_type].get_schema()
        
        # Apply custom requirements
        if custom_requirements:
            schema = self._apply_custom_requirements(schema, custom_requirements)
        
        return schema

    async def generate_schema_template(self, schema_prompt_request: SchemaPromptRequest) -> str:
        """
        Build a valid SchemaPromptRequest and call RagProcessImpl.generate_structured_response.
        Returns extracted_data (with braces escaped) or an empty string on failure.
        """
        print("####### generate_schema_template #########")
        file_path = schema_prompt_request.source_path
        # Basic validation of inputs
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path is required and must be a non-empty string")

        # Build the entity request using the builder pattern, ensuring the builder returns expected shape.
        structure_input_data = (
            StructureInputDataBuilder()
            .with_source_path(schema_prompt_request.source_path)
            .with_prompt_type(PromptType.SCHEMA_PROMPT)
            .with_organization_id(schema_prompt_request.organization_id)
            .with_domain_id(schema_prompt_request.domain_id)
            .with_document_type(schema_prompt_request.document_type)
            .with_user_id(schema_prompt_request.user_id)
            .with_source_lang(schema_prompt_request.source_lang)
            .with_output_mode(schema_prompt_request.desired_output_mode)
            .with_output_format(schema_prompt_request.desired_output_format)
            .build()
        )
        print(f"####### structure_input_data ######### --> {structure_input_data}")
        ragProcessImpl = RagProcessImpl()

        # Call the async method and handle possible exceptions
        response = None
        try:
            response = await ragProcessImpl.generate_structured_response(structure_input_data)
        except Exception as exc:
            # Log or re-raise depending on your application's logging strategy
            # For example: logger.exception("generate_structured_response failed")
            raise

        extracted_data = ""
        if response:
            # Clean and parse response safely
            try:
                cleaned_response = self._clean_json_response(response)
                # If _clean_json_response returns a JSON string, ensure it's valid json
                # It might already be a dict — handle both cases
                if isinstance(cleaned_response, (dict, list)):
                    json_obj = cleaned_response
                else:
                    json_obj = json.loads(cleaned_response)

                # Serialize back to string (pretty if needed)
                json_str = json.dumps(json_obj, ensure_ascii=False)

                # Escape braces for templating systems that use double braces
                #extracted_data = json_str.replace("{", "{{").replace("}", "}}")
                extracted_data = json_str
            except json.JSONDecodeError:
                # If response isn't valid JSON after cleaning, you can choose to fallback:
                # - keep raw cleaned_response
                # - raise an error
                # Here we default to using the raw cleaned_response (escaped) if it's a string
                if isinstance(cleaned_response, str):
                    #extracted_data = cleaned_response.replace("{", "{{").replace("}", "}}")
                    extracted_data = cleaned_response
                else:
                    # re-raise so the caller can see the issue
                    raise ValueError("JSON Decoading error")

        return extracted_data

    def _apply_custom_requirements(self, schema: DocumentSchema, requirements: Dict[str, Any]) -> DocumentSchema:
        """Apply custom requirements to schema"""
        # Add additional fields
        if "additional_fields" in requirements:
            for field_def in requirements["additional_fields"]:
                new_field = SchemaField(
                    name=field_def["name"],
                    field_type=FieldType(field_def["type"]),
                    description=field_def["description"],
                    required=field_def.get("required", False)
                )
                schema.fields.append(new_field)
        
        # Modify existing fields
        if "field_modifications" in requirements:
            for field_name, modifications in requirements["field_modifications"].items():
                for field in schema.fields:
                    if field.name == field_name:
                        if "required" in modifications:
                            field.required = modifications["required"]
                        if "description" in modifications:
                            field.description = modifications["description"]
        
        return schema
    
    async def generate_extraction_prompts(self, schema: DocumentSchema, content: str, 
        document_type: DocumentType, 
        schema_template,
        structure_input_data: StructureInputData
    ) -> Dict[str, str]:
        """Generate extraction prompts"""
        print(f"generate_extraction_prompts inside")
        prompt_output = get_prompt_template(
            structure_input_data.domain_id, structure_input_data.document_type, 
            structure_input_data.organization_id, structure_input_data.prompt_type 
        )
        print(f"prompt_output --------> {prompt_output}")

        system_prompt_template = prompt_output["system_prompt_template"]
        user_prompt_template = prompt_output["user_prompt_template"]
        input_variables = prompt_output["input_variables"]
        prompt_id = prompt_output["prompt_id"]

        print(f"input_variables1 -->{prompt_id} -  {input_variables}")
        param_result = await get_prompt_param(prompt_id)
        print(f"param_result --> {param_result}")

        variable_map = {
            "schema_template": schema_template
        }
        for result in param_result:
            param_key = getattr(result, "param_key", None)
            param_value = getattr(result, "param_value", None)
            #print(f"param_key --> {param_key} - {param_value}")
            if param_key is not None:
                variable_map[param_key] = param_value

        system_prompt = system_prompt_template.format_map(SafeDict(variable_map))
        user_prompt = user_prompt_template.format_map(SafeDict(variable_map))

        #template = self.schema_templates[document_type]
        #priorities = template.get_extraction_priorities()
        #validation_checks = template.get_validation_checks()
        
        #system_prompt = self._build_system_prompt(schema, priorities, validation_checks)
        #user_prompt = self._build_user_prompt(schema, content, priorities, schema_template)
        
        print(f"system_prompt ---########################-> {system_prompt}")
        print(f"user_prompt -###########################-> {user_prompt}")
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "input_variables": "context"
        }
    
    def _build_system_prompt(self, schema: DocumentSchema, priorities: List[str], validation_checks: List[str]) -> str:
        """Build system prompt for extraction"""
        parts = [
            f"You are an expert at extracting structured information from {schema.document_type.value} documents.",
            "Your task is to analyze the document content and extract information according to the provided schema.",
            "Return your response as valid JSON only, with no additional text or explanation.",
            " You are an AI assistant that extracts structured data from legal documents. " 
            "Your task is to carefully analyze the provided text and return only a valid JSON object following the schema below.  "
           " Ensure all values are extracted exactly as they appear, normalized when necessary (e.g., numbers as integers, dates in DD/MM/YYYY).  "
           " If a field is missing in the text, set it to null."
            "",
            "EXTRACTION GUIDELINES:",
            "1. Extract only information that is explicitly stated in the document",
            "2. Use null for missing required fields that cannot be found",
            "3. Follow the exact field names and data types specified in the schema",
            "4. For dates, use ISO format (YYYY-MM-DD) or DD-MM-YYYY or DD/MM/YY when possible",
            "5. For currency amounts, include the numeric value with appropriate currency symbol",
            "6. Be precise and accurate - double-check all extracted values",
            "",
            f"PRIORITY FIELDS (focus on these first): {', '.join(priorities)}",
            "",
            "VALIDATION REQUIREMENTS:",
        ]
        
        parts.extend([f"- {check}" for check in validation_checks])
        parts.extend([
            "",
            "IMPORTANT: Return only valid JSON that matches the schema structure. Do not include any explanatory text."
        ])
        
        return "\n".join(parts)
    
    def _build_user_prompt(self, schema: DocumentSchema, content: str, priorities: List[str], schema_json) -> str:
        """Build user prompt for extraction"""
        #schema_json = self._schema_to_json(schema)

        parts = [
            f"Document Type: {schema.document_type.value.title()}",
            f"Schema Description: {schema.description}",
            "",
            "EXTRACTION SCHEMA (JSON format expected):",
            "```json",
            f"{schema_json}",
            "```",
            "",
            "DOCUMENT CONTENT TO ANALYZE:",
            "=" * 50,
            #content[:4000] + ("..." if len(content) > 4000 else ""),
            "{context}",
            "=" * 50,
            "",
            "TASK: Extract information from the document above and return it as JSON matching the schema.",
            f"Focus especially on these priority fields: {', '.join(priorities)}",
            "",
            "Return valid JSON only:"
        ]
        
        return "\n".join(parts)
    
     
    def _schema_to_json(self, schema: DocumentSchema) -> str:
        """Convert schema to JSON format"""
        schema_dict = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for field in schema.fields:
            field_def = {
                "type": self._convert_field_type_to_json_type(field.field_type),
                "description": field.description
            }
            
            if field.examples:
                field_def["examples"] = field.examples
            
            schema_dict["properties"][field.name] = field_def
            
            if field.required:
                schema_dict["required"].append(field.name)
        
            json_str = json.dumps(schema_dict, indent=2)
        return json_str.replace("{", "{{").replace("}", "}}")
    
    def _convert_field_type_to_json_type(self, field_type: FieldType) -> str:
        """Convert FieldType to JSON schema type"""
        type_mapping = {
            FieldType.STRING: "string",
            FieldType.NUMBER: "number",
            FieldType.DATE: "string",
            FieldType.BOOLEAN: "boolean",
            FieldType.ARRAY: "array",
            FieldType.OBJECT: "object",
            FieldType.EMAIL: "string",
            FieldType.PHONE: "string",
            FieldType.ADDRESS: "string",
            FieldType.CURRENCY: "string",
            FieldType.PERCENTAGE: "number"
        }
        return type_mapping.get(field_type, "string")
    
    def _extract_with_llama3(self, prompts: Dict[str, str], result: ExtractionResult) -> Optional[Dict[str, Any]]:
        """Extract data using Llama3"""
        max_retries = self.config.max_retries
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting extraction with Llama3 (attempt {attempt + 1}/{max_retries})")
                
                """
                response = self.llama3_client.generate_sync(
                    system_prompt=prompts["system_prompt"],
                    user_prompt=prompts["user_prompt"],
                    timeout=self.config.timeout_seconds,
                    temperature=0.1,
                    max_tokens=2048
                )"""

                allModelObjects = model_singleton.modelServiceLoader or ModelServiceLoader()
                llm = allModelObjects.get_hf_llama_model_pipeline()
                final_prompt = get_schema(prompts)
                print(f"final_prompt --> {final_prompt}")

                inputs = {
                    
                }

                chain = final_prompt | llm
                # Invoke asynchronously
                response = chain.invoke(
                    inputs
                )

                print(f"Response ------> \n")
                print(response)
                
                # Clean response and extract JSON
                cleaned_response = self._clean_json_response(response)
                extracted_data = json.loads(cleaned_response)
                
                logger.info("Extraction completed successfully")
                logger.info(f"Extracted fields: {list(extracted_data.keys())}")
                
                return extracted_data
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
                logger.warning(f"Raw response: {response[:200]}...")
                result.warnings.append(f"Attempt {attempt + 1} JSON parsing failed: {str(e)}")
                
            except Exception as e:
                logger.warning(f"Extraction attempt {attempt + 1} failed: {e}")
                result.warnings.append(f"Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt == max_retries - 1:
                result.errors.append(f"All {max_retries} extraction attempts failed")
        
        return None
    
    def _clean_json_response(self, response: str) -> str:
        """Clean the response to extract valid JSON"""
        # Remove any text before the first { and after the last }
        response_str = response.strip()
        if Delimiter.JSON in response_str:
            schema_str = self._extract_schema_only(response_str, Delimiter.JSON)
        else:
            schema_str = response_str
            
        print(f"schema_str --> {schema_str}")

        # Find first { and last }
        start_idx = schema_str.find('{')
        end_idx = schema_str.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = schema_str[start_idx:end_idx + 1]
            return json_str
        
        # If no clear JSON structure, return the response as is
        return response

    def _extract_schema_only(self, response: str, delimiter: str) -> str:

        """
        Extracts the first number from a string that is enclosed by '```' symbols.

        Args:
            input_string (str): The string to process.

        Returns:
            str: The extracted number as a string, or None if no match is found.
        """
        # The regex pattern looks for a sequence of digits (\d+) that
        # is preceded and followed by '```' symbols.
        match = re.search(fr'{delimiter}(.*?){delimiter}', response, re.DOTALL)
        if match:
            return match.group(1)
        return None
    
    def _validate_extraction(self, extracted_data: Dict[str, Any], schema: DocumentSchema) -> Dict[str, Any]:
        """Validate extracted data against schema"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "completeness_score": 0.0,
            "field_validations": {}
        }
        
        required_fields = [f.name for f in schema.fields if f.required]
        present_fields = [field for field in required_fields 
                         if field in extracted_data and extracted_data[field] is not None and extracted_data[field] != ""]
        
        # Calculate completeness
        validation_result["completeness_score"] = len(present_fields) / len(required_fields) if required_fields else 1.0
        
        # Check required fields
        missing_required = [field for field in required_fields 
                          if field not in extracted_data or extracted_data[field] is None or extracted_data[field] == ""]
        if missing_required:
            validation_result["errors"].extend([f"Missing required field: {field}" for field in missing_required])
            validation_result["is_valid"] = False
        
        # Validate field types and formats
        for field in schema.fields:
            if field.name in extracted_data and extracted_data[field.name] is not None:
                field_validation = self._validate_field_value(
                    extracted_data[field.name], field
                )
                validation_result["field_validations"][field.name] = field_validation
                
                if not field_validation["is_valid"]:
                    validation_result["errors"].extend(field_validation["errors"])
                    validation_result["is_valid"] = False
                
                if field_validation["warnings"]:
                    validation_result["warnings"].extend(field_validation["warnings"])
        
        return validation_result
    
    def _validate_field_value(self, value: Any, field: SchemaField) -> Dict[str, Any]:
        """Validate individual field value"""
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        if value is None or value == "":
            return validation
        
        # Type-specific validation
        if field.field_type == FieldType.DATE:
            if not self._is_valid_date(str(value)):
                validation["errors"].append(f"Invalid date format: {value}")
                validation["is_valid"] = False
        
        elif field.field_type == FieldType.EMAIL:
            if not self._is_valid_email(str(value)):
                validation["errors"].append(f"Invalid email format: {value}")
                validation["is_valid"] = False
        
        elif field.field_type == FieldType.CURRENCY:
            if not self._is_valid_currency(str(value)):
                validation["warnings"].append(f"Currency format may be incorrect: {value}")
        
        elif field.field_type == FieldType.NUMBER:
            try:
                float(str(value))
            except ValueError:
                validation["errors"].append(f"Invalid number format: {value}")
                validation["is_valid"] = False
        
        return validation
    
    def _is_valid_date(self, value: str) -> bool:
        """Validate date format"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # ISO format
            r'\d{2}/\d{2}/\d{4}',  # US format
            r'\d{2}-\d{2}-\d{4}',  # EU format
            r'\d{1,2}/\d{1,2}/\d{4}',  # Flexible US format
        ]
        return any(re.match(pattern, str(value).strip()) for pattern in date_patterns)
    
    def _is_valid_email(self, value: str) -> bool:
        """Validate email format"""
        pattern=""
       # pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
        return re.match(pattern, str(value).strip()) is not None
    
    def _is_valid_currency(self, value: str) -> bool:
        """Validate currency format"""
        pattern=""
        #pattern = r'[$€£¥]?[\d,]+\.?\d{0,2}
        return re.match(pattern, str(value).replace(' ', '')) is not None
    
    def _post_process_results(self, result: ExtractionResult, schema: DocumentSchema) -> ExtractionResult:
        """Post-process extraction results"""
        if not result.extracted_data:
            return result
        
        # Clean and standardize data formats
        processed_data = {}
        
        for field in schema.fields:
            if field.name in result.extracted_data:
                value = result.extracted_data[field.name]
                processed_value = self._clean_field_value(value, field)
                processed_data[field.name] = processed_value
            else:
                # Add null for missing fields
                processed_data[field.name] = None
        
        result.extracted_data = processed_data
        
        return result
    
    def _clean_field_value(self, value: Any, field: SchemaField) -> Any:
        """Clean and standardize field values"""
        if value is None or value == "":
            return None
        
        # Clean based on field type
        if field.field_type == FieldType.CURRENCY:
            # Standardize currency format
            if isinstance(value, str):
                # Extract numeric value and add standard formatting
                numeric_value = re.sub(r'[^\d.]', '', value)
                try:
                    return f"${float(numeric_value):,.2f}"
                except ValueError:
                    return str(value).strip()
        
        elif field.field_type == FieldType.DATE:
            # Standardize date format
            if isinstance(value, str):
                return self._standardize_date(value.strip())
        
        elif field.field_type == FieldType.STRING:
            return str(value).strip()
        
        elif field.field_type == FieldType.NUMBER:
            try:
                # Try to convert to appropriate number type
                if '.' in str(value):
                    return float(value)
                else:
                    return int(float(value))
            except ValueError:
                return value
        
        return value
    
    def _standardize_date(self, date_str: str) -> str:
        """Standardize date to ISO format"""
        common_formats = [
            "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", 
            "%B %d, %Y", "%d %B %Y", "%m-%d-%Y"
        ]
        
        for fmt in common_formats:
            try:
                date_obj = datetime.strptime(date_str.strip(), fmt)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        # If no format matches, return original
        return date_str
    
    def _save_intermediate_results(self, file_path: str, result: ExtractionResult, prompts: Dict[str, str]):
        """Save intermediate results to output directory"""
        if not self.config.output_directory:
            return
        
        base_name = Path(file_path).stem
        output_dir = Path(self.config.output_directory)
        
        # Save extraction result
        result_file = output_dir / f"{base_name}_extraction_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            # Convert result to dict for JSON serialization
            result_dict = {
                "success": result.success,
                "extracted_data": result.extracted_data,
                "validation_result": result.validation_result,
                "processing_metadata": result.processing_metadata,
                "errors": result.errors,
                "warnings": result.warnings,
                "confidence_score": result.confidence_score
            }
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        # Save prompts used
        prompts_file = output_dir / f"{base_name}_prompts.json"
        with open(prompts_file, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Intermediate results saved to {output_dir}")

    def batch_extract_documents(
        self, 
        file_paths: List[str],
        document_types: Optional[List[str]] = None,
        global_requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract from multiple documents in batch"""
        
        batch_start_time = datetime.now()
        results = []
        
        logger.info(f"Starting batch processing of {len(file_paths)} documents")
        
        for i, file_path in enumerate(file_paths):
            try:
                doc_type = document_types[i] if document_types and i < len(document_types) else None
                
                logger.info(f"Processing document {i+1}/{len(file_paths)}: {file_path}")
                
                result = self.extract_from_document(
                    file_path=file_path,
                    doc_type=doc_type,
                    custom_requirements=global_requirements
                )
                
                result.processing_metadata["batch_index"] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process document {file_path}: {e}")
                failed_result = ExtractionResult(
                    success=False,
                    errors=[str(e)],
                    processing_metadata={"batch_index": i, "file_path": file_path}
                )
                results.append(failed_result)
        
        # Compile batch summary
        successful_extractions = [r for r in results if r.success]
        failed_extractions = [r for r in results if not r.success]
        
        batch_summary = {
            "total_documents": len(file_paths),
            "successful_extractions": len(successful_extractions),
            "failed_extractions": len(failed_extractions),
            "success_rate": len(successful_extractions) / len(file_paths) * 100 if file_paths else 0,
            "processing_time_seconds": (datetime.now() - batch_start_time).total_seconds(),
            "average_confidence_score": None
        }
        
        # Calculate average confidence score
        confidence_scores = [r.confidence_score for r in successful_extractions if r.confidence_score]
        if confidence_scores:
            batch_summary["average_confidence_score"] = sum(confidence_scores) / len(confidence_scores)
        
        return {
            "batch_summary": batch_summary,
            "individual_results": results,
            "successful_results": successful_extractions,
            "failed_results": failed_extractions
        }