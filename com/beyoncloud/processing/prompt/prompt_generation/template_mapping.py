import logging
from typing import Dict, List, Optional
from com.beyoncloud.schemas.prompt_gen_reqres_datamodel import (
    DocumentType,FieldType, SchemaField,DocumentSchema,ExtractionResult,ProcessingConfig
)

logger = logging.getLogger(__name__)

# Schema Templates  ( Class 3 PropertySchemaTemplate)
class PropertySchemaTemplate:
    @staticmethod
    def get_schema() -> DocumentSchema:
        fields = [
            SchemaField("property_address", FieldType.ADDRESS, "Complete property address", True, 
                       examples=["123 Main St, City, State 12345"]),
            SchemaField("property_type", FieldType.STRING, "Type of property", True,
                       examples=["Single Family", "Condo", "Townhouse", "Commercial"]),
            SchemaField("square_footage", FieldType.NUMBER, "Property size in square feet", True),
            SchemaField("bedrooms", FieldType.NUMBER, "Number of bedrooms", False),
            SchemaField("bathrooms", FieldType.NUMBER, "Number of bathrooms", False),
            SchemaField("purchase_price", FieldType.CURRENCY, "Purchase or listed price", True),
            SchemaField("property_taxes", FieldType.CURRENCY, "Annual property taxes", False),
            SchemaField("year_built", FieldType.NUMBER, "Year property was built", False),
            SchemaField("lot_size", FieldType.NUMBER, "Lot size in square feet", False),
            SchemaField("hoa_fees", FieldType.CURRENCY, "Monthly HOA fees", False),
        ]
        
        return DocumentSchema(
            document_type=DocumentType.PROPERTY,
            schema_name="PropertyDocumentSchema",
            description="Schema for extracting information from property-related documents",
            fields=fields,
            processing_hints=[
                "Look for MLS numbers and listing details",
                "Extract both current and historical price information",
                "Parse address components carefully for accuracy"
            ]
        )
    
    @staticmethod
    def get_extraction_priorities() -> List[str]:
        return ["property_address", "purchase_price", "property_type", "square_footage"]
    
    @staticmethod
    def get_validation_checks() -> List[str]:
        return [
            "Verify address format is complete",
            "Ensure price values are realistic",
            "Check that square footage is reasonable"
        ]