

# Main Document Extraction System
class DocumentExtractionSystem:
    """High-level system for document extraction using Hugging Face Llama3"""

    def __init__(
        self, 
        hf_api_key: str,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct:novita",
        config_type: str = "balanced"
    ):
        # Initialize Hugging Face Llama3 client
        self.llama3_client = HuggingFaceLlama3Client(
            api_key=hf_api_key,
            model_name=model_name
        )
        
        # Test connection
        if not self.llama3_client.test_connection():
            raise ConnectionError("Failed to connect to Hugging Face Llama3 API")
        
        # Set configuration
        if config_type == "high_accuracy":
            self.config = ConfigurationManager.create_high_accuracy_config()
        elif config_type == "fast":
            self.config = ConfigurationManager.create_fast_processing_config()
        elif config_type == "development":
            self.config = ConfigurationManager.create_development_config()
        else:
            self.config = ProcessingConfig()
        
        # Initialize processor
        self.processor = EnhancedDocumentProcessor(
            llama3_client=self.llama3_client,
            config=self.config
        )
        
        logger.info(f"Document Extraction System initialized with {model_name}")
    
    def extract_single_document(
        self, 
        file_path: str, 
        document_type: Optional[str] = None,
        detection_hints: Optional[List[str]] = None,
        export_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract from single document with optional export"""
        
        logger.info(f"Starting single document extraction: {file_path}")
        
        result = self.processor.extract_from_document(
            file_path=file_path,
            doc_type=document_type,
            detection_hints=detection_hints
        )
        
        # Export if requested
        if export_format and result.success:
            base_name = Path(file_path).stem
            output_path = f"{base_name}_extraction.{export_format}"
            
            if export_format == "json":
                ResultExporter.to_json([result], output_path)
            elif export_format == "csv":
                ResultExporter.to_csv([result], output_path)
        
        return {
            "extraction_result": result,
            "summary": {
                "success": result.success,
                "confidence_score": result.confidence_score,
                "errors": result.errors,
                "warnings": result.warnings,
                "document_type": result.processing_metadata.get("document_type") if result.processing_metadata else None,
                "processing_time": result.processing_metadata.get("processing_time_seconds") if result.processing_metadata else None
            }
        }
    
    def extract_multiple_documents(
        self, 
        file_paths: List[str],
        document_types: Optional[List[str]] = None,
        export_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract from multiple documents"""
        
        logger.info(f"Starting batch extraction of {len(file_paths)} documents")
        
        # Run batch processing
        batch_result = self.processor.batch_extract_documents(
            file_paths=file_paths, 
            document_types=document_types
        )
        
        # Export results if requested
        if export_format and batch_result.get("successful_results"):
            output_path = f"batch_extraction_results.{export_format}"
            
            if export_format == "csv":
                ResultExporter.to_csv(batch_result["successful_results"], output_path)
            elif export_format == "json":
                ResultExporter.to_json(batch_result["individual_results"], output_path)
        
        return batch_result
    
    def create_sample_document(self, doc_type: str, file_path: str):
        """Create sample documents for testing"""
        sample_contents = {
            "property": """
PROPERTY LISTING INFORMATION

Property Address: 1234 Maple Street, Springfield, IL 62701
Property Type: Single Family Home
Bedrooms: 3
Bathrooms: 2.5
Square Footage: 2,150 sq ft
Lot Size: 0.25 acres (10,890 sq ft)
Year Built: 1995
Listing Price: $285,000
Property Taxes: $3,200 annually
HOA Fees: N/A
Parking: 2-car attached garage
Additional Features: Hardwood floors, updated kitchen, finished basement

Contact: Jane Smith, Springfield Realty
Phone: (217) 555-0123
MLS #: SR789456
            """,
            
            "insurance": """
AUTO INSURANCE POLICY DECLARATION

Policy Number: AI-2024-789456
Policy Type: Auto Insurance
Policyholder Name: John Robert Smith
Policyholder Address: 567 Oak Avenue, Springfield, IL 62702

Coverage Details:
- Liability Coverage: $500,000
- Comprehensive Coverage: $100,000
- Collision Coverage: $100,000
- Uninsured Motorist: $250,000

Premium Amount: $1,200 annually
Payment Frequency: Semi-Annual
Deductible: $500
Policy Start Date: 01/15/2024
Policy End Date: 01/15/2025

Insurance Company: Springfield Insurance Co.
Agent: Mike Johnson
Policy Status: Active
            """,
            
            "education": """
OFFICIAL TRANSCRIPT

Springfield University
123 College Drive, Springfield, IL 62704

Student Name: Sarah Michelle Johnson
Student ID: SU2021-4567
Program: Bachelor of Science in Computer Science
Degree Level: Bachelor's

Academic Record:
GPA: 3.85
Credits Earned: 124
Credits Required: 120

Graduation Date: May 15, 2024
Academic Standing: Magna Cum Laude

Dean: Dr. Robert Wilson
Registrar: Mary Adams

This is an official transcript issued on June 1, 2024.
            """
        }
        
        content = sample_contents.get(doc_type, "Sample document content")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Sample {doc_type} document created: {file_path}")