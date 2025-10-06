import logging
from com.beyoncloud.schemas.prompt_gen_reqres_datamodel import SchemaPromptRequest
from com.beyoncloud.processing.prompt.prompt_generation.enhance_document_processor import EnhancedDocumentProcessor

logger = logging.getLogger(__name__)

class SchemaPrompt:

    def __init__(self):
        # Intentionally empty for now.
        # Reason: This class does not require instance state at construction
        # and will initialize attributes lazily when the analysis runs.
        # If future attributes are needed, initialize them here.
        pass


    async def generate_schema_prompt(self, schema_prompt_request: SchemaPromptRequest):

        file_path = schema_prompt_request.source_path
        document_type = schema_prompt_request.document_type
        enhanced_document_processor = EnhancedDocumentProcessor()
        results = await enhanced_document_processor.extract_from_document(file_path,document_type, None, None)
        print(f"results ==> {results}")
        return results
