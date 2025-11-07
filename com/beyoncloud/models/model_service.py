import logging
from com.beyoncloud.models.model_loader import ModelRegistry

logger = logging.getLogger(__name__)

class ModelServiceLoader:
    def __init__(self):
        logger.info("Initilize ModelServiceLoader...")
        ModelRegistry.initialize_all_model()

        self.dslim_ner_pipeline = ModelRegistry.dslim_ner_pipeline

        self.hf_llama_model_pipeline = ModelRegistry.hf_llama_model_pipeline
        self.device = ModelRegistry.device
        self.llama_tokenizer = ModelRegistry.llama_tokenizer

        self.model = ModelRegistry._model
        self.tokenizer = ModelRegistry._tokenizer

    def get_dslim_ner_pipeline(self):
        return self.dslim_ner_pipeline

    def get_hf_llama_model_pipeline(self):
        return self.hf_llama_model_pipeline

    def get_llama_tokenizer(self):
        return self.llama_tokenizer

    def get_device(self):
        return self.device

    def get_stream_model_and_tokenizer(self):
        return self.model, self.tokenizer
