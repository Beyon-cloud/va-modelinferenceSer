import torch
import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForTokenClassification, pipeline
import com.beyoncloud.config.settings.env_config as config
from langchain import HuggingFacePipeline

logger = logging.getLogger(__name__)

class ModelRegistry:

    hf_llama_model_pipeline = None
    llama_tokenizer = None

    dslim_ner_pipeline = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def initialize_all_model(cls):
        cls._load_llama_model()

    @classmethod
    def _load_llama_model(cls):
        model_path = config.LLM_MODEL_PATH
        logger.info(f"Initializing model from: {model_path}")
        logger.info(f"Using device: {'GPU' if cls.device == "cuda" else 'CPU'}")
        model_path = os.path.abspath(model_path)
        # Load tokenizer and model
        cls.llama_tokenizer = AutoTokenizer.from_pretrained(model_path)
        cls.llama_tokenizer.pad_token = cls.llama_tokenizer.eos_token

        cls.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if cls.device == "cuda" else torch.float32,
            device_map="auto" if cls.device == "cuda" else None,
            low_cpu_mem_usage=True
        )

        # Build pipeline
        cls.pipeline_obj = pipeline(
            "text-generation",
            model=cls.model,
            tokenizer=cls.llama_tokenizer,
            max_new_tokens=1024,
            temperature=0.8,
            do_sample=False,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,
            pad_token_id=cls.llama_tokenizer.eos_token_id
        )

        # Wrap with LangChain
        cls.hf_llama_model_pipeline = HuggingFacePipeline(pipeline=cls.pipeline_obj)
