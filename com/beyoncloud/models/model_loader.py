import torch
import logging
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from langchain import HuggingFacePipeline
from peft import PeftModel
import com.beyoncloud.config.settings.env_config as config

logger = logging.getLogger(__name__)

class ModelRegistry:
    hf_llama_model_pipeline = None
    llama_tokenizer = None
    dslim_ner_pipeline = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _model = None
    _tokenizer = None

    @classmethod
    def initialize_all_model(cls):
        cls._load_llm_model_stream()

    @classmethod
    def _load_llama_model(cls):
        model_path = os.path.abspath(config.LLM_MODEL_PATH)
        logger.info(f"Initializing model from: {model_path}")
        logger.info(f"Using device: {cls.device.upper()}")

        # Load tokenizer
        cls.llama_tokenizer = AutoTokenizer.from_pretrained(model_path)
        cls.llama_tokenizer.pad_token = cls.llama_tokenizer.eos_token

        # Load model with 4-bit quantization if available
        load_kwargs = {
            "dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True
        }

        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            load_kwargs["quantization_config"] = bnb_config
            logger.info("Loading model in 4-bit quantized mode for faster inference.")
        except ImportError:
            logger.warning("bitsandbytes not found. Loading in FP16 mode instead.")

        cls.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            **load_kwargs
        )
        cls.model.eval()
        torch.set_grad_enabled(False)

        # Build optimized text-generation pipeline
        cls.pipeline_obj = pipeline(
            "text-generation",
            model=cls.model,
            tokenizer=cls.llama_tokenizer,
            device_map="auto",
            max_new_tokens=config.MODEL_MAX_NEW_TOKENS,   # reduced for speed
            temperature=config.MODEL_TEMPERATURE,              # more stable and fast
            top_p=config.MODEL_TOP_P,
            do_sample=False,              # deterministic
            repetition_penalty=config.REPETITION_PENALTY,
            return_full_text=False,
            pad_token_id=cls.llama_tokenizer.eos_token_id
        )

        # Wrap with LangChain
        cls.hf_llama_model_pipeline = HuggingFacePipeline(pipeline=cls.pipeline_obj)
        logger.info("✅ Llama model initialized and ready for fast inference.")


    @classmethod
    def _load_llm_model_stream(cls):
        model_path = config.LLM_MODEL_PATH
        logger.info(f"Initializing streaming LLaMA model from: {model_path}")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        cls._tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        cls._tokenizer.pad_token = cls._tokenizer.eos_token

        cls._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            dtype=torch.float16
            low_cpu_mem_usage=True
        )

        try:
            cls._model = torch.compile(cls._model)
            print("line no 109")
            logger.info("✅ Model compiled successfully for optimized inference.")
        except Exception as e:
            logger.warning(f"⚠️ Model compile skipped: {e}")

        logger.info("✅ LLaMA model and tokenizer loaded successfully.")


    # -----------------------------------------------------------
    # STREAMING LOADER (Supports LoRA + Full Model)
    # -----------------------------------------------------------
    @classmethod
    def _load_llm_lora_ft_model_stream(cls):
        model_path = os.path.abspath(config.LLM_MODEL_PATH)
        base_model_path = os.path.abspath(config.BASE_MODEL_PATH)

        logger.info(f"Initializing LLaMA model from: {model_path}")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Always load tokenizer from base model (not LoRA directory)
        cls._tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, use_fast=True, local_files_only=True
        )
        cls._tokenizer.pad_token = cls._tokenizer.eos_token

        #LoRA checkpoint
        logger.info("Detected LoRA fine-tuned checkpoint. Loading base + merging LoRA.")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            local_files_only=True,
        )

        cls._model = PeftModel.from_pretrained(
            base_model,
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True,
        )

        # Optional: torch.compile
        try:
            cls._model = torch.compile(cls._model)
            logger.info("Model compiled successfully (torch.compile).")
        except Exception as e:
            logger.warning(f"Model compile skipped: {e}")

        logger.info("LLaMA model + tokenizer loaded successfully.")