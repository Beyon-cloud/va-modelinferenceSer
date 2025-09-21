from pydantic import BaseModel, Field
from enum import Enum
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class PromptType(Enum):
    """All available prompt types"""
    INSTRUCTION = "instruction"
    FEW_SHOT = "few_shot"
    ZERO_SHOT = "zero_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ROLE_BASED = "role_based"
    SYSTEM_MESSAGE = "system_message"
    USER_ASSISTANT_CHAT = "user_assistant_chat"
    DOCUMENT_CONTEXT = "document_context"
    RETRIEVAL_AUGMENTED = "retrieval_augmented"
    FUNCTION_CALLING = "function_calling"
    STRUCTURED_OUTPUT = "structured_output"
    TASK_DECOMPOSITION = "task_decomposition"
    QUESTION_ANSWERING = "question_answering"
    CONTEXT_AWARE = "context_aware"
    SEARCH_THEN_ANSWER = "search_then_answer"
    TOOL_USE = "tool_use"
    CLARIFICATION = "clarification"
    INTERACTIVE_AGENT = "interactive_agent"
    SUMMARIZATION = "summarization"
    RE_RANKING = "re_ranking"
    MULTI_TURN_DIALOGUE = "multi_turn_dialogue"
    MEMORY_AUGMENTED = "memory_augmented"
    SOURCE_ATTRIBUTION = "source_attribution"
    FACT_CHECKING = "fact_checking"
    TABLE_KNOWLEDGE_BASE = "table_knowledge_base"
    LONG_CONTEXT = "long_context"
    PROMPT_TEMPLATING = "prompt_templating"
    TEMPORAL_REASONING = "temporal_reasoning"

class QueryIntent(Enum):
    """Query intent classifications"""
    SIMPLE_EXTRACTION = "simple_extraction"
    BATCH_PROCESSING = "batch_processing"
    LEARNING_REQUEST = "learning_request"
    EXPLANATION_NEEDED = "explanation_needed"
    COMPARISON_TASK = "comparison_task"
    INTERACTIVE_SESSION = "interactive_session"
    DOMAIN_SPECIFIC = "domain_specific"
    COMPLEX_REASONING = "complex_reasoning"
    CONTEXT_DEPENDENT = "context_dependent"
    STRUCTURED_OUTPUT = "structured_output"
    FACT_VERIFICATION = "fact_verification"
    MULTI_DOCUMENT = "multi_document"
    REAL_TIME_PROCESSING = "real_time_processing"
    MEMORY_BASED = "memory_based"
    CLARIFICATION_NEEDED = "clarification_needed"

class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = 0
    MODERATE = 1
    COMPLEX = 2
    EXPERT = 3

class ModelPrediction(BaseModel):
    """Unified prediction structure"""
    prediction: Union[str, int, List[str]]
    confidence: float
    probabilities: Dict[str, float]
    features_used: List[str]
    model_name: str
    timestamp: datetime

class QueryAnalysis(BaseModel):
    """Complete query analysis results"""
    intent: QueryIntent
    complexity: QueryComplexity
    entities_detected: List[Dict[str, Any]]
    domain: str
    patterns_found: List[Dict[str, Any]]
    context_requirements: List[str]
    suggested_prompt_types: List[PromptType]
    confidence_scores: Dict[str, float]
    feature_importance: Dict[str, float]
    model_predictions: Dict[str, ModelPrediction]