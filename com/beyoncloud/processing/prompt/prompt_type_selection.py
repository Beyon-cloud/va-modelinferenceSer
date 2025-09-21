import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from com.beyoncloud.schemas.prompt_datamodel import QueryIntent, QueryComplexity, PromptType

logger = logging.getLogger(__name__)

class PromptTypeSelection:

    def __init__(self):
        # Prompt type mapping
        self.prompt_type_mapping = self._initialize_prompt_mapping()

    def _initialize_prompt_mapping(self) -> Dict[str, List[PromptType]]:
        """Initialize mapping from predictions to prompt types"""
        return {
            # Intent-based mapping
            "simple_extraction": [PromptType.INSTRUCTION, PromptType.ZERO_SHOT],
            "learning_request": [PromptType.FEW_SHOT, PromptType.CHAIN_OF_THOUGHT, PromptType.ROLE_BASED],
            "explanation_needed": [PromptType.CHAIN_OF_THOUGHT, PromptType.CLARIFICATION, PromptType.INTERACTIVE_AGENT],
            "batch_processing": [PromptType.LONG_CONTEXT, PromptType.TASK_DECOMPOSITION, PromptType.MEMORY_AUGMENTED],
            "domain_specific": [PromptType.ROLE_BASED, PromptType.DOCUMENT_CONTEXT, PromptType.RETRIEVAL_AUGMENTED],
            "interactive_session": [PromptType.USER_ASSISTANT_CHAT, PromptType.INTERACTIVE_AGENT, PromptType.MULTI_TURN_DIALOGUE],
            "structured_output": [PromptType.STRUCTURED_OUTPUT, PromptType.FUNCTION_CALLING, PromptType.TABLE_KNOWLEDGE_BASE],
            "context_dependent": [PromptType.CONTEXT_AWARE, PromptType.MEMORY_AUGMENTED, PromptType.RETRIEVAL_AUGMENTED],
            "fact_verification": [PromptType.FACT_CHECKING, PromptType.SOURCE_ATTRIBUTION, PromptType.RETRIEVAL_AUGMENTED],
            "complex_reasoning": [PromptType.CHAIN_OF_THOUGHT, PromptType.TASK_DECOMPOSITION, PromptType.RE_RANKING],
            
            # Complexity-based mapping
            "simple": [PromptType.INSTRUCTION, PromptType.ZERO_SHOT],
            "moderate": [PromptType.FEW_SHOT, PromptType.ROLE_BASED, PromptType.CHAIN_OF_THOUGHT],
            "complex": [PromptType.CHAIN_OF_THOUGHT, PromptType.TASK_DECOMPOSITION, PromptType.RETRIEVAL_AUGMENTED],
            "expert": [PromptType.ROLE_BASED, PromptType.RETRIEVAL_AUGMENTED, PromptType.FACT_CHECKING],
            
            # Domain-based mapping
            "medical": [PromptType.ROLE_BASED, PromptType.DOCUMENT_CONTEXT, PromptType.FACT_CHECKING],
            "legal": [PromptType.ROLE_BASED, PromptType.SOURCE_ATTRIBUTION, PromptType.STRUCTURED_OUTPUT],
            "financial": [PromptType.ROLE_BASED, PromptType.FACT_CHECKING, PromptType.TABLE_KNOWLEDGE_BASE],
            "technical": [PromptType.ROLE_BASED, PromptType.FUNCTION_CALLING, PromptType.TOOL_USE],
            "academic": [PromptType.ROLE_BASED, PromptType.SOURCE_ATTRIBUTION, PromptType.RETRIEVAL_AUGMENTED],
            
            # Entity-based mapping
            "PERSON": [PromptType.STRUCTURED_OUTPUT, PromptType.CONTEXT_AWARE],
            "ORGANIZATION": [PromptType.FACT_CHECKING, PromptType.SOURCE_ATTRIBUTION],
            "LOCATION": [PromptType.CONTEXT_AWARE, PromptType.RETRIEVAL_AUGMENTED],
            "DATE": [PromptType.STRUCTURED_OUTPUT, PromptType.TEMPORAL_REASONING],
            "MONEY": [PromptType.STRUCTURED_OUTPUT, PromptType.FACT_CHECKING]
        }

    def _select_prompt_types_with_ensemble(
        self, 
        intent: QueryIntent, 
        complexity: QueryComplexity, 
        domain: str, 
        entities: List[Dict], 
        patterns: List[Dict]
    ) -> List[PromptType]:
        """Ensemble-based prompt type selection"""
        
        prompt_scores = {}
        
        # Initialize all prompt types with base score
        for prompt_type in PromptType:
            prompt_scores[prompt_type] = 0.0
        
        # Intent-based scoring
        intent_prompts = self.prompt_type_mapping.get(intent.value, [])
        for prompt in intent_prompts:
            if hasattr(PromptType, prompt.name):
                prompt_type = getattr(PromptType, prompt.name)
                prompt_scores[prompt_type] += 3.0
        
        # Complexity-based scoring
        complexity_name = complexity.name.lower()
        complexity_prompts = self.prompt_type_mapping.get(complexity_name, [])
        for prompt in complexity_prompts:
            if hasattr(PromptType, prompt.name):
                prompt_type = getattr(PromptType, prompt.name)
                prompt_scores[prompt_type] += 2.0
        
        # Domain-based scoring
        domain_prompts = self.prompt_type_mapping.get(domain, [])
        for prompt in domain_prompts:
            if hasattr(PromptType, prompt.name):
                prompt_type = getattr(PromptType, prompt.name)
                prompt_scores[prompt_type] += 2.5
        
        # Entity-based scoring
        entity_types = list(set([e['label'] for e in entities]))
        for entity_type in entity_types:
            entity_prompts = self.prompt_type_mapping.get(entity_type, [])
            for prompt in entity_prompts:
                if hasattr(PromptType, prompt.name.replace(' ', '_')):
                    try:
                        prompt_type = getattr(PromptType, prompt.name.replace(' ', '_'))
                        prompt_scores[prompt_type] += 1.5
                    except AttributeError:
                        continue
        
        # Pattern-based scoring
        for pattern in patterns:
            pattern_type = pattern['type']
            confidence = pattern['confidence']
            
            if pattern_type == 'question':
                prompt_scores[PromptType.QUESTION_ANSWERING] += confidence * 2
            elif pattern_type == 'instruction':
                prompt_scores[PromptType.INSTRUCTION] += confidence * 2
            elif pattern_type == 'learning':
                prompt_scores[PromptType.FEW_SHOT] += confidence * 2
                prompt_scores[PromptType.CHAIN_OF_THOUGHT] += confidence * 1.5
            elif pattern_type == 'format_request':
                prompt_scores[PromptType.STRUCTURED_OUTPUT] += confidence * 3
            elif pattern_type == 'batch_processing':
                prompt_scores[PromptType.LONG_CONTEXT] += confidence * 2
                prompt_scores[PromptType.TASK_DECOMPOSITION] += confidence * 1.5
            elif pattern_type == 'reasoning_request':
                prompt_scores[PromptType.CHAIN_OF_THOUGHT] += confidence * 3
                prompt_scores[PromptType.CLARIFICATION] += confidence * 2
            elif pattern_type == 'interaction_request':
                prompt_scores[PromptType.INTERACTIVE_AGENT] += confidence * 3
                prompt_scores[PromptType.USER_ASSISTANT_CHAT] += confidence * 2
        
        # Apply complexity adjustments
        if complexity == QueryComplexity.SIMPLE:
            # Boost simple prompts
            prompt_scores[PromptType.INSTRUCTION] += 1.0
            prompt_scores[PromptType.ZERO_SHOT] += 1.0
            # Reduce complex prompts
            prompt_scores[PromptType.CHAIN_OF_THOUGHT] -= 0.5
            prompt_scores[PromptType.TASK_DECOMPOSITION] -= 1.0
        
        elif complexity == QueryComplexity.EXPERT:
            # Boost expert-level prompts
            prompt_scores[PromptType.ROLE_BASED] += 2.0
            prompt_scores[PromptType.RETRIEVAL_AUGMENTED] += 1.5
            prompt_scores[PromptType.FACT_CHECKING] += 1.5
        
        # Sort by score and return top 5
        sorted_prompts = sorted(prompt_scores.items(), key=lambda x: x[1], reverse=True)
        top_prompts = [prompt_type for prompt_type, score in sorted_prompts[:5] if score > 0]
        print(f"top_prompts --> {top_prompts}")
        return top_prompts if top_prompts else [PromptType.INSTRUCTION]