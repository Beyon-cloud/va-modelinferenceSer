import logging
from typing import List, Dict
from com.beyoncloud.schemas.prompt_datamodel import QueryIntent, QueryComplexity, PromptType

logger = logging.getLogger(__name__)


class PromptTypeSelection:

    def __init__(self):
        self.prompt_type_mapping = self._initialize_prompt_mapping()

    # -------------------------
    # Initialization
    # -------------------------
    def _initialize_prompt_mapping(self) -> Dict[str, List[PromptType]]:
        """Initialize mapping from predictions to prompt types."""
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
            "MONEY": [PromptType.STRUCTURED_OUTPUT, PromptType.FACT_CHECKING],
        }

    # -------------------------
    # Private helper methods
    # -------------------------
    def _apply_mapping_score(self, key: str, weight: float, prompt_scores: Dict):
        """Apply weight from mapping."""
        for prompt in self.prompt_type_mapping.get(key, []):
            if hasattr(PromptType, prompt.name):
                prompt_type = getattr(PromptType, prompt.name)
                prompt_scores[prompt_type] += weight

    def _apply_entity_scores(self, entities: List[Dict], prompt_scores: Dict):
        """Apply entity-based scores."""
        entity_labels = {e.get("label") for e in entities if "label" in e}
        for label in entity_labels:
            for prompt in self.prompt_type_mapping.get(label, []):
                name = prompt.name.replace(" ", "_")
                if hasattr(PromptType, name):
                    prompt_scores[getattr(PromptType, name)] += 1.5

    def _apply_pattern_scores(self, patterns: List[Dict], prompt_scores: Dict):
        """Apply pattern-based weights dynamically."""
        pattern_weights = {
            "question": [(PromptType.QUESTION_ANSWERING, 2)],
            "instruction": [(PromptType.INSTRUCTION, 2)],
            "learning": [(PromptType.FEW_SHOT, 2), (PromptType.CHAIN_OF_THOUGHT, 1.5)],
            "format_request": [(PromptType.STRUCTURED_OUTPUT, 3)],
            "batch_processing": [(PromptType.LONG_CONTEXT, 2), (PromptType.TASK_DECOMPOSITION, 1.5)],
            "reasoning_request": [(PromptType.CHAIN_OF_THOUGHT, 3), (PromptType.CLARIFICATION, 2)],
            "interaction_request": [(PromptType.INTERACTIVE_AGENT, 3), (PromptType.USER_ASSISTANT_CHAT, 2)],
        }

        for pattern in patterns:
            for target, weight in pattern_weights.get(pattern.get("type"), []):
                confidence = pattern.get("confidence", 1.0)
                prompt_scores[target] += confidence * weight

    def _apply_complexity_adjustments(self, complexity: QueryComplexity, prompt_scores: Dict):
        """Adjust prompt weights based on complexity."""
        if complexity == QueryComplexity.SIMPLE:
            prompt_scores[PromptType.INSTRUCTION] += 1.0
            prompt_scores[PromptType.ZERO_SHOT] += 1.0
            prompt_scores[PromptType.CHAIN_OF_THOUGHT] -= 0.5
            prompt_scores[PromptType.TASK_DECOMPOSITION] -= 1.0
        elif complexity == QueryComplexity.EXPERT:
            prompt_scores[PromptType.ROLE_BASED] += 2.0
            prompt_scores[PromptType.RETRIEVAL_AUGMENTED] += 1.5
            prompt_scores[PromptType.FACT_CHECKING] += 1.5

    # -------------------------
    # Main Ensemble Method
    # -------------------------
    def _select_prompt_types_with_ensemble(
        self,
        intent: QueryIntent,
        complexity: QueryComplexity,
        domain: str,
        entities: List[Dict],
        patterns: List[Dict]
    ) -> List[PromptType]:
        """Main ensemble-based prompt type selection with low cognitive complexity."""

        prompt_scores = dict.fromkeys(PromptType, 0.0)

        # Apply scoring factors
        self._apply_mapping_score(intent.value, 3.0, prompt_scores)
        self._apply_mapping_score(complexity.name.lower(), 2.0, prompt_scores)
        self._apply_mapping_score(domain, 2.5, prompt_scores)
        self._apply_entity_scores(entities, prompt_scores)
        self._apply_pattern_scores(patterns, prompt_scores)
        self._apply_complexity_adjustments(complexity, prompt_scores)

        # Select top 5
        top_prompts = [
            ptype for ptype, score in sorted(prompt_scores.items(), key=lambda x: x[1], reverse=True)[:5] if score > 0
        ]

        logger.debug(f"Top selected prompts: {top_prompts}")
        return top_prompts or [PromptType.INSTRUCTION]
