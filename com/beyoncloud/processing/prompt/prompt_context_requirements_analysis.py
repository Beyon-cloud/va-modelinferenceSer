import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ContextRequirementsAnalysis:

    def __init__(self):
        # Intentionally empty for now.
        # Reason: This class does not require instance state at construction
        # and will initialize attributes lazily when the analysis runs.
        # If future attributes are needed, initialize them here.
        pass

    # Helper to apply a regex and append a requirement if matched
    def _maybe_add(self, conditions_text: str, pattern: str, requirement_key: str, requirements: List[str]):
        if re.search(pattern, conditions_text):
            requirements.append(requirement_key)

    def _analyze_context_requirements_with_model(self, query: str, entities: List[Dict]) -> List[str]:
        """Model-based context requirements analysis"""
        requirements: List[str] = []
        q = (query or "").lower()

        # Define pattern-driven checks
        pattern_definitions: List[Dict[str, Any]] = [
            {"pattern": r'\b(previous|before|earlier|remember|context)\b', "req": "previous_knowledge"},
            {"pattern": r'\b(verify|check|validate|reference|database)\b', "req": "external_verification"},
            {"pattern": r'\b(expert|professional|specialized|advanced)\b', "req": "domain_expertise"},
            {"pattern": r'\b(example|sample|demonstrate|show)\b', "req": "examples_needed"},
            {"pattern": r'\b(explain|why|how|reason)\b', "req": "explanation_needed"},
            {"pattern": r'\b(format|json|csv|structure|table)\b', "req": "specific_formatting"},
        ]

        # Apply the pattern definitions
        for d in pattern_definitions:
            self._maybe_add(q, d["pattern"], d["req"], requirements)

        # Entity-based requirements
        entity_types = [e.get('label') for e in entities]
        if 'PERSON' in entity_types or 'ORGANIZATION' in entity_types:
            requirements.append('entity_verification')

        if len(entity_types) > 3:
            requirements.append('complex_entity_handling')

        return requirements