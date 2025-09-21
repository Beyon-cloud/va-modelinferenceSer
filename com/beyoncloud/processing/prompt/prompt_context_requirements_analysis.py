import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from com.beyoncloud.schemas.prompt_datamodel import QueryIntent

logger = logging.getLogger(__name__)

class ContextRequirementsAnalysis:

    def __init__(self):
        pass

    def _analyze_context_requirements_with_model(self, query: str, entities: List[Dict]) -> List[str]:
        """Model-based context requirements analysis"""
        requirements = []
        query_lower = query.lower()
        
        # Previous knowledge requirements
        if re.search(r'\b(previous|before|earlier|remember|context)\b', query_lower):
            requirements.append('previous_knowledge')
        
        # External data requirements
        if re.search(r'\b(verify|check|validate|reference|database)\b', query_lower):
            requirements.append('external_verification')
        
        # Domain expertise requirements
        if re.search(r'\b(expert|professional|specialized|advanced)\b', query_lower):
            requirements.append('domain_expertise')
        
        # Example requirements
        if re.search(r'\b(example|sample|demonstrate|show)\b', query_lower):
            requirements.append('examples_needed')
        
        # Explanation requirements
        if re.search(r'\b(explain|why|how|reason)\b', query_lower):
            requirements.append('explanation_needed')
        
        # Formatting requirements
        if re.search(r'\b(format|json|csv|structure|table)\b', query_lower):
            requirements.append('specific_formatting')
        
        # Entity-based requirements
        entity_types = [e['label'] for e in entities]
        if 'PERSON' in entity_types or 'ORGANIZATION' in entity_types:
            requirements.append('entity_verification')
        
        if len(entity_types) > 3:
            requirements.append('complex_entity_handling')
        
        return requirements