import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from com.beyoncloud.schemas.prompt_datamodel import QueryIntent

logger = logging.getLogger(__name__)


class PatternRecognition:

    def __init__(self):
        # Intentionally empty for now.
        # Reason: This class does not require instance state at construction
        # and will initialize attributes lazily when the analysis runs.
        # If future attributes are needed, initialize them here.
        pass

    _PATTERN_DEFINITIONS = [
    {"type": "question", "pattern": r'\?', "indicator": "question_mark", "check": lambda t: t.strip().endswith('?')},
    {"type": "instruction", "pattern": r'\b(extract|find|identify|get|list|show)\b', "indicator": "imperative_verb"},
    {"type": "learning", "pattern": r'\b(how|teach|show me|example|learn)\b', "indicator": "learning_keywords"},
    {"type": "format_request", "pattern": r'\b(json|csv|table|format|structure)\b', "indicator": "format_keywords"},
    {"type": "batch_processing", "pattern": r'\b(multiple|all|many|batch|documents|files)\b', "indicator": "quantity_keywords"},
    {"type": "reasoning_request", "pattern": r'\b(why|explain|reason|because|how)\b', "indicator": "reasoning_keywords"},
    {"type": "complexity_indicator", "pattern": r'\b(comprehensive|detailed|thorough|advanced|complex)\b', "indicator": "complexity_adjectives"},
    {"type": "interaction_request", "pattern": r'\b(let\'s|can we|together|discuss|chat)\b', "indicator": "interaction_keywords"},
]

    def _recognize_query_patterns(self, query: str) -> List[Dict[str, Any]]:
        q = query.strip().lower()
        patterns = []

        # Special-case: ends with '?'
        if q.endswith('?'):
            patterns.append({'type': 'question', 'confidence': 0.95, 'indicator': 'question_mark'})

        for p in self._PATTERN_DEFINITIONS:
            if p.get("check"):
                if p["check"](q):
                    patterns.append({"type": p["type"], "confidence": 0.9, "indicator": p.get("indicator")})
            else:
                if re.search(p["pattern"], q):
                    patterns.append({"type": p["type"], "confidence": 0.9, "indicator": p.get("indicator")})

        # Domain indicators (kept as-is, but applied to q)
        domain_indicators = {
            'medical': r'\b(patient|doctor|medical|clinical|diagnosis|treatment)\b',
            'legal': r'\b(court|legal|law|contract|case|attorney)\b',
            'financial': r'\b(financial|money|investment|bank|stock|trading)\b',
            'technical': r'\b(software|code|algorithm|system|technical|API)\b'
        }

        for domain, pattern in domain_indicators.items():
            if re.search(pattern, q):
                patterns.append({
                    'type': f'domain_{domain}',
                    'confidence': 0.85,
                    'indicator': f'{domain}_terminology'
                })

        return patterns