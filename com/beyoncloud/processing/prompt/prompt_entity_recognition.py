import re
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    pipeline
)
import logging
from typing import List, Dict, Any
from com.beyoncloud.models.model_service import ModelServiceLoader
from com.beyoncloud.models import model_singleton

logger = logging.getLogger(__name__)

class AdvancedEntityRecognizer:
    """Advanced entity recognition with pattern learning"""
    
    def __init__(self):
        self.pattern_model = None
        self.entity_patterns = {}
        all_model_objects = model_singleton.modelServiceLoader or ModelServiceLoader()
        self.dslim_ner_pipeline = all_model_objects.get_dslim_ner_pipeline()
    
    def _load_models(self):
        """Load NER and pattern recognition models"""
        try:
            self.dslim_ner_pipeline = pipeline(
                "ner",
                model=AutoModelForTokenClassification.from_pretrained(self.model_name),
                tokenizer=AutoTokenizer.from_pretrained(self.model_name),
                aggregation_strategy="simple"
            )
            logger.info(f"✅ NER model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Error loading NER model: {e}")
    
    def extract_entities_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities and learn patterns"""
        entities = []
        
        if self.dslim_ner_pipeline:
            # BERT-based extraction
            results = []
            for output in self.dslim_ner_pipeline(text):
                # Convert tensors/lists without NumPy
                if hasattr(output, "logits"):
                    output["logits"] = output["logits"].detach().cpu().tolist()
                results.append(output)
            
            for entity in results:
                if entity['score'] > 0.7:
                    entity_info = {
                        'text': entity['word'],
                        'label': entity['entity_group'],
                        'start': entity['start'],
                        'end': entity['end'],
                        'confidence': entity['score'],
                        'method': 'bert_ner',
                        'context': self._get_context(text, entity['start'], entity['end']),
                        'patterns': self._identify_patterns(text, entity)
                    }
                    entities.append(entity_info)
        
        # Add rule-based patterns
        rule_based_entities = self._extract_rule_based_entities(text)
        entities.extend(rule_based_entities)
        
        # Learn new patterns
        self._learn_patterns(entities)
        
        return entities
    
    def _get_context(self, text: str, start: int, end: int, window: int = 20) -> Dict[str, str]:
        """Get context around entity"""
        left_context = text[max(0, start-window):start]
        right_context = text[end:min(len(text), end+window)]
        return {
            'left': left_context.strip(),
            'right': right_context.strip()
        }
    
    def _identify_patterns(self, text: str, entity: Dict) -> List[Dict[str, Any]]:
        """Identify patterns around entities"""
        patterns = []
        entity_text = entity['word']
        
        # Pattern: Title + Name
        if re.search(r'\b(?:Dr|Mr|Mrs|Ms|Prof|President|CEO|Director)\.?\s+' + re.escape(entity_text), text, re.IGNORECASE):
            patterns.append({'type': 'title_prefix', 'confidence': 0.9})
        
        # Pattern: Organization suffix
        if re.search(re.escape(entity_text) + r'\s+(?:Inc|Corp|Ltd|LLC|University|Hospital)', text, re.IGNORECASE):
            patterns.append({'type': 'org_suffix', 'confidence': 0.85})
        
        # Pattern: Location preposition
        if re.search(r'\b(?:in|at|from|to)\s+' + re.escape(entity_text), text, re.IGNORECASE):
            patterns.append({'type': 'location_prep', 'confidence': 0.8})
        
        # Pattern: Date format
        if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}', entity_text):
            patterns.append({'type': 'date_format', 'confidence': 0.95})
        
        # Pattern: Money format
        money_symbol_pattern = r'[$€£¥]\s?\d+(?:\.\d{1,2})?'
        money_word_pattern = r'\d+(?:\.\d{1,2})?\s*(?:dollars?|euros?|pounds?)'

        if re.match(f'(?:{money_symbol_pattern}|{money_word_pattern})', entity_text, re.IGNORECASE):
            patterns.append({'type': 'money_format', 'confidence': 0.9})

        
        return patterns
    
    def _extract_rule_based_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using rule-based patterns"""
        entities = []
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'EMAIL',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.95,
                'method': 'rule_based',
                'context': self._get_context(text, match.start(), match.end()),
                'patterns': [{'type': 'email_format', 'confidence': 0.95}]
            })
        
        # Phone pattern
        phone_pattern = r'\b(?:\+\d{1,3}\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        for match in re.finditer(phone_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'PHONE',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9,
                'method': 'rule_based',
                'context': self._get_context(text, match.start(), match.end()),
                'patterns': [{'type': 'phone_format', 'confidence': 0.9}]
            })
        
        # URL pattern
        url_pattern = r'https?://[\w.-]+(?::\d+)?(?:/[^\s?#]*)*(?:\?[^\s#]*)?(?:#\S*)?'
        for match in re.finditer(url_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'URL',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.95,
                'method': 'rule_based',
                'context': self._get_context(text, match.start(), match.end()),
                'patterns': [{'type': 'url_format', 'confidence': 0.95}]
            })
        
        return entities
    
    def _learn_patterns(self, entities: List[Dict[str, Any]]):
        """Learn new patterns from extracted entities"""
        for entity in entities:
            entity_type = entity['label']
            if entity_type not in self.entity_patterns:
                self.entity_patterns[entity_type] = []
            
            # Store successful patterns
            for pattern in entity.get('patterns', []):
                if pattern['confidence'] > 0.8:
                    pattern_info = {
                        'pattern': pattern,
                        'context': entity.get('context', {}),
                        'frequency': 1,
                        'accuracy': pattern['confidence']
                    }
                    self.entity_patterns[entity_type].append(pattern_info)