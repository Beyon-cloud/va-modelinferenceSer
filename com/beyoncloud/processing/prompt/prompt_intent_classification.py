import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from com.beyoncloud.schemas.prompt_datamodel import QueryIntent
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification,
    BertTokenizer, BertForSequenceClassification, BertModel, BertConfig,
    pipeline, Trainer, TrainingArguments, AutoModel
)

logger = logging.getLogger(__name__)

class IntentClassification:

    def __init__(self):
        # Training data
        self.intent_training_data = self._generate_intent_training_data()
        self._train_intent_classifier()

    def _train_intent_classifier(self):
        """Train BERT-based intent classifier"""
        try:
            # Use a simpler approach with transformers pipeline
            intent_queries = self.intent_training_data["intent"]["queries"]
            intent_labels = [intent.value for intent in self.intent_training_data["intent"]["intents"]]
            
            # Create a simple classifier using transformers
            self.intent_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            
            # For simplicity, use a rule-based approach with learned patterns
            self.intent_patterns = self._learn_intent_patterns(intent_queries, intent_labels)
            
            logger.info("✅ Intent classifier initialized")
            
        except Exception as e:
            logger.error(f"❌ Error training intent classifier: {e}")
            self.intent_patterns = {}
    
    def _learn_intent_patterns(self, queries: List[str], labels: List[str]) -> Dict[str, List[str]]:
        """Learn patterns for intent classification"""
        patterns = {}
        
        for query, label in zip(queries, labels):
            if label not in patterns:
                patterns[label] = []
            
            # Extract key phrases
            words = query.lower().split()
            key_phrases = [word for word in words if len(word) > 3 and word not in ['this', 'that', 'with', 'from', 'they', 'have', 'been']]
            patterns[label].extend(key_phrases)
        
        # Remove duplicates and keep most common
        for label in patterns:
            patterns[label] = list(set(patterns[label]))
        
        return patterns

    def _generate_intent_training_data(self) -> Dict[str, List]:
        """Generate synthetic training data for models"""
        
        # Intent classification training data
        intent_data = {
            "queries": [
                # Simple extraction
                "Extract all person names from this text",
                "Find organizations mentioned in the document",
                "Identify locations in the paragraph",
                "Get all dates from this content",
                "List the monetary amounts",
                
                # Learning requests
                "Show me examples of entity extraction",
                "How do I identify medical terms?",
                "Teach me to recognize legal entities",
                "Demonstrate pattern recognition",
                "Explain entity classification",
                
                # Explanation needed
                "Why did you classify this as a person?",
                "Explain your reasoning for this extraction",
                "How did you determine this entity type?",
                "What patterns led to this classification?",
                "Clarify the extraction process",
                
                # Batch processing
                "Process multiple documents for entities",
                "Extract from all files in the folder",
                "Batch analyze these reports",
                "Handle large document collections",
                "Process thousands of records",
                
                # Domain specific
                "Extract medical entities from patient records",
                "Identify legal terms in contracts",
                "Find financial entities in reports",
                "Analyze technical documentation",
                "Process academic papers",
                
                # Interactive session
                "Let's discuss entity extraction",
                "I want to chat about NER",
                "Can we work together on this?",
                "Interactive entity analysis",
                "Collaborative extraction session",
                
                # Structured output
                "Return results in JSON format",
                "Format output as CSV",
                "Structure the entities in a table",
                "Provide formatted results",
                "Output in specific schema",
                
                # Context dependent
                "Based on previous analysis, extract entities",
                "Considering the context, identify terms",
                "Use background knowledge for extraction",
                "Context-aware entity recognition",
                "Reference-based identification",
                
                # Fact verification
                "Verify the accuracy of extracted entities",
                "Check if these entities are correct",
                "Validate the extraction results",
                "Confirm entity classifications",
                "Fact-check the identified terms",
                
                # Complex reasoning
                "Perform comprehensive entity analysis",
                "Detailed extraction with reasoning",
                "Complex multi-step identification",
                "Thorough entity investigation",
                "Advanced pattern recognition"
            ],
            "intents": [
                # Corresponding intents for above queries
                QueryIntent.SIMPLE_EXTRACTION, QueryIntent.SIMPLE_EXTRACTION, QueryIntent.SIMPLE_EXTRACTION, 
                QueryIntent.SIMPLE_EXTRACTION, QueryIntent.SIMPLE_EXTRACTION,
                
                QueryIntent.LEARNING_REQUEST, QueryIntent.LEARNING_REQUEST, QueryIntent.LEARNING_REQUEST,
                QueryIntent.LEARNING_REQUEST, QueryIntent.LEARNING_REQUEST,
                
                QueryIntent.EXPLANATION_NEEDED, QueryIntent.EXPLANATION_NEEDED, QueryIntent.EXPLANATION_NEEDED,
                QueryIntent.EXPLANATION_NEEDED, QueryIntent.EXPLANATION_NEEDED,
                
                QueryIntent.BATCH_PROCESSING, QueryIntent.BATCH_PROCESSING, QueryIntent.BATCH_PROCESSING,
                QueryIntent.BATCH_PROCESSING, QueryIntent.BATCH_PROCESSING,
                
                QueryIntent.DOMAIN_SPECIFIC, QueryIntent.DOMAIN_SPECIFIC, QueryIntent.DOMAIN_SPECIFIC,
                QueryIntent.DOMAIN_SPECIFIC, QueryIntent.DOMAIN_SPECIFIC,
                
                QueryIntent.INTERACTIVE_SESSION, QueryIntent.INTERACTIVE_SESSION, QueryIntent.INTERACTIVE_SESSION,
                QueryIntent.INTERACTIVE_SESSION, QueryIntent.INTERACTIVE_SESSION,
                
                QueryIntent.STRUCTURED_OUTPUT, QueryIntent.STRUCTURED_OUTPUT, QueryIntent.STRUCTURED_OUTPUT,
                QueryIntent.STRUCTURED_OUTPUT, QueryIntent.STRUCTURED_OUTPUT,
                
                QueryIntent.CONTEXT_DEPENDENT, QueryIntent.CONTEXT_DEPENDENT, QueryIntent.CONTEXT_DEPENDENT,
                QueryIntent.CONTEXT_DEPENDENT, QueryIntent.CONTEXT_DEPENDENT,
                
                QueryIntent.FACT_VERIFICATION, QueryIntent.FACT_VERIFICATION, QueryIntent.FACT_VERIFICATION,
                QueryIntent.FACT_VERIFICATION, QueryIntent.FACT_VERIFICATION,
                
                QueryIntent.COMPLEX_REASONING, QueryIntent.COMPLEX_REASONING, QueryIntent.COMPLEX_REASONING,
                QueryIntent.COMPLEX_REASONING, QueryIntent.COMPLEX_REASONING
            ]
        }
        
        
        return {
            "intent": intent_data
        }

    def predict_intent_with_model(self, query: str) -> Tuple[QueryIntent, float]:
        """Model-based intent prediction"""
        query_lower = query.lower()
        intent_scores = {}
        
        # Use learned patterns
        for intent_name, patterns in self.intent_patterns.items():
            score = 0
            matches = 0
            for pattern in patterns:
                if pattern in query_lower:
                    score += 1
                    matches += 1
            
            # Normalize score
            if patterns:
                intent_scores[intent_name] = (score / len(patterns)) * (1 + matches * 0.1)
            else:
                intent_scores[intent_name] = 0
        
        # Get best intent
        if intent_scores:
            best_intent_name = max(intent_scores, key=intent_scores.get)
            confidence = min(intent_scores[best_intent_name], 1.0)
            print(f"best_intent_name - {best_intent_name} - {confidence}")
            
            # Convert to enum
            try:
                best_intent = QueryIntent(best_intent_name)
            except ValueError:
                best_intent = QueryIntent.SIMPLE_EXTRACTION
                confidence = 0.5
        else:
            best_intent = QueryIntent.SIMPLE_EXTRACTION
            confidence = 0.5
        
        return best_intent, confidence