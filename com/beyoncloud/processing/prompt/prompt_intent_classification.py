import logging
from typing import List, Dict, Any, Tuple
from com.beyoncloud.schemas.prompt_datamodel import QueryIntent
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class IntentClassification:
    """Intent classification using lightweight pattern learning."""

    def __init__(self):
        self.intent_training_data = self._generate_intent_training_data()
        self._train_intent_classifier()

    # ===========================
    # TRAINING
    # ===========================
    def _train_intent_classifier(self):
        """Initialize tokenizer and learn intent patterns."""
        try:
            intent_data = self.intent_training_data["intent"]
            self.intent_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.intent_patterns = self._learn_intent_patterns(
                intent_data["queries"], [i.value for i in intent_data["intents"]]
            )
            logger.info("✅ Intent classifier initialized")
        except Exception as e:
            logger.error(f"❌ Error training intent classifier: {e}")
            self.intent_patterns = {}

    @staticmethod
    def _learn_intent_patterns(queries: List[str], labels: List[str]) -> Dict[str, List[str]]:
        """Extract keyword patterns per intent."""
        stopwords = {"this", "that", "with", "from", "they", "have", "been"}
        patterns: Dict[str, List[str]] = {}

        for query, label in zip(queries, labels):
            words = [w for w in query.lower().split() if len(w) > 3 and w not in stopwords]
            patterns.setdefault(label, []).extend(words)

        # Deduplicate
        return {label: sorted(set(words)) for label, words in patterns.items()}

    # ===========================
    # DATA GENERATION
    # ===========================
    def _generate_intent_training_data(self) -> Dict[str, Any]:
        """Generate compact synthetic intent training data."""
        intent_examples = {
            QueryIntent.SIMPLE_EXTRACTION: [
                "Extract all person names from this text",
                "Find organizations mentioned in the document",
                "Identify locations in the paragraph",
                "Get all dates from this content",
                "List the monetary amounts",
            ],
            QueryIntent.LEARNING_REQUEST: [
                "Show me examples of entity extraction",
                "How do I identify medical terms?",
                "Teach me to recognize legal entities",
                "Demonstrate pattern recognition",
                "Explain entity classification",
            ],
            QueryIntent.EXPLANATION_NEEDED: [
                "Why did you classify this as a person?",
                "Explain your reasoning for this extraction",
                "How did you determine this entity type?",
                "What patterns led to this classification?",
                "Clarify the extraction process",
            ],
            QueryIntent.BATCH_PROCESSING: [
                "Process multiple documents for entities",
                "Extract from all files in the folder",
                "Batch analyze these reports",
                "Handle large document collections",
                "Process thousands of records",
            ],
            QueryIntent.DOMAIN_SPECIFIC: [
                "Extract medical entities from patient records",
                "Identify legal terms in contracts",
                "Find financial entities in reports",
                "Analyze technical documentation",
                "Process academic papers",
            ],
            QueryIntent.INTERACTIVE_SESSION: [
                "Let's discuss entity extraction",
                "I want to chat about NER",
                "Can we work together on this?",
                "Interactive entity analysis",
                "Collaborative extraction session",
            ],
            QueryIntent.STRUCTURED_OUTPUT: [
                "Return results in JSON format",
                "Format output as CSV",
                "Structure the entities in a table",
                "Provide formatted results",
                "Output in specific schema",
            ],
            QueryIntent.CONTEXT_DEPENDENT: [
                "Based on previous analysis, extract entities",
                "Considering the context, identify terms",
                "Use background knowledge for extraction",
                "Context-aware entity recognition",
                "Reference-based identification",
            ],
            QueryIntent.FACT_VERIFICATION: [
                "Verify the accuracy of extracted entities",
                "Check if these entities are correct",
                "Validate the extraction results",
                "Confirm entity classifications",
                "Fact-check the identified terms",
            ],
            QueryIntent.COMPLEX_REASONING: [
                "Perform comprehensive entity analysis",
                "Detailed extraction with reasoning",
                "Complex multi-step identification",
                "Thorough entity investigation",
                "Advanced pattern recognition",
            ],
        }

        # Flatten
        queries, intents = [], []
        for intent, qlist in intent_examples.items():
            queries.extend(qlist)
            intents.extend([intent] * len(qlist))

        return {"intent": {"queries": queries, "intents": intents}}

    # ===========================
    # PREDICTION
    # ===========================
    def predict_intent_with_model(self, query: str) -> Tuple[QueryIntent, float]:
        """Predict intent and confidence score based on learned patterns."""
        query_lower = query.lower()
        intent_scores = {}

        for intent, patterns in self.intent_patterns.items():
            matches = sum(pattern in query_lower for pattern in patterns)
            score = (matches / len(patterns)) * (1 + matches * 0.1) if patterns else 0
            intent_scores[intent] = score

        if not intent_scores:
            return QueryIntent.SIMPLE_EXTRACTION, 0.5

        best_intent_name = max(intent_scores, key=intent_scores.get)
        confidence = min(intent_scores[best_intent_name], 1.0)

        try:
            best_intent = QueryIntent(best_intent_name)
        except ValueError:
            best_intent = QueryIntent.SIMPLE_EXTRACTION
            confidence = 0.5

        return best_intent, confidence
