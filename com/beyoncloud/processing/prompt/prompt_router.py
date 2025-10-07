import numpy as np
import json
import re
import pickle
from typing import List, Dict, Any, Tuple
import warnings
import logging
from datetime import datetime
warnings.filterwarnings('ignore')
from com.beyoncloud.utils.file_util import JsonLoader
import com.beyoncloud.config.settings.env_config as config
from com.beyoncloud.processing.prompt.prompt_template import get_prompt_template, get_prompt_input_variables
from com.beyoncloud.schemas.prompt_datamodel import QueryIntent, QueryComplexity, PromptType, QueryAnalysis, ModelPrediction
from com.beyoncloud.processing.prompt.prompt_entity_recognition import AdvancedEntityRecognizer
from com.beyoncloud.processing.prompt.prompt_intent_classification import IntentClassification
from com.beyoncloud.processing.prompt.prompt_complexity_assessment import ComplexityAssessmentModel
from com.beyoncloud.processing.prompt.prompt_domain_classification import DomainClassificationModel
from com.beyoncloud.processing.prompt.prompt_pattern_recognition import PatternRecognition
from com.beyoncloud.processing.prompt.prompt_feature_importance_analysis import FeatureImportanceAnalysis
from com.beyoncloud.processing.prompt.prompt_context_requirements_analysis import ContextRequirementsAnalysis
from com.beyoncloud.processing.prompt.prompt_comprehensive_analysis import ComprehensiveAnalysis

logger = logging.getLogger(__name__)

class DynamicPromptRouter:
    """Complete model-based prompt routing system"""
    
    def __init__(self):
        logger.info("🚀 Initializing Model-Based Prompt Router")
        
        # Initialize all models
        self.entity_recognizer = AdvancedEntityRecognizer()
        self.complexity_model = ComplexityAssessmentModel()
        self.domain_model = DomainClassificationModel()
        self.intent_classification = IntentClassification()
        self.pattern_recognition = PatternRecognition()
        self.feature_importance_analysis = FeatureImportanceAnalysis()
        self.context_requirements_analysis = ContextRequirementsAnalysis()
        self.comprehensive_analysis = ComprehensiveAnalysis()
        
        # Intent classification model (neural)
        self.intent_model = None
        self.intent_tokenizer = None
        
        # Pattern recognition model
        self.pattern_model = None
        
        # Training data
        self.training_data = self._generate_training_data()
        
        # Train models
        self._train_all_models()
        
        logger.info("✅ Model-Based Prompt Router initialized successfully")
    

    
    
    def _generate_training_data(self) -> Dict[str, List]:
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
        
        # Complexity classification training data
        complexity_data = {
            "queries": [
                # Simple queries
                "Find names", "Get dates", "List locations", "Extract emails", "Show organizations",
                
                # Moderate queries
                "Extract all person names and their roles from the document",
                "Identify medical conditions and treatments mentioned",
                "Find companies and their locations in the text",
                "Analyze the document for financial entities",
                "Classify entities by their importance",
                
                # Complex queries
                "Perform comprehensive entity extraction with confidence scoring and context analysis",
                "Extract, classify, and verify all entities while maintaining relationships between them",
                "Analyze multiple document types for cross-referenced entity identification",
                "Implement sophisticated pattern matching for domain-specific entity recognition",
                "Conduct detailed entity analysis with temporal and spatial relationship mapping",
                
                # Expert queries
                "Execute advanced multi-modal entity extraction using state-of-the-art NLP techniques",
                "Implement hierarchical entity classification with uncertainty quantification",
                "Perform cross-lingual entity recognition with cultural context consideration",
                "Deploy ensemble-based entity extraction with active learning integration",
                "Conduct expert-level biomedical entity recognition with ontology alignment"
            ],
            "complexities": [
                QueryComplexity.SIMPLE, QueryComplexity.SIMPLE, QueryComplexity.SIMPLE, QueryComplexity.SIMPLE, QueryComplexity.SIMPLE,
                QueryComplexity.MODERATE, QueryComplexity.MODERATE, QueryComplexity.MODERATE, QueryComplexity.MODERATE, QueryComplexity.MODERATE,
                QueryComplexity.COMPLEX, QueryComplexity.COMPLEX, QueryComplexity.COMPLEX, QueryComplexity.COMPLEX, QueryComplexity.COMPLEX,
                QueryComplexity.EXPERT, QueryComplexity.EXPERT, QueryComplexity.EXPERT, QueryComplexity.EXPERT, QueryComplexity.EXPERT
            ]
        }
        
        # Domain classification training data
        domain_data = {
            "queries": [
                # Medical
                "Extract patient names and diagnoses", "Find drug names and dosages", "Identify medical procedures",
                "Analyze clinical trial data", "Process hospital records",
                
                # Legal
                "Extract case names and court decisions", "Identify legal precedents", "Find contract clauses",
                "Analyze legal documents", "Process court transcripts",
                
                # Financial
                "Extract stock prices and company names", "Find investment amounts", "Identify financial instruments",
                "Analyze market data", "Process banking records",
                
                # Technical
                "Extract API endpoints and functions", "Find software versions", "Identify programming languages",
                "Analyze code documentation", "Process technical specifications",
                
                # Academic
                "Extract author names and citations", "Find research methodologies", "Identify academic institutions",
                "Analyze research papers", "Process conference proceedings",
                
                # General
                "Extract basic information", "Find common entities", "Identify standard patterns",
                "Analyze general text", "Process everyday documents"
            ],
            "domains": [
                "medical", "medical", "medical", "medical", "medical",
                "legal", "legal", "legal", "legal", "legal",
                "financial", "financial", "financial", "financial", "financial",
                "technical", "technical", "technical", "technical", "technical",
                "academic", "academic", "academic", "academic", "academic",
                "general", "general", "general", "general", "general"
            ]
        }
        
        return {
            "intent": intent_data,
            "complexity": complexity_data,
            "domain": domain_data
        }
    
    def _train_all_models(self):
        """Train all ML models with generated data"""
        logger.info("🎯 Training all models...")
        
        try:
            # Train complexity model
            complexity_queries = self.training_data["complexity"]["queries"]
            complexity_labels = self.training_data["complexity"]["complexities"]
            self.complexity_model.train(complexity_queries, complexity_labels)
            
            # Train domain model
            domain_queries = self.training_data["domain"]["queries"]
            domain_labels = self.training_data["domain"]["domains"]
            self.domain_model.train(domain_queries, domain_labels)
            
            # Train intent classifier (simplified BERT-based)
            #self._train_intent_classifier()
            
            logger.info("✅ All models trained successfully")
            
        except Exception as e:
            logger.error(f"❌ Error training models: {e}")
    
    
    
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
    
    def _recognize_query_patterns(self, query: str) -> List[Dict[str, Any]]:
        """Advanced pattern recognition in queries"""
        patterns = []
        
        # Question patterns
        if query.strip().endswith('?'):
            patterns.append({'type': 'question', 'confidence': 0.95, 'indicator': 'question_mark'})
        
        # Instruction patterns
        if re.search(r'\b(extract|find|identify|get|list|show)\b', query.lower()):
            patterns.append({'type': 'instruction', 'confidence': 0.9, 'indicator': 'imperative_verb'})
        
        # Learning patterns
        if re.search(r'\b(how|teach|show me|example|learn)\b', query.lower()):
            patterns.append({'type': 'learning', 'confidence': 0.85, 'indicator': 'learning_keywords'})
        
        # Format request patterns
        if re.search(r'\b(json|csv|table|format|structure)\b', query.lower()):
            patterns.append({'type': 'format_request', 'confidence': 0.9, 'indicator': 'format_keywords'})
        
        # Batch processing patterns
        if re.search(r'\b(multiple|all|many|batch|documents|files)\b', query.lower()):
            patterns.append({'type': 'batch_processing', 'confidence': 0.8, 'indicator': 'quantity_keywords'})
        
        # Reasoning patterns
        if re.search(r'\b(why|explain|reason|because|how)\b', query.lower()):
            patterns.append({'type': 'reasoning_request', 'confidence': 0.85, 'indicator': 'reasoning_keywords'})
        
        # Complexity patterns
        if re.search(r'\b(comprehensive|detailed|thorough|advanced|complex)\b', query.lower()):
            patterns.append({'type': 'complexity_indicator', 'confidence': 0.8, 'indicator': 'complexity_adjectives'})
        
        # Interaction patterns
        if re.search(r'\b(let\'s|can we|together|discuss|chat)\b', query.lower()):
            patterns.append({'type': 'interaction_request', 'confidence': 0.8, 'indicator': 'interaction_keywords'})
        
        # Domain-specific patterns
        domain_indicators = {
            'medical': r'\b(patient|doctor|medical|clinical|diagnosis|treatment)\b',
            'legal': r'\b(court|legal|law|contract|case|attorney)\b',
            'financial': r'\b(financial|money|investment|bank|stock|trading)\b',
            'technical': r'\b(software|code|algorithm|system|technical|API)\b'
        }
        
        for domain, pattern in domain_indicators.items():
            if re.search(pattern, query.lower()):
                patterns.append({
                    'type': f'domain_{domain}', 
                    'confidence': 0.85, 
                    'indicator': f'{domain}_terminology'
                })
        
        return patterns
    
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
        entity_types = {e['label'] for e in entities}
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
        
        return top_prompts if top_prompts else [PromptType.INSTRUCTION]
    
    def _calculate_feature_importance(
        self, 
        model_predictions: Dict[str, ModelPrediction], 
        complexity_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate feature importance across all models"""
        
        feature_importance = {}
        
        # Add complexity features
        for feature, value in complexity_features.items():
            feature_importance[f'complexity_{feature}'] = value
        
        # Add model confidence as features
        for model_name, prediction in model_predictions.items():
            feature_importance[f'{model_name}_confidence'] = prediction.confidence
        
        # Add entity-based features
        if 'entities' in model_predictions:
            entity_pred = model_predictions['entities']
            if isinstance(entity_pred.prediction, list):
                feature_importance['entity_count'] = len(entity_pred.prediction)
                feature_importance['entity_diversity'] = len(set(entity_pred.prediction))
        
        # Normalize feature importance
        if feature_importance:
            max_importance = max(feature_importance.values())
            if max_importance > 0:
                for feature in feature_importance:
                    feature_importance[feature] = feature_importance[feature] / max_importance
        
        return feature_importance
    
    def _log_detailed_analysis(self, analysis: QueryAnalysis):
        """Log detailed analysis results"""
        logger.info("=" * 80)
        logger.info("📊 DETAILED MODEL-BASED ANALYSIS RESULTS")
        logger.info("=" * 80)
        
        logger.info(f"🎯 Intent: {analysis.intent.value} (confidence: {analysis.confidence_scores['intent']:.3f})")
        logger.info(f"🔢 Complexity: {analysis.complexity.name} (confidence: {analysis.confidence_scores['complexity']:.3f})")
        logger.info(f"🏷️  Domain: {analysis.domain} (confidence: {analysis.confidence_scores['domain']:.3f})")
        
        logger.info(f"🏗️  Entities Detected: {len(analysis.entities_detected)}")
        for entity in analysis.entities_detected[:3]:  # Show first 3
            logger.info(f"   • {entity['text']} ({entity['label']}) - {entity['confidence']:.3f} [{entity['method']}]")
        
        logger.info(f"🔍 Patterns Found: {len(analysis.patterns_found)}")
        for pattern in analysis.patterns_found[:3]:  # Show first 3
            logger.info(f"   • {pattern['type']} - {pattern['confidence']:.3f}")
        
        logger.info(f"📋 Context Requirements: {', '.join(analysis.context_requirements)}")
        
        logger.info("🎨 Suggested Prompt Types:")
        for i, prompt_type in enumerate(analysis.suggested_prompt_types[:3], 1):
            logger.info(f"   {i}. {prompt_type.value.upper()}")
        
        logger.info(f"🏆 Overall Confidence: {analysis.confidence_scores['overall']:.3f}")
        
        logger.info("🔬 Model Predictions Summary:")
        for model_name, prediction in analysis.model_predictions.items():
            logger.info(f"   • {model_name}: {prediction.confidence:.3f} confidence")
        
        logger.info("=" * 80)
    
    def generate_optimal_prompt(
        self, 
        query: str, 
        text: str, 
        domain_id: str = "",
        history_prompt: List=[]
    ) -> Dict[str, Any]:
        """Generate optimal prompt using complete model ensemble"""
        
        logger.info("🚀 STARTING COMPLETE MODEL-BASED PROMPT GENERATION")
        logger.info("=" * 80)
        
        # Perform comprehensive analysis
        analysis = self.comprehensive_analysis.analyze_query_with_models(query, text, domain_id)
        
        # Select the best prompt type
        best_prompt_type = analysis.suggested_prompt_types[0] if analysis.suggested_prompt_types else PromptType.INSTRUCTION
        
        print(f" best_prompt_type ---> {best_prompt_type} ")
        # Generate the actual prompt
        generated_prompt = self._generate_dynamic_prompt(query, domain_id, best_prompt_type, text, history_prompt, analysis)
        
        # Calculate routing confidence
        routing_confidence = analysis.confidence_scores['overall']
        
        # Generate comprehensive reasoning
        routing_reasoning = self._generate_comprehensive_reasoning(analysis)
        
        # Prepare result
        result = {
            'analysis': analysis,
            'selected_prompt_type': best_prompt_type,
            'generated_prompt': generated_prompt,
            'routing_confidence': routing_confidence,
            'routing_reasoning': routing_reasoning,
            'model_predictions': analysis.model_predictions,
            'feature_importance': analysis.feature_importance,
            'alternative_prompts': self._generate_alternative_prompts(query, domain_id, analysis.suggested_prompt_types[1:3], text, history_prompt, analysis)
        }
        
        logger.info(f"✅ OPTIMAL PROMPT GENERATED: {best_prompt_type.value.upper()}")
        logger.info(f"🎯 Routing Confidence: {routing_confidence:.3f}")
        
        return result
    
    def _generate_dynamic_prompt(
        self, 
        query: str,
        domain_id: str,
        prompt_type: PromptType, 
        text: str, 
        history_prompt: List,
        analysis: QueryAnalysis
    ) -> str:
        """Generate dynamic prompt based on analysis"""
        
        # Extract key information
        print(f"domain_id --> {domain_id} ; prompt_type.value --> {prompt_type.value}")
        prompt_template = get_prompt_template(domain_id, prompt_type.value)
        prompt_input_variables = get_prompt_input_variables(domain_id, prompt_type.value)

        variable_map = {
            "query": query,
            "context": text,
            "chat_history": history_prompt
        }
        # Dynamically build the inputs dictionary
        inputs = {var: variable_map[var] for var in prompt_input_variables}
        print("Inputs : ",inputs)
        print(f"customPrompt - {prompt_template}")

        final_prompt = prompt_template.format(**inputs)
        
        """
        # Base prompt templates with dynamic elements
        prompt_templates = {
            PromptType.INSTRUCTION: self._generate_instruction_prompt(query, entity_str, domain_context, text, analysis),
            PromptType.FEW_SHOT: self._generate_few_shot_prompt(query, entity_str, domain_context, text, analysis),
            PromptType.CHAIN_OF_THOUGHT: self._generate_cot_prompt(query, entity_str, domain_context, text, analysis),
            PromptType.ROLE_BASED: self._generate_role_based_prompt(query, entity_str, domain_context, text, analysis),
            PromptType.STRUCTURED_OUTPUT: self._generate_structured_prompt(query, entity_str, domain_context, text, analysis),
            PromptType.INTERACTIVE_AGENT: self._generate_interactive_prompt(query, entity_str, domain_context, text, analysis),
            PromptType.CONTEXT_AWARE: self._generate_context_aware_prompt(query, entity_str, domain_context, text, analysis)
        }
        
        print(f"Final prompt type --> {prompt_type}; value --> {prompt_type.value}")

        # Get prompt or default to instruction
        return prompt_templates.get(prompt_type, prompt_templates[PromptType.INSTRUCTION])
        """
        return final_prompt

    
    def _get_domain_examples(self, domain: str) -> str:
        """Generate domain-specific examples"""
        examples = {
            "medical": """Example 1: "Dr. Sarah Johnson treated the patient at Mayo Clinic."
→ Entities: [{{"text": "Dr. Sarah Johnson", "type": "PERSON", "confidence": 0.95}}, {{"text": "Mayo Clinic", "type": "ORGANIZATION", "confidence": 0.92}}]

Example 2: "The patient was diagnosed with diabetes and prescribed metformin."
→ Entities: [{{"text": "diabetes", "type": "CONDITION", "confidence": 0.89}}, {{"text": "metformin", "type": "MEDICATION", "confidence": 0.94}}]""",
            
            "legal": """Example 1: "Johnson & Associates represented Apple Inc. in the patent case."
→ Entities: [{{"text": "Johnson & Associates", "type": "ORGANIZATION", "confidence": 0.93}}, {{"text": "Apple Inc.", "type": "ORGANIZATION", "confidence": 0.96}}]

Example 2: "The Supreme Court ruled in favor of the defendant."
→ Entities: [{{"text": "Supreme Court", "type": "ORGANIZATION", "confidence": 0.98}}]""",
            
            "financial": """Example 1: "Microsoft's stock price rose to $350.50 per share."
→ Entities: [{{"text": "Microsoft", "type": "ORGANIZATION", "confidence": 0.96}}, {{"text": "$350.50", "type": "MONEY", "confidence": 0.94}}]

Example 2: "Goldman Sachs invested $2.5 billion in the startup."
→ Entities: [{{"text": "Goldman Sachs", "type": "ORGANIZATION", "confidence": 0.95}}, {{"text": "$2.5 billion", "type": "MONEY", "confidence": 0.93}}]""",
            
            "general": """Example 1: "John Smith works at Google in California."
→ Entities: [{{"text": "John Smith", "type": "PERSON", "confidence": 0.92}}, {{"text": "Google", "type": "ORGANIZATION", "confidence": 0.96}}, {{"text": "California", "type": "LOCATION", "confidence": 0.89}}]

Example 2: "The meeting is scheduled for January 15, 2024."
→ Entities: [{{"text": "January 15, 2024", "type": "DATE", "confidence": 0.94}}]"""
        }
        
        return examples.get(domain, examples["general"])
    
    def _get_reasoning_steps(self, analysis: QueryAnalysis) -> str:
        """Generate reasoning steps based on analysis"""
        steps = []
        
        if analysis.complexity == QueryComplexity.SIMPLE:
            steps = [
                "1. Scan for obvious entity patterns",
                "2. Apply basic classification rules",
                "3. Verify with high-confidence matches"
            ]
        elif analysis.complexity == QueryComplexity.MODERATE:
            steps = [
                "1. Comprehensive text scanning for entity candidates",
                "2. Apply pattern matching algorithms",
                "3. Context analysis for disambiguation",
                "4. Confidence scoring and validation"
            ]
        elif analysis.complexity == QueryComplexity.COMPLEX:
            steps = [
                "1. Multi-pass entity detection with different strategies",
                "2. Advanced pattern recognition and classification",
                "3. Cross-reference validation and context analysis",
                "4. Relationship mapping between entities",
                "5. Confidence assessment and uncertainty quantification"
            ]
        else:  # EXPERT
            steps = [
                "1. Advanced multi-modal entity detection",
                "2. Sophisticated pattern analysis with domain expertise",
                "3. Cross-domain validation and fact-checking",
                "4. Complex relationship and dependency analysis",
                "5. Uncertainty quantification and confidence intervals",
                "6. Expert-level validation and quality assurance"
            ]
        
        return "\n".join(steps)
    
    def _get_expert_role(self, domain: str, complexity: QueryComplexity) -> str:
        """Generate expert role description"""
        roles = {
            "medical": {
                QueryComplexity.SIMPLE: "a medical assistant",
                QueryComplexity.MODERATE: "a clinical data analyst",
                QueryComplexity.COMPLEX: "a medical informatics specialist",
                QueryComplexity.EXPERT: "a senior medical informatics expert with 15+ years of experience"
            },
            "legal": {
                QueryComplexity.SIMPLE: "a legal assistant",
                QueryComplexity.MODERATE: "a legal research analyst",
                QueryComplexity.COMPLEX: "a legal technology specialist",
                QueryComplexity.EXPERT: "a senior legal technology expert specializing in document analysis"
            },
            "financial": {
                QueryComplexity.SIMPLE: "a financial analyst",
                QueryComplexity.MODERATE: "a senior financial data analyst",
                QueryComplexity.COMPLEX: "a financial technology specialist",
                QueryComplexity.EXPERT: "a senior fintech expert with deep knowledge of financial entity recognition"
            },
            "technical": {
                QueryComplexity.SIMPLE: "a technical analyst",
                QueryComplexity.MODERATE: "a software documentation specialist",
                QueryComplexity.COMPLEX: "a technical information extraction expert",
                QueryComplexity.EXPERT: "a senior NLP engineer specializing in technical document analysis"
            },
            "general": {
                QueryComplexity.SIMPLE: "an information extraction specialist",
                QueryComplexity.MODERATE: "a senior data analyst",
                QueryComplexity.COMPLEX: "an advanced NLP specialist",
                QueryComplexity.EXPERT: "a world-class expert in named entity recognition and natural language processing"
            }
        }
        
        return roles.get(domain, roles["general"]).get(complexity, "an expert analyst")
    
    def _generate_dynamic_schema(self, analysis: QueryAnalysis) -> str:
        """Generate dynamic JSON schema based on analysis"""
        entity_types = {e['label'] for e in analysis.entities_detected}
        if not entity_types:
            entity_types = {"PERSON", "ORGANIZATION", "LOCATION", "DATE", "MONEY"}
        
        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "The extracted entity text"},
                            "type": {"type": "string", "enum": entity_types, "description": "Entity classification"},
                            "start": {"type": "integer", "description": "Start position in text"},
                            "end": {"type": "integer", "description": "End position in text"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Confidence score"},
                            "context": {"type": "string", "description": "Surrounding context"},
                            "patterns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Matched patterns"
                            }
                        },
                        "required": ["text", "type", "start", "end", "confidence"]
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "total_entities": {"type": "integer"},
                        "confidence_threshold": {"type": "number"},
                        "processing_method": {"type": "string"},
                        "domain": {"type": "string"}
                    }
                }
            },
            "required": ["entities", "metadata"]
        }
        
        return json.dumps(schema, indent=2)
    
    def _build_context_information(self, analysis: QueryAnalysis) -> str:
        """Build context information string"""
        context_parts = []
        
        context_parts.append(f"Domain: {analysis.domain}")
        context_parts.append(f"Complexity Level: {analysis.complexity.name}")
        context_parts.append(f"Intent: {analysis.intent.value}")
        
        if analysis.patterns_found:
            patterns_str = ", ".join([p['type'] for p in analysis.patterns_found[:3]])
            context_parts.append(f"Detected Patterns: {patterns_str}")
        
        if analysis.entities_detected:
            entity_types = {e['label'] for e in analysis.entities_detected}
            context_parts.append(f"Expected Entity Types: {', '.join(entity_types)}")
        
        if analysis.context_requirements:
            context_parts.append(f"Context Requirements: {', '.join(analysis.context_requirements)}")
        
        return "\n".join([f"- {part}" for part in context_parts])
    
    def _generate_comprehensive_reasoning(self, analysis: QueryAnalysis) -> str:
        """Generate comprehensive reasoning explanation"""
        reasoning_parts = []
        
        # Intent reasoning
        reasoning_parts.append(f"Intent Analysis: Classified as '{analysis.intent.value}' with {analysis.confidence_scores['intent']:.1%} confidence based on query pattern matching and semantic analysis.")
        
        # Complexity reasoning
        reasoning_parts.append(f"Complexity Assessment: Determined as '{analysis.complexity.name}' level with {analysis.confidence_scores['complexity']:.1%} confidence using linguistic feature analysis and ML model prediction.")
        
        # Domain reasoning
        reasoning_parts.append(f"Domain Classification: Identified as '{analysis.domain}' domain with {analysis.confidence_scores['domain']:.1%} confidence through keyword analysis and domain-specific pattern matching.")
        
        # Entity reasoning
        if analysis.entities_detected:
            entity_count = len(analysis.entities_detected)
            entity_types = {e['label'] for e in analysis.entities_detected}
            reasoning_parts.append(f"Entity Detection: Found {entity_count} entities across {entity_types} different types using BERT-NER and rule-based pattern recognition.")
        
        # Pattern reasoning
        if analysis.patterns_found:
            pattern_count = len(analysis.patterns_found)
            high_conf_patterns = len([p for p in analysis.patterns_found if p['confidence'] > 0.8])
            reasoning_parts.append(f"Pattern Recognition: Identified {pattern_count} linguistic patterns with {high_conf_patterns} high-confidence matches using advanced pattern recognition algorithms.")
        
        # Model ensemble reasoning
        model_count = len(analysis.model_predictions)
        avg_confidence = analysis.confidence_scores['overall']
        reasoning_parts.append(f"Model Ensemble: Combined predictions from {model_count} specialized models with overall confidence of {avg_confidence:.1%} using weighted ensemble approach.")
        
        # Feature importance reasoning
        if analysis.feature_importance:
            top_features = sorted(analysis.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            top_feature_names = [f[0] for f in top_features]
            reasoning_parts.append(f"Feature Analysis: Most important features for this prediction were {', '.join(top_feature_names)} based on model feature importance analysis.")
        
        return "; ".join(reasoning_parts)
    
    def _generate_alternative_prompts(
        self, 
        query: str, 
        domain_id: str, 
        alternative_types: List[PromptType], 
        text: str, 
        history_prompt: List,
        analysis: QueryAnalysis) -> Dict[str, str]:

        """Generate alternative prompt options"""
        alternatives = {}
        
        for prompt_type in alternative_types:
            if prompt_type:
                alternatives[prompt_type.value] = self._generate_dynamic_prompt(query, domain_id, prompt_type, text, history_prompt, analysis)
        
        return alternatives
    
    def save_model_state(self, filepath: str):
        """Save the current state of all models"""
        try:
            model_state = {
                'complexity_model': {
                    'model': self.complexity_model.model if self.complexity_model.is_trained else None,
                    'vectorizer': self.complexity_model.feature_extractor,
                    'is_trained': self.complexity_model.is_trained
                },
                'domain_model': {
                    'model': self.domain_model.model if self.domain_model.is_trained else None,
                    'vectorizer': self.domain_model.vectorizer,
                    'label_encoder': self.domain_model.label_encoder,
                    'is_trained': self.domain_model.is_trained
                },
                'intent_patterns': self.intent_patterns,
                'entity_patterns': self.entity_recognizer.entity_patterns,
                'training_data': self.training_data,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_state, f)
            
            logger.info(f"✅ Model state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error saving model state: {e}")
    
    def load_model_state(self, filepath: str):
        """Load previously saved model state"""
        try:
            with open(filepath, 'rb') as f:
                model_state = pickle.load(f)
            
            # Restore complexity model
            if model_state['complexity_model']['is_trained']:
                self.complexity_model.model = model_state['complexity_model']['model']
                self.complexity_model.feature_extractor = model_state['complexity_model']['vectorizer']
                self.complexity_model.is_trained = True
            
            # Restore domain model
            if model_state['domain_model']['is_trained']:
                self.domain_model.model = model_state['domain_model']['model']
                self.domain_model.vectorizer = model_state['domain_model']['vectorizer']
                self.domain_model.label_encoder = model_state['domain_model']['label_encoder']
                self.domain_model.is_trained = True
            
            # Restore patterns
            self.intent_patterns = model_state['intent_patterns']
            self.entity_recognizer.entity_patterns = model_state['entity_patterns']
            
            logger.info(f"✅ Model state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model state: {e}")

def demonstrate_complete_model_system():
    """Comprehensive demonstration of the model-based system"""
    print("🧠 COMPLETE MODEL-BASED DYNAMIC PROMPT SYSTEM DEMO")
    print("=" * 80)
    
    # Initialize the complete system
    router = DynamicPromptRouter()
    
    # Test cases covering various scenarios
    test_cases = [
        {
            "query": "Extract all medical entities and explain your reasoning process",
            "text": "Dr. Sarah Johnson from Johns Hopkins Hospital diagnosed the patient with type 2 diabetes and prescribed metformin 500mg twice daily. The patient will return in 3 months for follow-up.",
            "context": "medical records analysis",
            "description": "Complex medical entity extraction with explanation request"
        },
        {
            "query": "Show me examples of how to identify legal entities in contracts",
            "text": "Johnson & Associates LLC, representing Apple Inc., filed a motion in the Superior Court of California on March 15, 2024, seeking damages of $2.5 million.",
            "context": "legal document training",
            "description": "Learning-oriented query for legal domain"
        },
        {
            "query": "I need JSON formatted results for financial entities",
            "text": "Goldman Sachs invested $50 billion in renewable energy stocks. The investment was announced by CEO David Solomon on January 10, 2024.",
            "context": "financial data processing",
            "description": "Structured output request for financial domain"
        },
        {
            "query": "Let's have an interactive discussion about entity extraction from multiple technical documents",
            "text": "The new API version 2.3.1 supports OAuth 2.0 authentication and includes endpoints for user management. The documentation is available at https://api.example.com/docs.",
            "context": "technical documentation review",
            "description": "Interactive session for batch technical processing"
        },
        {
            "query": "Quick extraction of basic entities",
            "text": "John Smith met with Jane Doe at Microsoft headquarters in Seattle on Friday.",
            "context": "simple meeting notes",
            "description": "Simple extraction request"
        }
    ]
    
    # Process each test case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*20} TEST CASE {i} {'='*20}")
        print(f"Description: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print(f"Context: {test_case['context']}")
        #print("-" * 60)
        
        # Generate optimal prompt using complete model ensemble
        result = router.generate_optimal_prompt(
            test_case["query"],
            test_case["text"],
            test_case["context"]
        )
        
        print(f"\n🎯 SELECTED PROMPT TYPE: {result['selected_prompt_type'].value.upper()}")
        print(f"🏆 ROUTING CONFIDENCE: {result['routing_confidence']:.3f}")
        
        print("\n📝 GENERATED PROMPT:")
        print("-" * 40)
        print(result["generated_prompt"])
        
        print("\n🧠 MODEL PREDICTIONS:")
        for model_name, prediction in result['model_predictions'].items():
            print(f"   • {model_name}: {prediction.model_name} - {prediction.confidence:.3f}")
        
        print("\n💡 ROUTING REASONING:")
        print(result['routing_reasoning'])
        
        if result.get('alternative_prompts'):
            print("\n🔄 ALTERNATIVE PROMPT TYPES:")
            for alt_type in result['alternative_prompts'].keys():
                print(f"   • {alt_type.upper()}")
        
        print("\n" + "="*80)
    
    # Save trained models
    try:
        router.save_model_state("model_state.pkl")
        print("💾 Model state saved successfully!")
    except Exception as e:
        print(f"❌ Error saving model state: {e}")
    
    return router

def run_performance_analysis():
    """Run performance analysis on the model system"""
    print("\n🏃‍♂️ PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Generate test queries
    test_queries = [
        "Extract person names",
        "Show me examples of medical entity extraction with detailed explanations",
        "I need comprehensive analysis of financial entities in JSON format",
        "Let's discuss advanced pattern recognition techniques",
        "Perform expert-level biomedical entity recognition"
    ]
    
    import time
    
    total_time = 0
    results = []
    
    comprehensive_analysis = ComprehensiveAnalysis()
    for query in test_queries:
        start_time = time.time()
        
        # Analyze query

        analysis = comprehensive_analysis.analyze_query_with_models(query)
        
        end_time = time.time()
        processing_time = end_time - start_time
        total_time += processing_time
        
        results.append({
            'query': query[:30] + "..." if len(query) > 30 else query,
            'processing_time': processing_time,
            'confidence': analysis.confidence_scores['overall'],
            'models_used': len(analysis.model_predictions)
        })
    
    # Display results
    print("📊 Performance Results:")
    print(f"   Total processing time: {total_time:.3f} seconds")
    print(f"   Average time per query: {total_time/len(test_queries):.3f} seconds")
    print(f"   Average confidence: {np.mean([r['confidence'] for r in results]):.3f}")
    
    print("\n📈 Individual Results:")
    for result in results:
        print(f"   • {result['query']}: {result['processing_time']:.3f}s (conf: {result['confidence']:.3f})")