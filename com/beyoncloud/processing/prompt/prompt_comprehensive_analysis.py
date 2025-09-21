import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification,
    BertTokenizer, BertForSequenceClassification, BertModel, BertConfig,
    pipeline, Trainer, TrainingArguments, AutoModel
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
import re
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import logging
from datetime import datetime
import os
warnings.filterwarnings('ignore')
from com.beyoncloud.schemas.prompt_datamodel import QueryAnalysis, ModelPrediction, PromptType
from com.beyoncloud.processing.prompt.prompt_entity_recognition import AdvancedEntityRecognizer
from com.beyoncloud.processing.prompt.prompt_intent_classification import IntentClassification
from com.beyoncloud.processing.prompt.prompt_complexity_assessment import ComplexityAssessmentModel
from com.beyoncloud.processing.prompt.prompt_domain_classification import DomainClassificationModel
from com.beyoncloud.processing.prompt.prompt_pattern_recognition import PatternRecognition
from com.beyoncloud.processing.prompt.prompt_feature_importance_analysis import FeatureImportanceAnalysis
from com.beyoncloud.processing.prompt.prompt_context_requirements_analysis import ContextRequirementsAnalysis
from com.beyoncloud.processing.prompt.prompt_type_selection import PromptTypeSelection

logger = logging.getLogger(__name__)


class ComprehensiveAnalysis:

    def __init__(self):
        # Initialize all models
        self.entity_recognizer = AdvancedEntityRecognizer()
        self.intent_classification = IntentClassification()
        self.complexity_model = ComplexityAssessmentModel()
        self.domain_model = DomainClassificationModel()
        self.pattern_recognition = PatternRecognition()
        self.context_requirements_analysis = ContextRequirementsAnalysis()
        self.promptype_selection = PromptTypeSelection()
        self.feature_importance_analysis = FeatureImportanceAnalysis()
        

    def analyze_query_with_models(self, query: str, text: str = "", domain_id: str = "") -> QueryAnalysis:
        """Complete model-based query analysis"""
        logger.info(f"🔍 Analyzing query with models: '{query[:50]}...'")
        
        model_predictions = {}
        
        # 1. Entity Recognition with Advanced Patterns
        entities = self.entity_recognizer.extract_entities_with_patterns(query + " " + text)
        entity_prediction = ModelPrediction(
            prediction=[e['label'] for e in entities],
            confidence=np.mean([e['confidence'] for e in entities]) if entities else 0.0,
            probabilities={e['label']: e['confidence'] for e in entities},
            features_used=['bert_ner', 'rule_based', 'pattern_recognition'],
            model_name='AdvancedEntityRecognizer',
            timestamp=datetime.now()
        )
        model_predictions['entities'] = entity_prediction
        
        # 2. Intent Classification
        intent, intent_confidence = self.intent_classification.predict_intent_with_model(query)
        intent_prediction = ModelPrediction(
            prediction=intent.value,
            confidence=intent_confidence,
            probabilities={intent.value: intent_confidence},
            features_used=['keyword_matching', 'pattern_recognition', 'semantic_similarity'],
            model_name='IntentClassifier',
            timestamp=datetime.now()
        )
        model_predictions['intent'] = intent_prediction
        
        # 3. Complexity Assessment
        complexity, complexity_confidence, complexity_features = self.complexity_model.predict_complexity(query)
        complexity_prediction = ModelPrediction(
            prediction=complexity.value,
            confidence=complexity_confidence,
            probabilities={str(complexity.value): complexity_confidence},
            features_used=list(complexity_features.keys()),
            model_name='ComplexityAssessmentModel',
            timestamp=datetime.now()
        )
        model_predictions['complexity'] = complexity_prediction
        
        # 4. Domain Classification
        domain, domain_confidence, domain_probs = self.domain_model.predict_domain(query + " " + domain_id)
        domain_prediction = ModelPrediction(
            prediction=domain,
            confidence=domain_confidence,
            probabilities=domain_probs,
            features_used=['tfidf_features', 'keyword_matching'],
            model_name='DomainClassificationModel',
            timestamp=datetime.now()
        )
        model_predictions['domain'] = domain_prediction
        
        # 5. Pattern Recognition
        patterns_found = self.pattern_recognition._recognize_query_patterns(query)
        pattern_prediction = ModelPrediction(
            prediction=[p['type'] for p in patterns_found],
            confidence=np.mean([p['confidence'] for p in patterns_found]) if patterns_found else 0.0,
            probabilities={p['type']: p['confidence'] for p in patterns_found},
            features_used=['regex_patterns', 'linguistic_analysis'],
            model_name='PatternRecognitionModel',
            timestamp=datetime.now()
        )
        model_predictions['patterns'] = pattern_prediction
        
        # 6. Context Requirements Analysis
        context_requirements = self.context_requirements_analysis._analyze_context_requirements_with_model(query, entities)
        
        # 7. Prompt Type Selection using Model Ensemble
        suggested_prompt_types = self.promptype_selection._select_prompt_types_with_ensemble(
            intent, complexity, domain, entities, patterns_found
        )
        print(f"intent - {intent}")
        print(f"complexity - {complexity}")
        print(f"domain - {domain}")
        print(f"entities - {entities}")
        print(f"patterns_found - {patterns_found}")
        print(f"suggested_prompt_types - {suggested_prompt_types}")
        
        # 8. Calculate confidence scores
        confidence_scores = {
            'overall': np.mean([pred.confidence for pred in model_predictions.values()]),
            'intent': intent_confidence,
            'complexity': complexity_confidence,
            'domain': domain_confidence,
            'entities': entity_prediction.confidence
        }
        
        # 9. Feature importance analysis
        feature_importance = self.feature_importance_analysis._calculate_feature_importance(model_predictions, complexity_features)
        
        # Create comprehensive analysis
        analysis = QueryAnalysis(
            intent=intent,
            complexity=complexity,
            entities_detected=entities,
            domain=domain,
            patterns_found=patterns_found,
            context_requirements=context_requirements,
            suggested_prompt_types=suggested_prompt_types,
            confidence_scores=confidence_scores,
            feature_importance=feature_importance,
            model_predictions=model_predictions
        )
        
        self._log_detailed_analysis(analysis)
        return analysis

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
        
        logger.info(f"🎨 Suggested Prompt Types:")
        for i, prompt_type in enumerate(analysis.suggested_prompt_types[:3], 1):
            logger.info(f"   {i}. {prompt_type.value.upper()}")
        
        logger.info(f"🏆 Overall Confidence: {analysis.confidence_scores['overall']:.3f}")
        
        logger.info(f"🔬 Model Predictions Summary:")
        for model_name, prediction in analysis.model_predictions.items():
            logger.info(f"   • {model_name}: {prediction.confidence:.3f} confidence")
        
        logger.info("=" * 80)