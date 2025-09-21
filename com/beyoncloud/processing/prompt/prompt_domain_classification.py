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
from com.beyoncloud.schemas.prompt_datamodel import QueryComplexity

logger = logging.getLogger(__name__)

class DomainClassificationModel:
    """Model for domain classification"""
    
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.domain_keywords = self._initialize_domain_keywords()

        # Training data
        self.domain_training_data = self._generate_domain_training_data()
        # Train domain model
        domain_queries = self.domain_training_data["domain"]["queries"]
        domain_labels = self.domain_training_data["domain"]["domains"]
        self.train(domain_queries, domain_labels)
    
    def _generate_domain_training_data(self) -> Dict[str, List]:
        """Generate synthetic training data for models"""

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
                "Analyze general text", "Process everyday documents", "general",

                # Insurance
                "insurance"
            ],
            "domains": [
                "medical", "medical", "medical", "medical", "medical",
                "legal", "legal", "legal", "legal", "legal",
                "financial", "financial", "financial", "financial", "financial",
                "technical", "technical", "technical", "technical", "technical",
                "academic", "academic", "academic", "academic", "academic",
                "general", "general", "general", "general", "general", "general",
                "insurance"
            ]
        }
        
        return {
            "domain": domain_data
        }

    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        """Initialize domain-specific keywords"""
        return {
            "medical": ["patient", "doctor", "hospital", "diagnosis", "treatment", "clinical", "medical", "health", "drug", "symptom"],
            "legal": ["court", "judge", "lawyer", "case", "law", "legal", "contract", "litigation", "attorney", "defendant"],
            "financial": ["bank", "investment", "stock", "financial", "money", "trading", "market", "finance", "credit", "loan"],
            "technical": ["software", "algorithm", "system", "technology", "engineering", "code", "programming", "database", "API"],
            "academic": ["research", "study", "university", "professor", "academic", "publication", "journal", "thesis", "conference"],
            "business": ["company", "business", "corporate", "management", "strategy", "market", "sales", "revenue", "customer"],
            "scientific": ["experiment", "hypothesis", "data", "analysis", "research", "laboratory", "scientific", "methodology"],
            "general": ["general", "common", "basic", "standard", "typical", "normal", "everyday", "simple"]
        }
    
    def train(self, queries: List[str], domains: List[str]):
        """Train domain classification model"""
        # Vectorize queries
        X = self.vectorizer.fit_transform(queries)
        
        # Encode labels
        y = self.label_encoder.fit_transform(domains)
        
        # Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Domain classification accuracy: {accuracy:.3f}")
        
        self.is_trained = True
    
    def predict_domain(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        """Predict domain of query"""
        print(f"is_trained - {self.is_trained} ")
        if not self.is_trained:
            return self._rule_based_domain(query)
        
        # Vectorize query
        X = self.vectorizer.transform([query])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        print(f"prediction - {prediction} ; probabilities -- {probabilities}")
        
        # Get domain name
        domain = self.label_encoder.inverse_transform([prediction])[0]
        confidence = float(max(probabilities))
        
        # Get probability distribution
        prob_dict = {}
        for i, prob in enumerate(probabilities):
            domain_name = self.label_encoder.inverse_transform([i])[0]
            prob_dict[domain_name] = float(prob)
        
        return domain, confidence, prob_dict
    
    def _rule_based_domain(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        """Fallback rule-based domain classification"""
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            print(f"domain - {domain} ; keywords -- {keywords}")
            score = sum(1 for keyword in keywords if keyword in query_lower)
            domain_scores[domain] = score
        
        best_domain = max(domain_scores, key=domain_scores.get)
        max_score = domain_scores[best_domain]
        confidence = min(max_score / 3.0, 1.0) if max_score > 0 else 0.5
        
        return best_domain, confidence, domain_scores