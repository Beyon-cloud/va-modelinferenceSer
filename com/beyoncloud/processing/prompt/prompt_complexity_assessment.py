import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
import re
import pickle
from typing import List, Dict, Tuple
import warnings
import logging
warnings.filterwarnings('ignore')
from com.beyoncloud.schemas.prompt_datamodel import QueryComplexity

logger = logging.getLogger(__name__)

class ComplexityAssessmentModel:
    """ML model for query complexity assessment"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1, random_state=42)
        self.feature_extractor = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def extract_linguistic_features(self, query: str) -> Dict[str, float]:
        """Extract linguistic features for complexity assessment"""
        features = {
            'word_count': len(query.split()),
            'char_count': len(query),
            'sentence_count': len(re.split(r'[.!?]+', query)),
            'avg_word_length': np.mean([len(word) for word in query.split()]),
            'question_marks': query.count('?'),
            'exclamation_marks': query.count('!'),
            'semicolons': query.count(';'),
            'commas': query.count(','),
            'parentheses': query.count('(') + query.count(')'),
            'technical_terms': len(re.findall(r'\b(?:API|ML|AI|algorithm|neural|model|system)\b', query.lower())),
            'complexity_words': len(re.findall(r'\b(?:complex|comprehensive|detailed|thorough|advanced|sophisticated)\b', query.lower())),
            'simple_words': len(re.findall(r'\b(?:simple|basic|easy|quick|just|only)\b', query.lower())),
            'action_words': len(re.findall(r'\b(?:extract|find|identify|analyze|compare|explain)\b', query.lower())),
            'caps_ratio': sum(1 for c in query if c.isupper()) / len(query) if query else 0,
            'digit_ratio': sum(1 for c in query if c.isdigit()) / len(query) if query else 0
        }
        return features
    
    def train(self, queries: List[str], complexities: List[QueryComplexity]):
        """Train the complexity assessment model"""
        # Extract features
        linguistic_features = [self.extract_linguistic_features(q) for q in queries]
        feature_df = pd.DataFrame(linguistic_features)
        
        # TF-IDF features
        tfidf_features = self.feature_extractor.fit_transform(queries).toarray()
        
        # Combine features
        all_features = np.hstack([feature_df.to_numpy(), tfidf_features])
        
        # Encode labels
        complexity_values = [c.value for c in complexities]
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            all_features, complexity_values, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Complexity model accuracy: {accuracy:.3f}")
        
        self.is_trained = True
    
    def predict_complexity(self, query: str) -> Tuple[QueryComplexity, float, Dict[str, float]]:
        """Predict query complexity"""
        if not self.is_trained:
            # Default rule-based prediction if not trained
            return self._rule_based_complexity(query)
        
        # Extract features
        linguistic_features = self.extract_linguistic_features(query)
        feature_vector = np.array(list(linguistic_features.values())).reshape(1, -1)
        
        # TF-IDF features
        tfidf_features = self.feature_extractor.transform([query]).toarray()
        
        # Combine features
        all_features = np.hstack([feature_vector, tfidf_features])
        
        # Predict
        prediction = self.model.predict(all_features)[0]
        probabilities = self.model.predict_proba(all_features)[0]
        
        # Get feature importance
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            feature_names = list(linguistic_features.keys()) + [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            for name, importance in zip(feature_names[:len(self.model.feature_importances_)], self.model.feature_importances_):
                feature_importance[name] = float(importance)
        
        complexity = QueryComplexity(prediction)
        confidence = float(max(probabilities))
        
        return complexity, confidence, feature_importance
    
    def _rule_based_complexity(self, query: str) -> Tuple[QueryComplexity, float, Dict[str, float]]:
        """Fallback rule-based complexity assessment"""
        features = self.extract_linguistic_features(query)
        
        score = 0
        if features['word_count'] > 20: score += 1
        if features['technical_terms'] > 0: score += 1
        if features['complexity_words'] > 0: score += 1
        if features['semicolons'] > 0: score += 1
        if features['sentence_count'] > 2: score += 1
        
        if score >= 4: complexity = QueryComplexity.EXPERT
        elif score >= 3: complexity = QueryComplexity.COMPLEX
        elif score >= 1: complexity = QueryComplexity.MODERATE
        else: complexity = QueryComplexity.SIMPLE
        
        return complexity, 0.7, features