import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from com.beyoncloud.schemas.prompt_datamodel import ModelPrediction, QueryComplexity, PromptType

logger = logging.getLogger(__name__)


class FeatureImportanceAnalysis:

    def __init__(self):
        # Intentionally empty for now.
        # Reason: This class does not require instance state at construction
        # and will initialize attributes lazily when the analysis runs.
        # If future attributes are needed, initialize them here.
        pass

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