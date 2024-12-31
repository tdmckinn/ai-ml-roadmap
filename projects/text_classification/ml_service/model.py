# projects/text_classification/ml_service/model.py

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

class PlantClassifier:
    """
    A basic supervised learning model for plant classification.
    This demonstrates core concepts while keeping the implementation simple.
    """
    def __init__(self):
        # RandomForestClassifier is a good starting point because:
        # 1. It handles different types of data well
        # 2. It's less prone to overfitting than a single decision tree
        # 3. It can tell us which features are most important
        self.model = RandomForestClassifier(n_estimators=10)
        
    def train(self, features, labels):
        """
        Train the model with examples.
        features: List of plant measurements (like height, leaf width)
        labels: List of plant names corresponding to each feature set
        """
        # Convert inputs to numpy arrays for scikit-learn
        X = np.array(features)
        y = np.array(labels)
        
        # The actual learning happens here
        self.model.fit(X, y)
        
    def predict(self, features):
        """
        Make a prediction about what type of plant we're looking at
        """
        # Convert input to the right shape
        X = np.array(features).reshape(1, -1)
        
        # Get both the prediction and how confident we are
        prediction = self.model.predict(X)[0]
        confidence = self.model.predict_proba(X).max()
        
        return {
            "plant_type": prediction,
            "confidence": float(confidence)
        }
