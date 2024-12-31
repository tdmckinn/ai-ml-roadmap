from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import numpy as np

class GardeningTextClassifier:
    """
    A text classifier specialized for gardening content. This classifier can 
    categorize gardening-related text into different categories like plant care 
    instructions, disease identification, or growing conditions.
    
    The classifier uses TF-IDF to convert text into numbers and a Random Forest
    to make predictions, demonstrating core NLP concepts while keeping the 
    implementation straightforward.
    """
    def __init__(self):
        # Create a pipeline that handles both text processing and classification
        # Pipeline ensures we apply the same steps to training and prediction data
        self.pipeline = Pipeline([
            # First step: Convert text into numbers using TF-IDF
            # TF-IDF helps identify important words while reducing the impact of common words
            ('vectorizer', TfidfVectorizer(
                max_features=1000,        # Limit to most frequent words for simplicity
                stop_words='english',     # Remove common English words
                ngram_range=(1, 2)        # Use both single words and pairs of words
            )),
            
            # Second step: Classify the text using Random Forest
            ('classifier', RandomForestClassifier(
                n_estimators=10,          # Start with a small number of trees
                random_state=42           # For reproducible results
            ))
        ])
        
    def train(self, texts, labels):
        """
        Train the model with examples of gardening text and their categories.
        
        Parameters:
        texts: List of strings - The gardening-related text content
        labels: List of strings - The category for each text
        
        Example:
        texts = [
            "Water the plant weekly and keep in partial shade",
            "Black spots on leaves indicate fungal infection",
            "Prepare soil with compost before planting"
        ]
        labels = ["care_instructions", "disease_identification", "planting_guide"]
        """
        # The pipeline handles text processing and model training in one step
        self.pipeline.fit(texts, labels)
        
        # Store the possible categories for later use
        self.categories = list(set(labels))
        
    def predict(self, text):
        """
        Classify new gardening text into categories.
        
        Parameters:
        text: String - The gardening text to classify
        
        Returns:
        dict with prediction and confidence score
        
        Example:
        >>> classifier.predict("Water deeply once per week")
        {"category": "care_instructions", "confidence": 0.85}
        """
        # Make sure input is a string
        if not isinstance(text, str):
            raise ValueError("Input must be a string of text")
            
        # Get both the prediction and confidence scores
        prediction = self.pipeline.predict([text])[0]
        confidence = float(self.pipeline.predict_proba([text]).max())
        
        return {
            "category": prediction,
            "confidence": confidence
        }
    
    def get_important_words(self, text):
        """
        Identify which words were most important in making the classification.
        This helps understand why the model made its decision.
        
        Parameters:
        text: String - The text to analyze
        
        Returns:
        dict with the prediction and most influential words
        """
        # Get the vectorizer and classifier from our pipeline
        vectorizer = self.pipeline.named_steps['vectorizer']
        classifier = self.pipeline.named_steps['classifier']
        
        # Transform the text into numbers
        features = vectorizer.transform([text])
        
        # Get all words (features) that we look for
        words = vectorizer.get_feature_names_out()
        
        # Get feature importance scores
        importance = classifier.feature_importances_
        
        # Find the most important words
        top_indices = importance.argsort()[-5:][::-1]  # Get top 5 words
        top_words = [(words[i], float(importance[i])) for i in top_indices]
        
        return {
            "prediction": self.predict(text),
            "important_words": top_words
        }

# Example usage to demonstrate how it works
if __name__ == "__main__":
    # Create some example training data
    example_texts = [
        "Water the plant once a week and mist leaves daily",
        "Yellow spots on leaves spreading quickly",
        "Best soil mix includes peat moss and perlite",
        "Prune dead leaves and stems in spring",
        "White powdery coating on leaves needs treatment"
    ]
    example_labels = [
        "care_instructions",
        "disease_identification",
        "soil_preparation",
        "care_instructions",
        "disease_identification"
    ]
    
    # Create and train the classifier
    classifier = GardeningTextClassifier()
    classifier.train(example_texts, example_labels)
    
    # Try classifying new text
    new_text = "Water thoroughly when soil feels dry"
    result = classifier.predict(new_text)
    print(f"Prediction: {result}")
    
    # See which words were important
    analysis = classifier.get_important_words(new_text)
    print(f"Important words: {analysis['important_words']}")
