from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import load_dataset

import numpy as np
import pandas as pd
import cv2
import io
from PIL import Image

def process_image(image_bytes, target_size=(64, 64)):
    # Convert bytes to image
    image = Image.open(io.BytesIO(image_bytes))
    # Convert to numpy array and resize
    image = np.array(image)
    image = cv2.resize(image, target_size)
    # Flatten the image
    return image.flatten()



if __name__ == "__main__":
    
    # Load the dataset with its metadata
    dataset = load_dataset("jbarat/plant_species")

    # Get the label names from the dataset features
    label_names = dataset['train'].features['label'].names
    
    print(label_names)
    # 1. First, you're reading a dataset
    df = pd.read_parquet("hf://datasets/jbarat/plant_species/data/train-00000-of-00001-15efca0bf2e6a460.parquet")

    # Your data has two columns:
    # 'image': Contains binary image data (like a blob in a database)
    # 'label': Contains numbers (0-7) representing different plant species
    print("Available columns:", df.columns)
    print("First few rows:", df.head())


    # select relevant columns
    category_mapping = {
        '0aechmea_fasciata': 'care_instructions',
        '0aechmea_floribunda': 'care_instructions',
        '0aechmea_grandiflora': 'care_instructions',
        '0aechmea_fasciata': 'care_instructions',
        '2agave_attenuata': 'care_instructions',
        '0aechmea_guatemalensis': 'care_instructions',
        '0aechmea_guatemalensis': 'care_instructions',
        '2agave_attenuata': 'care_instructions',
        '2agave_attenuata': 'care_instructions',
        '2agave_attenuata': 'care_instructions',
    }

    df['image'] = df['image']
    df['image_bytes'] = df['image'].apply(lambda x: x['bytes'])

    # Extract and process images
    X = df['image'].apply(lambda x: process_image(x['bytes']))
    y = df['label']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to numpy arrays
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)

    # create, train, and evaluate the model 
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

  
class PlantSpeciesClassifier:
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
            # TF-IDF helps identify important words while reducing the impact of common wordsxa
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
