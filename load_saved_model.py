"""
HDFS Anomaly Detection Model Loader

This script provides utilities to load the saved HDFS anomaly detection model
with all its components including the custom AttentionLayer and Drain3 parser.

For beginners:
- This handles all the complex loading details
- Just use load_hdfs_model() to get everything you need
- The loaded model is ready for predictions or retraining
"""

import os
import pickle
import json
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_attention_layer_class(model_dir):
    """
    Load the AttentionLayer class definition from saved code
    
    This is necessary because TensorFlow needs the class definition
    to properly load models with custom layers.
    
    Parameters:
    - model_dir: Directory containing the saved model artifacts
    
    Returns:
    - AttentionLayer: The custom layer class
    """
    
    attention_code_path = os.path.join(model_dir, "attention_layer_code.py")
    
    if not os.path.exists(attention_code_path):
        raise FileNotFoundError(f"AttentionLayer code not found: {attention_code_path}")
    
    # Execute the AttentionLayer code to define the class
    with open(attention_code_path, 'r') as f:
        attention_code = f.read()
    
    # Create a local namespace to execute the code
    local_namespace = {}
    exec(attention_code, globals(), local_namespace)
    
    # Return the AttentionLayer class
    return local_namespace['AttentionLayer']

def load_hdfs_model(model_dir="Saved_ModelAndArtifacts"):
    """
    Load the complete HDFS anomaly detection model with all artifacts
    
    This function loads everything needed to use the model:
    1. The trained Keras model with custom AttentionLayer
    2. Label encoder for preprocessing events
    3. Drain3 parser for processing new logs
    4. Configuration and metadata
    5. Training history
    
    Parameters:
    - model_dir: Directory containing the saved model artifacts
    
    Returns:
    - model: Loaded Keras model ready for predictions
    - artifacts: Dictionary containing all other components
    """
    
    print(f"Loading HDFS Anomaly Detection Model from: {model_dir}")
    print("="*60)
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    artifacts = {}
    
    try:
        # 1. Load the AttentionLayer class
        print("1. Loading AttentionLayer class...")
        AttentionLayer = load_attention_layer_class(model_dir)
        print("   ✓ AttentionLayer class loaded")
        
        # 2. Load the trained model with custom objects
        print("\n2. Loading trained model...")
        model_path = os.path.join(model_dir, "HDFS_Anomaly_Model.h5")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
        print(f"   ✓ Model loaded: {model_path}")
        
        # 3. Load label encoder
        print("\n3. Loading label encoder...")
        encoder_path = os.path.join(model_dir, "Label_encoder.joblib")
        
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {encoder_path}")
        
        label_encoder = joblib.load(encoder_path)
        artifacts['label_encoder'] = label_encoder
        print(f"   ✓ Label encoder loaded: {encoder_path}")
        print(f"   - Vocabulary size: {len(label_encoder.classes_)}")
        
        # 4. Load configuration
        print("\n4. Loading configuration...")
        config_path = os.path.join(model_dir, "model_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            artifacts['config'] = config
            print(f"   ✓ Configuration loaded: {config_path}")
            print(f"   - Max sequence length: {config['model_info']['max_length']}")
            print(f"   - Training accuracy: {config['training_info']['final_train_accuracy']:.4f}")
            print(f"   - Validation accuracy: {config['training_info']['final_val_accuracy']:.4f}")
        else:
            print("   ⚠️  Configuration file not found, using defaults")
            config = {'model_info': {'max_length': 50, 'vocab_size': len(label_encoder.classes_)}}
            artifacts['config'] = config
        
        # 5. Load Drain3 parser (if available)
        print("\n5. Loading Drain3 parser...")
        drain_path = os.path.join(model_dir, "Drain_state.pkl")
        
        if os.path.exists(drain_path):
            with open(drain_path, 'rb') as f:
                drain_parser = pickle.load(f)
            artifacts['drain_parser'] = drain_parser
            print(f"   ✓ Drain3 parser loaded: {drain_path}")
        else:
            print("   ⚠️  Drain3 parser not found")
            artifacts['drain_parser'] = None
        
        # 6. Load attention layer weights (if available)
        print("\n6. Loading attention layer artifacts...")
        attention_path = os.path.join(model_dir, "attention_layer.pkl")
        
        if os.path.exists(attention_path):
            with open(attention_path, 'rb') as f:
                attention_data = pickle.load(f)
            artifacts['attention_data'] = attention_data
            print(f"   ✓ Attention layer data loaded: {attention_path}")
        else:
            print("   ⚠️  Attention layer data not found")
        
        # 7. Load training history (if available)
        print("\n7. Loading training history...")
        history_path = os.path.join(model_dir, "training_history.pkl")
        
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                training_history = pickle.load(f)
            artifacts['training_history'] = training_history
            print(f"   ✓ Training history loaded: {history_path}")
        else:
            print("   ⚠️  Training history not found")
        
        # 8. Add convenience attributes
        artifacts['vocab_size'] = len(label_encoder.classes_)
        artifacts['max_length'] = config['model_info']['max_length']
        artifacts['event_classes'] = label_encoder.classes_.tolist()
        artifacts['AttentionLayer'] = AttentionLayer  # For creating new models
        
        print(f"\n{'='*60}")
        print("MODEL LOADING COMPLETE")
        print(f"{'='*60}")
        print("✅ Successfully loaded:")
        print(f"   - Trained model with custom AttentionLayer")
        print(f"   - Label encoder ({len(label_encoder.classes_)} events)")
        print(f"   - Configuration and metadata")
        if artifacts.get('drain_parser'):
            print(f"   - Drain3 parser for log processing")
        if artifacts.get('training_history'):
            print(f"   - Training history for analysis")
        
        return model, artifacts
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the model directory exists and contains all required files")
        print("2. Check that TensorFlow is properly installed")
        print("3. Verify that the model was saved correctly")
        print("4. Try loading individual components to identify the issue")
        raise

def predict_with_loaded_model(model, artifacts, event_sequences, threshold=0.5):
    """
    Make predictions using the loaded model
    
    Parameters:
    - model: Loaded Keras model
    - artifacts: Model artifacts (contains label_encoder, config, etc.)
    - event_sequences: List of event sequences (list of lists of event IDs)
    - threshold: Decision threshold for anomaly classification
    
    Returns:
    - predictions: Binary predictions (0=normal, 1=anomaly)
    - probabilities: Prediction probabilities
    - details: Detailed prediction information
    """
    
    print(f"Making predictions for {len(event_sequences)} sequences...")
    
    label_encoder = artifacts['label_encoder']
    max_length = artifacts['max_length']
    known_events = set(artifacts['event_classes'])
    
    # Encode sequences
    X_encoded = []
    unknown_events = set()
    
    for seq in event_sequences:
        encoded_seq = []
        for event in seq:
            if event in known_events:
                encoded_seq.append(label_encoder.transform([event])[0])
            else:
                unknown_events.add(event)
                encoded_seq.append(0)  # Use padding value for unknown events
        X_encoded.append(encoded_seq)
    
    # Pad sequences
    X_padded = pad_sequences(X_encoded, maxlen=max_length, padding="post", value=0)
    
    # Make predictions
    probabilities = model.predict(X_padded, verbose=0).flatten()
    predictions = (probabilities > threshold).astype(int)
    
    # Create detailed results
    details = {
        'total_sequences': len(event_sequences),
        'unknown_events': list(unknown_events),
        'unknown_event_count': len(unknown_events),
        'sequence_lengths': [len(seq) for seq in event_sequences],
        'padded_shape': X_padded.shape,
        'threshold': threshold,
        'anomaly_count': np.sum(predictions),
        'normal_count': len(predictions) - np.sum(predictions)
    }
    
    if unknown_events:
        print(f"⚠️  Found {len(unknown_events)} unknown events: {list(unknown_events)[:5]}...")
        print("   These were treated as padding (value=0)")
    
    print(f"✅ Predictions complete:")
    print(f"   - Normal sequences: {details['normal_count']}")
    print(f"   - Anomaly sequences: {details['anomaly_count']}")
    
    return predictions, probabilities, details

def create_prediction_pipeline(model_dir="Saved_ModelAndArtifacts"):
    """
    Create a ready-to-use prediction pipeline
    
    Returns a class that encapsulates the model and provides easy prediction methods
    """
    
    class HDFSAnomalyPredictor:
        """
        Production-ready HDFS anomaly prediction pipeline
        """
        
        def __init__(self, model_dir):
            self.model_dir = model_dir
            self.model = None
            self.artifacts = None
            self.load_model()
        
        def load_model(self):
            """Load the model and artifacts"""
            self.model, self.artifacts = load_hdfs_model(self.model_dir)
        
        def predict_single(self, event_sequence, threshold=0.5):
            """Predict anomaly for a single sequence"""
            predictions, probabilities, _ = predict_with_loaded_model(
                self.model, self.artifacts, [event_sequence], threshold
            )
            return predictions[0], probabilities[0]
        
        def predict_batch(self, event_sequences, threshold=0.5):
            """Predict anomalies for multiple sequences"""
            return predict_with_loaded_model(
                self.model, self.artifacts, event_sequences, threshold
            )
        
        def get_model_info(self):
            """Get model information"""
            config = self.artifacts.get('config', {})
            return {
                'vocab_size': self.artifacts['vocab_size'],
                'max_length': self.artifacts['max_length'],
                'event_count': len(self.artifacts['event_classes']),
                'training_accuracy': config.get('training_info', {}).get('final_train_accuracy', 'Unknown'),
                'validation_accuracy': config.get('training_info', {}).get('final_val_accuracy', 'Unknown')
            }
    
    return HDFSAnomalyPredictor(model_dir)

# Example usage
if __name__ == "__main__":
    print("HDFS Anomaly Detection Model Loader - Example Usage")
    print("="*60)
    
    try:
        # Load the model
        model, artifacts = load_hdfs_model()
        
        # Create prediction pipeline
        predictor = create_prediction_pipeline()
        
        # Show model info
        info = predictor.get_model_info()
        print(f"\nModel Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Example prediction (using first few known events)
        if len(artifacts['event_classes']) >= 3:
            sample_sequence = artifacts['event_classes'][:3]
            is_anomaly, probability = predictor.predict_single(sample_sequence)
            
            print(f"\nExample Prediction:")
            print(f"  Sequence: {sample_sequence}")
            print(f"  Result: {'Anomaly' if is_anomaly else 'Normal'}")
            print(f"  Probability: {probability:.3f}")
        
        print(f"\n✅ Model loading example completed successfully!")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("Make sure you have trained and saved a model first.")