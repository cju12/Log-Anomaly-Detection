"""
HDFS Anomaly Prediction Script

This script demonstrates how to use the trained model to predict anomalies
in new HDFS log sequences.

For beginners:
- This shows how to load a saved model and make predictions
- You can use this to detect anomalies in real-time or batch processing
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets.Preprocessing import initialize_data

# Initialize the preprocessed data
X_padded, y, vocab_size, LabelEnc, grouped = initialize_data()
import matplotlib.pyplot as plt

def load_trained_model(model_path='hdfs_anomaly_model.h5'):
    """
    Load the trained anomaly detection model
    
    Parameters:
    - model_path: Path to the saved model file
    
    Returns:
    - Loaded Keras model
    """
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first by running Detection_Model.py")
        return None

def predict_sequence_anomaly(model, event_sequence, threshold=0.5):
    """
    Predict if a sequence of events is anomalous
    
    Parameters:
    - model: Trained Keras model
    - event_sequence: List of event IDs (strings like ['E1', 'E2', 'E3'])
    - threshold: Decision threshold (default 0.5)
    
    Returns:
    - prediction: 0 (normal) or 1 (anomaly)
    - confidence: Probability score (0-1)
    """
    try:
        # Encode the sequence
        encoded_seq = LabelEnc.transform(event_sequence)
        
        # Pad to model's expected length
        padded_seq = pad_sequences([encoded_seq], maxlen=50, padding="post", value=0)
        
        # Make prediction
        prob = model.predict(padded_seq, verbose=0)[0][0]
        prediction = 1 if prob > threshold else 0
        
        return prediction, prob
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None

def analyze_sample_blocks(model, num_samples=10):
    """
    Analyze a few sample blocks from the dataset
    
    This shows how the model performs on known data
    """
    print("\nAnalyzing Sample Blocks:")
    print("="*60)
    
    # Get random samples
    sample_indices = np.random.choice(len(grouped), num_samples, replace=False)
    
    correct_predictions = 0
    
    for i, idx in enumerate(sample_indices):
        block_id = grouped.iloc[idx]['BlockId']
        event_sequence = grouped.iloc[idx]['EventId']
        actual_label = grouped.iloc[idx]['Label']
        
        # Make prediction
        pred_label, confidence = predict_sequence_anomaly(model, event_sequence)
        
        if pred_label is not None:
            # Check if prediction is correct
            is_correct = pred_label == actual_label
            if is_correct:
                correct_predictions += 1
            
            # Display results
            actual_str = "Anomaly" if actual_label == 1 else "Normal"
            pred_str = "Anomaly" if pred_label == 1 else "Normal"
            status = "✓" if is_correct else "✗"
            
            print(f"{i+1:2d}. Block: {block_id}")
            print(f"    Events: {event_sequence[:5]}... ({len(event_sequence)} total)")
            print(f"    Actual: {actual_str}, Predicted: {pred_str} ({confidence:.3f}) {status}")
            print()
    
    accuracy = correct_predictions / num_samples
    print(f"Sample Accuracy: {accuracy:.1%} ({correct_predictions}/{num_samples})")

def predict_custom_sequence():
    """
    Example of predicting a custom sequence
    
    This shows how you might use the model with new data
    """
    print("\nCustom Sequence Prediction Example:")
    print("="*40)
    
    # Example: Create a custom sequence (you would get this from new log data)
    # These should be actual event IDs from your training data
    available_events = list(LabelEnc.classes_)[:10]  # First 10 event types
    custom_sequence = np.random.choice(available_events, 8).tolist()
    
    print(f"Custom sequence: {custom_sequence}")
    
    # Load model and make prediction
    model = load_trained_model()
    if model is not None:
        prediction, confidence = predict_sequence_anomaly(model, custom_sequence)
        
        if prediction is not None:
            result = "Anomaly" if prediction == 1 else "Normal"
            print(f"Prediction: {result} (confidence: {confidence:.3f})")
            
            # Interpretation
            if confidence > 0.8:
                print("High confidence prediction")
            elif confidence > 0.6:
                print("Medium confidence prediction")
            else:
                print("Low confidence prediction - might need more investigation")

def plot_confidence_distribution(model, num_samples=100):
    """
    Plot the distribution of prediction confidences
    
    This helps understand how confident the model is in its predictions
    """
    print(f"\nAnalyzing confidence distribution for {num_samples} samples...")
    
    # Get random samples
    sample_indices = np.random.choice(len(grouped), num_samples, replace=False)
    
    normal_confidences = []
    anomaly_confidences = []
    
    for idx in sample_indices:
        event_sequence = grouped.iloc[idx]['EventId']
        actual_label = grouped.iloc[idx]['Label']
        
        _, confidence = predict_sequence_anomaly(model, event_sequence)
        
        if confidence is not None:
            if actual_label == 0:  # Normal
                normal_confidences.append(confidence)
            else:  # Anomaly
                anomaly_confidences.append(confidence)
    
    # Create plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(normal_confidences, bins=20, alpha=0.7, label='Normal blocks', color='blue')
    plt.hist(anomaly_confidences, bins=20, alpha=0.7, label='Anomaly blocks', color='red')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution by True Label')
    plt.legend()
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision threshold')
    
    plt.subplot(1, 2, 2)
    all_confidences = normal_confidences + anomaly_confidences
    plt.boxplot([normal_confidences, anomaly_confidences], 
                labels=['Normal', 'Anomaly'])
    plt.ylabel('Prediction Confidence')
    plt.title('Confidence Box Plot')
    
    plt.tight_layout()
    plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function demonstrating various prediction capabilities
    """
    print("HDFS Anomaly Detection - Prediction Demo")
    print("="*50)
    
    # Load the trained model
    model = load_trained_model()
    
    if model is None:
        print("Cannot proceed without a trained model.")
        print("Please run Detection_Model.py first to train the model.")
        return
    
    # Analyze sample blocks
    analyze_sample_blocks(model, num_samples=10)
    
    # Predict custom sequence
    predict_custom_sequence()
    
    # Plot confidence distribution
    plot_confidence_distribution(model, num_samples=200)
    
    print("\nPrediction demo completed!")

if __name__ == "__main__":
    main()