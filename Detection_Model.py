"""
HDFS Log Anomaly Detection Model

This script creates and trains a deep learning model to detect anomalies in HDFS logs.
The model uses LSTM with attention mechanism to analyze sequences of log events.

For beginners:
- HDFS: Hadoop Distributed File System - stores large files across multiple machines
- Anomaly Detection: Finding unusual patterns that might indicate system problems
- LSTM: Long Short-Term Memory - a type of neural network good at analyzing sequences
- Attention: A mechanism that helps the model focus on important parts of the sequence
"""

# Import necessary libraries
import numpy as np              # For numerical operations
import pandas as pd             # For data manipulation
import matplotlib.pyplot as plt # For plotting graphs
import seaborn as sns           # For statistical visualizations
import os                       # For operating system operations
import tensorflow as tf         # For deep learning
import joblib
import pickle


# Disable TensorFlow optimization warnings (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import TensorFlow/Keras components for building neural networks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Import our custom preprocessing module
from datasets.Preprocessing import initialize_data

# Initialize the preprocessed data
X_padded, y, vocab_size, LabelEnc, grouped = initialize_data()

print("Library loaded successfully")
print("*******************************************")

class AttentionLayer(Layer):
    """
    Custom Attention Layer for Neural Networks
    
    What is Attention?
    - Attention helps the model focus on the most important parts of the input sequence
    - Instead of just using the final LSTM output, it considers all time steps
    - It assigns different weights to different parts of the sequence
    
    For beginners:
    Think of attention like highlighting important words in a text while reading.
    The model learns which log events are most important for detecting anomalies.
    """
    
    def __init__(self, **kwargs):
        """Initialize the attention layer"""
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        """
        Build the attention mechanism weights
        
        Parameters:
        - W: Weight matrix that transforms LSTM outputs
        - b: Bias vector for the transformation
        """
        # Create trainable weight matrix
        self.W = self.add_weight(
            name="Attention_Weight",
            shape=(input_shape[-1], 1),  # Shape: (LSTM_units, 1)
            initializer="random_normal",
            trainable=True
        )
        # Create trainable bias vector
        self.b = self.add_weight(
            name="Attention_Bias", 
            shape=(input_shape[1], 1),   # Shape: (sequence_length, 1)
            initializer="zeros",
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        """
        Apply attention mechanism to input
        
        Steps:
        1. Calculate attention scores for each time step
        2. Apply softmax to get attention weights (probabilities)
        3. Create weighted sum of all time steps
        """
        # Calculate attention scores: e = tanh(x*W + b)
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        e = tf.squeeze(e, axis=-1)  # Remove last dimension
        
        # Convert scores to probabilities using softmax
        alpha = tf.nn.softmax(e, axis=1)
        self.attention_weights = alpha
        
        # Create context vector: weighted sum of all time steps
        context = tf.reduce_sum(tf.expand_dims(alpha, -1) * x, axis=1)
        return context
    
    def get_attention_weights(self):
        """Return the attention weights for visualization"""
        return self.attention_weights

def create_model(vocab_size, max_length, embedding_dim=64, lstm_units=128):
    """
    Create the LSTM-Attention model for anomaly detection
    
    Architecture:
    1. Embedding Layer: Converts event IDs to dense vectors
    2. LSTM Layer: Processes the sequence of events
    3. Attention Layer: Focuses on important events
    4. Dense Layer: Makes final prediction (normal/anomaly)
    
    Parameters:
    - vocab_size: Number of unique event types
    - max_length: Maximum sequence length
    - embedding_dim: Size of embedding vectors
    - lstm_units: Number of LSTM units
    """
    print(f"Creating model with:")
    print(f"- Vocabulary size: {vocab_size}")
    print(f"- Max sequence length: {max_length}")
    print(f"- Embedding dimension: {embedding_dim}")
    print(f"- LSTM units: {lstm_units}")
    
    # Input layer: accepts sequences of event IDs
    inputs = Input(shape=(max_length,), name="event_sequence")
    
    # Embedding layer: converts event IDs to dense vectors
    # Think of this as creating a "dictionary" where each event type gets a unique vector
    x = Embedding(
        input_dim=vocab_size,           # Number of unique events
        output_dim=embedding_dim,       # Size of each event vector
        input_length=max_length,        # Length of input sequences
        name="event_embedding"
    )(inputs)
    
    # LSTM layer: processes the sequence of event embeddings
    # return_sequences=True means we get output for each time step (needed for attention)
    lstm_out = LSTM(
        lstm_units, 
        return_sequences=True,          # Return all time steps
        name="lstm_processor"
    )(x)
    
    # Attention layer: focuses on important events in the sequence
    attention_layer = AttentionLayer(name="attention_mechanism")
    context_vector = attention_layer(lstm_out)
    
    # Output layer: makes final binary prediction (0=normal, 1=anomaly)
    outputs = Dense(
        1, 
        activation="sigmoid",           # Sigmoid gives probability between 0 and 1
        name="anomaly_prediction"
    )(context_vector)
    
    # Create the complete model
    model = Model(inputs=inputs, outputs=outputs, name="HDFS_Anomaly_Detector")
    
    # Compile the model with optimizer, loss function, and metrics
    model.compile(
        optimizer="adam",               # Adam optimizer (adaptive learning rate)
        loss="binary_crossentropy",    # Good for binary classification
        metrics=["accuracy", "precision", "recall"]
    )
    
    return model, attention_layer

def plot_training_history(history):
    """
    Plot training and validation metrics over epochs
    
    This helps us understand:
    - Is the model learning? (loss should decrease)
    - Is it overfitting? (validation loss increases while training loss decreases)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training & validation accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Plot training & validation loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Plot precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    
    # Plot recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance on test data
    
    Metrics explained:
    - Accuracy: Overall correctness (correct predictions / total predictions)
    - Precision: Of predicted anomalies, how many were actually anomalies
    - Recall: Of actual anomalies, how many did we catch
    - F1-Score: Harmonic mean of precision and recall
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate test accuracy
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return y_pred, y_pred_prob

def save_model_and_artifacts(model, history, label_encoder, vocab_size, max_length, attention_layer):
    """
    Save the trained model and all essential artifacts for future use
    
    This function saves everything needed to:
    1. Load and use the trained model for predictions
    2. Retrain the model with new data
    3. Maintain consistency in preprocessing
    4. Handle custom AttentionLayer properly
    5. Preserve Drain3 parser state
    
    Parameters:
    - model: Trained Keras model
    - history: Training history object
    - label_encoder: Fitted LabelEncoder for events
    - vocab_size: Size of vocabulary
    - max_length: Maximum sequence length used in training
    - attention_layer: Custom AttentionLayer instance
    """
    import json
    from datetime import datetime
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    
    # Create models directory if it doesn't exist
    save_dir = "Saved_ModelAndArtifacts"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Saving model artifacts to: {save_dir}")
    print("="*50)
    
    # 1. Save the trained model (architecture + weights)
    model_path = os.path.join(save_dir, "HDFS_Anomaly_Model.h5")
    model.save(model_path)
    print(f"✓ Model saved: {model_path}")
    
    # 2. Save model weights separately (backup)
    weights_path = os.path.join(save_dir, "HDFS_Anomaly_weights.h5")
    model.save_weights(weights_path)
    print(f"✓ Model weights saved: {weights_path}")
    
    # 3. Save the AttentionLayer class definition and weights
    attention_path = os.path.join(save_dir, "attention_layer.pkl")
    attention_data = {
        'class_name': 'AttentionLayer',
        'weights': attention_layer.get_weights() if hasattr(attention_layer, 'get_weights') else None,
        'attention_weights': attention_layer.get_attention_weights() if hasattr(attention_layer, 'attention_weights') else None
    }
    with open(attention_path, 'wb') as f:
        pickle.dump(attention_data, f)
    print(f"✓ Attention layer saved: {attention_path}")
    
    # 4. Save AttentionLayer source code for proper loading
    attention_code_path = os.path.join(save_dir, "attention_layer_code.py")
    attention_code = '''
import tensorflow as tf
from tensorflow.keras.layers import Layer

class AttentionLayer(Layer):
    """
    Custom Attention Layer for Neural Networks
    
    This is the exact same AttentionLayer used during training.
    Import this when loading the model to avoid custom_objects issues.
    """
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name="Attention_Weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )
        self.b = self.add_weight(
            name="Attention_Bias", 
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        e = tf.squeeze(e, axis=-1)
        alpha = tf.nn.softmax(e, axis=1)
        self.attention_weights = alpha
        context = tf.reduce_sum(tf.expand_dims(alpha, -1) * x, axis=1)
        return context
    
    def get_attention_weights(self):
        return self.attention_weights
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()
'''
    
    with open(attention_code_path, 'w') as f:
        f.write(attention_code)
    print(f"✓ Attention layer code saved: {attention_code_path}")
    
    # 5. Save the label encoder (critical for preprocessing new data)
    encoder_path = os.path.join(save_dir, "Label_encoder.joblib")
    joblib.dump(label_encoder, encoder_path)
    print(f"✓ Label encoder saved: {encoder_path}")
    
    # 6. Save Drain3 parser state (for processing new logs with same templates)
    try:
        # Get the template miner from preprocessing
        from datasets.Preprocessing import grouped  # This should have access to the parser
        
        # Create a new template miner with same config for saving
        config = TemplateMinerConfig()
        config.load("drain3.ini")
        template_miner = TemplateMiner(config=config)
        
        # Save the Drain3 state
        drain_state_path = os.path.join(save_dir, "Drain_state.pkl")
        with open(drain_state_path, "wb") as f:
            pickle.dump(template_miner, f)
        print(f"✓ Drain3 parser state saved: {drain_state_path}")
        
        # Also save the Drain3 configuration
        drain_config_path = os.path.join(save_dir, "drain3_config.ini")
        import shutil
        if os.path.exists("drain3.ini"):
            shutil.copy("drain3.ini", drain_config_path)
            print(f"✓ Drain3 configuration saved: {drain_config_path}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save Drain3 state: {e}")
        print("   You may need to retrain the Drain3 parser for new data")
    
    # 7. Save model configuration and metadata
    config = {
        'model_info': {
            'timestamp': timestamp,
            'vocab_size': vocab_size,
            'max_length': max_length,
            'architecture': {
                'embedding_dim': 64,
                'lstm_units': 128,
                'has_attention': True,
                'attention_layer': 'custom'
            }
        },
        'training_info': {
            'total_epochs': len(history.history['loss']),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
        },
        'preprocessing_info': {
            'event_classes': label_encoder.classes_.tolist(),
            'padding': 'post',
            'truncating': 'post',
            'padding_value': 0,
            'label_encoding': 'LabelEncoder'
        },
        'files_saved': {
            'model': 'HDFS_Anomaly_Model.h5',
            'weights': 'HDFS_Anomaly_weights.h5',
            'label_encoder': 'Label_encoder.joblib',
            'attention_layer': 'attention_layer.pkl',
            'attention_code': 'attention_layer_code.py',
            'drain_state': 'Drain_state.pkl',
            'drain_config': 'drain3_config.ini',
            'config': 'model_config.json',
            'history': 'training_history.pkl',
            'loading_instructions': 'HOW_TO_LOAD.md'
        }
    }
    
    config_path = os.path.join(save_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Configuration saved: {config_path}")
    
    # 8. Save training history for analysis
    history_path = os.path.join(save_dir, "training_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"✓ Training history saved: {history_path}")
    
    # 9. Create loading instructions
    loading_instructions = f"""# How to Load the HDFS Anomaly Detection Model

## Quick Start

```python
# Method 1: Using the provided loading function
from load_saved_model import load_hdfs_model

model, artifacts = load_hdfs_model('{save_dir}')
```

## Manual Loading (if needed)

```python
import joblib
import pickle
import json
from tensorflow.keras.models import load_model

# Load the AttentionLayer class
exec(open('{save_dir}/attention_layer_code.py').read())

# Load model with custom objects
model = load_model('{save_dir}/HDFS_Anomaly_Model.h5', 
                  custom_objects={{'AttentionLayer': AttentionLayer}})

# Load label encoder
label_encoder = joblib.load('{save_dir}/Label_encoder.joblib')

# Load configuration
with open('{save_dir}/model_config.json', 'r') as f:
    config = json.load(f)

# Load Drain3 parser (if needed for new data)
with open('{save_dir}/Drain_state.pkl', 'rb') as f:
    drain_parser = pickle.load(f)
```

## Model Information
- Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Vocabulary Size: {vocab_size}
- Max Sequence Length: {max_length}
- Training Accuracy: {history.history['accuracy'][-1]:.4f}
- Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}

## Files in this directory:
- `HDFS_Anomaly_Model.h5`: Complete trained model
- `HDFS_Anomaly_weights.h5`: Model weights only
- `Label_encoder.joblib`: Event label encoder
- `attention_layer.pkl`: Attention layer weights
- `attention_layer_code.py`: AttentionLayer class definition
- `Drain_state.pkl`: Drain3 parser state
- `drain3_config.ini`: Drain3 configuration
- `model_config.json`: Complete model configuration
- `training_history.pkl`: Training history data
- `HOW_TO_LOAD.md`: This file

## For New Predictions:
1. Load the model using the code above
2. Preprocess new log sequences using the same label_encoder
3. Pad sequences to length {max_length}
4. Use model.predict() for anomaly detection

## For Retraining:
1. Load existing model and artifacts
2. Process new logs with the saved Drain3 parser
3. Encode new sequences with the saved label_encoder
4. Continue training with model.fit()
"""
    
    instructions_path = os.path.join(save_dir, "HOW_TO_LOAD.md")
    with open(instructions_path, 'w') as f:
        f.write(loading_instructions)
    print(f"✓ Loading instructions saved: {instructions_path}")
    
    print(f"\n{'='*50}")
    print("MODEL SAVING COMPLETE")
    print(f"{'='*50}")
    print(f"All artifacts saved to: {save_dir}")
    print(f"Model can be loaded using the instructions in: {instructions_path}")
    print("\nSaved components:")
    print("  ✓ Trained model with architecture")
    print("  ✓ Model weights (backup)")
    print("  ✓ Custom AttentionLayer (code + weights)")
    print("  ✓ Label encoder for event preprocessing")
    print("  ✓ Drain3 parser state for log parsing")
    print("  ✓ Complete configuration and metadata")
    print("  ✓ Training history for analysis")
    print("  ✓ Loading instructions and examples")

def main():
    """
    Main function that orchestrates the entire training process
    """
    print("Starting HDFS Anomaly Detection Model Training")
    print("="*60)
    
    # Display data information
    print(f"Total samples: {len(X_padded)}")
    print(f"Sequence length: {X_padded.shape[1]}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Anomaly ratio: {np.mean(y):.2%}")
    
    # Split data into training and testing sets
    # stratify=y ensures both sets have similar anomaly ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, 
        test_size=0.2,          # 20% for testing
        random_state=42,        # For reproducible results
        stratify=y              # Maintain class distribution
    )
    
    print(f"\nTraining data size: {X_train.shape}")
    print(f"Testing data size: {X_test.shape}")
    print(f"Training anomaly ratio: {np.mean(y_train):.2%}")
    print(f"Testing anomaly ratio: {np.mean(y_test):.2%}")
    
    # Create the model
    model, attention_layer = create_model(vocab_size, X_padded.shape[1])
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Train the model
    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,              # Process 32 samples at a time
        epochs=20,                  # Train for 20 complete passes through data
        validation_data=(X_test, y_test),  # Use test data for validation
        verbose=1,                  # Show progress
        shuffle=True                # Shuffle training data each epoch
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)
    
    # Save the complete model and training artifacts
    save_model_and_artifacts(model, history, LabelEnc, vocab_size, X_padded.shape[1], attention_layer)
    print("\nModel and all artifacts saved successfully!")
    
    # Show some example predictions
    print("\nExample Predictions:")
    print("-" * 40)
    for i in range(5):
        actual = "Anomaly" if y_test[i] == 1 else "Normal"
        predicted = "Anomaly" if y_pred[i] == 1 else "Normal"
        confidence = y_pred_prob[i][0]
        print(f"Sample {i+1}: Actual={actual}, Predicted={predicted}, Confidence={confidence:.3f}")
    
    print("\nTraining completed successfully!")
    return model, history

# Run the main function when script is executed
if __name__ == "__main__":
    model, history = main()