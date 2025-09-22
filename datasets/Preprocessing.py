"""
Data Preprocessing for HDFS Anomaly Detection

This module handles the preprocessing of parsed HDFS log data for machine learning.

Steps performed:
1. Load parsed and grouped log data
2. Encode event sequences into numerical format
3. Pad sequences to uniform length
4. Prepare labels for binary classification

For beginners:
- Event sequences are like sentences made of log events
- We convert text events to numbers because ML models work with numbers
- Padding ensures all sequences have the same length (like making all sentences 50 words)
"""

from .ParsingData import ParsedHDFS
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess_hdfs_data(max_length=50):
    """
    Preprocess HDFS log data for anomaly detection model
    
    Parameters:
    - max_length: Maximum sequence length for padding
    
    Returns:
    - X_padded: Padded and encoded sequences
    - y: Binary labels (0=normal, 1=anomaly)
    - vocab_size: Number of unique event types
    - LabelEnc: Fitted label encoder for future use
    - grouped: Original grouped data
    """
    
    print("Starting data preprocessing...")
    print("="*40)
    
    # Step 1: Load parsed and grouped data
    print("1. Loading parsed HDFS data...")
    parser = ParsedHDFS()
    grouped = parser.get_grouped_data()
    
    print(f"   - Total blocks: {len(grouped)}")
    print(f"   - Normal blocks: {len(grouped[grouped['Label'] == 0])}")
    print(f"   - Anomaly blocks: {len(grouped[grouped['Label'] == 1])}")
    
    # Step 2: Collect all unique events for encoding
    print("\n2. Collecting all unique events...")
    all_events = []
    for seq in grouped["EventId"]:
        all_events.extend(seq)  # Add all events from this sequence
    
    unique_events = set(all_events)
    print(f"   - Total events in dataset: {len(all_events)}")
    print(f"   - Unique event types: {len(unique_events)}")
    
    # Step 3: Create and fit label encoder
    print("\n3. Creating event encoder...")
    LabelEnc = LabelEncoder()
    LabelEnc.fit(list(unique_events))
    
    # Show some example event mappings
    print("   - Example event mappings:")
    for i, event in enumerate(list(unique_events)[:5]):
        encoded = LabelEnc.transform([event])[0]
        print(f"     '{event}' -> {encoded}")
    
    # Step 4: Encode sequences
    print("\n4. Encoding event sequences...")
    X_encoded = []
    for seq in grouped["EventId"]:
        encoded_seq = LabelEnc.transform(seq)
        X_encoded.append(encoded_seq)
    
    # Step 5: Pad sequences to uniform length
    print(f"\n5. Padding sequences to length {max_length}...")
    X_padded = pad_sequences(
        X_encoded, 
        maxlen=max_length,      # Maximum length
        padding="post",         # Add zeros at the end
        truncating="post",      # Cut from the end if too long
        value=0                 # Padding value
    )
    
    # Show sequence length statistics
    seq_lengths = [len(seq) for seq in X_encoded]
    print(f"   - Original sequence lengths:")
    print(f"     Min: {min(seq_lengths)}, Max: {max(seq_lengths)}")
    print(f"     Mean: {np.mean(seq_lengths):.1f}, Median: {np.median(seq_lengths):.1f}")
    print(f"   - Sequences longer than {max_length}: {sum(1 for l in seq_lengths if l > max_length)}")
    
    # Step 6: Prepare labels
    print("\n6. Preparing labels...")
    y = grouped["Label"].values
    
    # Final summary
    print("\n" + "="*40)
    print("PREPROCESSING COMPLETE")
    print("="*40)
    print(f"Input shape (X_padded): {X_padded.shape}")
    print(f"Labels shape (y): {y.shape}")
    print(f"Vocabulary size: {len(LabelEnc.classes_)}")
    print(f"Sample sequence: {X_padded[0][:10]}... (first 10 elements)")
    print(f"Sample label: {y[0]} ({'Anomaly' if y[0] == 1 else 'Normal'})")
    
    return X_padded, y, len(LabelEnc.classes_), LabelEnc, grouped

# Initialize variables as None - they will be set when preprocess_hdfs_data() is called
X_padded = None
y = None
vocab_size = None
LabelEnc = None
grouped = None

def initialize_data():
    """Initialize the preprocessed data if not already done"""
    global X_padded, y, vocab_size, LabelEnc, grouped
    
    if X_padded is None:
        print("Initializing preprocessed data...")
        X_padded, y, vocab_size, LabelEnc, grouped = preprocess_hdfs_data()
    
    return X_padded, y, vocab_size, LabelEnc, grouped

# Make variables and functions available for import
__all__ = ['X_padded', 'y', 'vocab_size', 'LabelEnc', 'grouped', 'preprocess_hdfs_data', 'initialize_data']