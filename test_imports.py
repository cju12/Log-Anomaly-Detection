"""
Test script to verify all imports work correctly
"""

print("Testing imports...")

try:
    print("1. Testing ParsingData import...")
    from datasets.ParsingData import ParsedHDFS
    print("   ✓ ParsingData import successful")
    
    print("2. Testing Preprocessing import...")
    from datasets.Preprocessing import initialize_data
    print("   ✓ Preprocessing import successful")
    
    print("3. Testing data initialization...")
    # This will actually run the preprocessing
    # X_padded, y, vocab_size, LabelEnc, grouped = initialize_data()
    # print(f"   ✓ Data initialized successfully")
    # print(f"   - Data shape: {X_padded.shape if X_padded is not None else 'Not initialized'}")
    # print(f"   - Vocab size: {vocab_size}")
    
    print("4. Testing Detection_Model imports...")
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Layer
    print("   ✓ TensorFlow imports successful")
    
    print("\n" + "="*50)
    print("ALL IMPORTS SUCCESSFUL! 🎉")
    print("="*50)
    print("\nYou can now run:")
    print("1. python Detection_Model.py  # To train the model")
    print("2. python predict_anomalies.py  # To make predictions")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure you're in the project root directory")
    print("2. Check that all required packages are installed:")
    print("   pip install tensorflow pandas numpy matplotlib seaborn scikit-learn drain3")
    print("3. Ensure the datasets/ directory contains ParsingData.py and Preprocessing.py")

except Exception as e:
    print(f"❌ Unexpected error: {e}")