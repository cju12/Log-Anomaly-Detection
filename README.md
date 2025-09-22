# HDFS Log Anomaly Detection

A deep learning system for detecting anomalies in Hadoop Distributed File System (HDFS) logs using LSTM with attention mechanism.

## 🎯 Project Overview

This project implements an end-to-end anomaly detection system for HDFS logs. It processes raw log files, extracts meaningful patterns, and trains a neural network to identify unusual behavior that might indicate system problems.

### What is this project about?

- **HDFS**: A system that stores large files across multiple computers
- **Log Files**: Records of what the system is doing (like a diary)
- **Anomaly Detection**: Finding unusual patterns that might mean something is wrong
- **Deep Learning**: Teaching a computer to recognize patterns like humans do

## 🏗️ Architecture

```
Raw HDFS Logs → Parsing → Preprocessing → Model Training → Anomaly Detection
```

1. **Parsing** (`ParsingData.py`): Converts raw logs into structured data
2. **Preprocessing** (`Preprocessing.py`): Prepares data for machine learning
3. **Model Training** (`Detection_Model.py`): Trains the neural network
4. **Prediction** (`predict_anomalies.py`): Uses trained model for detection

## 📊 Data Requirements

This project requires two main data files:

1. **HDFS.log**: Raw HDFS log file containing system events
2. **anomaly_label.csv**: Labels indicating which blocks are normal/anomalous

**Getting the Data:**
- Original HDFS dataset: [LogHub HDFS Dataset](https://github.com/logpai/loghub/tree/master/HDFS)
- Place `HDFS.log` in the `datasets/` directory
- Place `anomaly_label.csv` in the `datasets/` directory

## 📁 Project Structure

```
├── datasets/
│   ├── ParsingData.py          # Log parsing and grouping
│   ├── Preprocessing.py        # Data preprocessing
│   ├── HDFS.log               # Raw log file (input)
│   ├── anomaly_label.csv      # Labels for training
│   └── HDFS.log_structured.csv # Parsed logs (generated)
├── Detection_Model.py          # Main training script
├── predict_anomalies.py        # Prediction and analysis
├── drain3.ini                 # Configuration for log parsing
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn drain3
```

### Step 1: Prepare Your Data

**Required Data Files:**
- `datasets/HDFS.log` - Your raw HDFS log file
- `datasets/anomaly_label.csv` - Labels file with BlockId and Label columns

**Note:** The actual data files are not included in this repository due to size constraints. Sample files are provided:
- `datasets/sample_HDFS.log` - Shows expected log format
- `datasets/sample_anomaly_label.csv` - Shows expected label format

**Data Format:**
- HDFS.log: Raw log lines with timestamps, log levels, and messages
- anomaly_label.csv: CSV with columns `BlockId,Label` where Label is "Normal" or "Anomaly"

### Step 2: Train the Model

```bash
python Detection_Model.py
```

This will:
- Parse the raw logs
- Preprocess the data
- Train the LSTM-Attention model
- Save the trained model as `hdfs_anomaly_model.h5`
- Generate training plots and evaluation metrics

### Step 3: Make Predictions

```bash
python predict_anomalies.py
```

This will:
- Load the trained model
- Analyze sample predictions
- Show confidence distributions
- Demonstrate custom sequence prediction

## 🧠 Model Architecture

### LSTM with Attention Mechanism

```
Input Sequence → Embedding → LSTM → Attention → Dense → Prediction
```

**Components Explained:**

1. **Embedding Layer**: Converts event IDs to dense vectors
   - Think of it as creating a "dictionary" where each log event gets a unique number representation

2. **LSTM Layer**: Processes sequences of events
   - Like reading a sentence word by word and remembering the context

3. **Attention Layer**: Focuses on important events
   - Like highlighting the most important words in a sentence

4. **Dense Layer**: Makes final prediction
   - Decides: "Is this sequence normal or anomalous?"

### Why This Architecture?

- **LSTM**: Good at understanding sequences and temporal patterns
- **Attention**: Helps identify which specific events are most important
- **Binary Classification**: Simple output (normal vs anomaly)

## 📊 Data Flow

### 1. Raw Log Processing

```python
# Raw log line example:
"081109 203518 148 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906"

# After parsing:
{
    "LineId": 1,
    "BlockId": "blk_-1608999687919862906", 
    "EventId": "E5",
    "Template": "Receiving block <*>"
}
```

### 2. Sequence Creation

```python
# Group by BlockId to create sequences:
{
    "BlockId": "blk_123",
    "EventId": ["E1", "E5", "E2", "E1", "E3"],
    "Label": 0  # 0=normal, 1=anomaly
}
```

### 3. Numerical Encoding

```python
# Convert events to numbers:
["E1", "E5", "E2"] → [1, 5, 2]

# Pad to fixed length:
[1, 5, 2] → [1, 5, 2, 0, 0, 0, ..., 0]  # length 50
```

## 📈 Model Performance

The model tracks several metrics:

- **Accuracy**: Overall correctness
- **Precision**: Of predicted anomalies, how many were actually anomalies
- **Recall**: Of actual anomalies, how many did we catch
- **F1-Score**: Balance between precision and recall

### Interpreting Results

- **High Precision**: Few false alarms
- **High Recall**: Catches most real problems
- **High F1**: Good balance of both

## 🔧 Configuration

### Model Hyperparameters

```python
# In Detection_Model.py
embedding_dim = 64      # Size of event embeddings
lstm_units = 128        # Number of LSTM units
max_length = 50         # Maximum sequence length
batch_size = 32         # Training batch size
epochs = 20             # Training epochs
```

### Preprocessing Settings

```python
# In Preprocessing.py
max_length = 50         # Sequence padding length
test_size = 0.2         # 20% data for testing
```

## 📝 Understanding the Output

### Training Output

```
Epoch 1/20
Loss: 0.4523 - Accuracy: 0.8234 - Precision: 0.7891 - Recall: 0.8456
```

- **Loss**: How wrong the model is (lower is better)
- **Accuracy**: Percentage of correct predictions
- **Precision**: Quality of anomaly predictions
- **Recall**: Coverage of actual anomalies

### Prediction Output

```
Block: blk_123456
Prediction: Anomaly (confidence: 0.847)
```

- **Confidence > 0.8**: High confidence
- **Confidence 0.6-0.8**: Medium confidence  
- **Confidence < 0.6**: Low confidence

## 🛠️ Customization

### Adding New Features

1. **Different Sequence Lengths**: Modify `max_length` in preprocessing
2. **Model Architecture**: Add more layers in `create_model()`
3. **New Metrics**: Add custom metrics in model compilation

### Handling Different Data

1. **New Log Format**: Modify parsing logic in `ParsingData.py`
2. **Different Labels**: Update label processing in preprocessing
3. **Real-time Processing**: Adapt prediction script for streaming data

## 🐛 Troubleshooting

### Common Issues

1. **"extract_block_id not defined"**
   - Fixed: Function is now properly defined as static method

2. **"grouped variable not found"**
   - Fixed: Now properly exported from preprocessing module

3. **Memory Issues**
   - Reduce batch_size or max_length
   - Use smaller embedding dimensions

4. **Poor Performance**
   - Increase training epochs
   - Adjust learning rate
   - Add more training data

### Debug Tips

```python
# Check data shapes
print(f"X shape: {X_padded.shape}")
print(f"y shape: {y.shape}")

# Verify preprocessing
print(f"Vocab size: {vocab_size}")
print(f"Sample sequence: {X_padded[0]}")
```

## 📚 Next Steps

### Improvements to Consider

1. **Advanced Models**: Try Transformer architecture
2. **Feature Engineering**: Add temporal features, log severity levels
3. **Real-time Detection**: Implement streaming anomaly detection
4. **Explainability**: Add attention visualization
5. **Ensemble Methods**: Combine multiple models

### Production Deployment

1. **Model Serving**: Use TensorFlow Serving or Flask API
2. **Monitoring**: Track model performance over time
3. **Retraining**: Implement automated retraining pipeline
4. **Alerting**: Set up notifications for detected anomalies

## 🤝 Contributing

Feel free to improve this project by:
- Adding new features
- Improving documentation
- Fixing bugs
- Adding tests

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Anomaly Hunting! 🕵️‍♂️**