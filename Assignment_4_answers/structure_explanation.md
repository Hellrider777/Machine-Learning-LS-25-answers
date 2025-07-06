# Week 4 Assignment: Transformer-Based Next Word Predictor
## Complete Implementation Structure & Explanation

###  **Assignment Overview**
This implementation creates a complete next-word prediction system using GPT-2, fulfilling all requirements of the Week 4 wrap-up project. The code trains a transformer model, evaluates it using multiple metrics, and provides interactive testing capabilities.

---

##  **Code Structure Breakdown**

### **1. Initial Setup & Imports (Lines 1-15)**
```python
import os, math, torch, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
```

**Purpose**: Imports all necessary libraries for:
- **Data handling**: `datasets` for WikiText-2
- **Model components**: `transformers` for GPT-2 and training utilities
- **Evaluation**: `torch` for tensor operations and accuracy calculations

### **2. Dataset Loading & Preprocessing (Lines 17-30)**
```python
# Load WikiText-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Setup tokenizer with padding
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
```

**Key Features**:
- Uses WikiText-2 as specified in assignment
- Handles GPT-2's missing pad token issue
- Truncates sequences to 128 tokens for efficiency
- Applies consistent preprocessing across all text

### **3. Model Setup & Training Configuration (Lines 32-55)**
```python
# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Load pre-trained GPT-2
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_nextword",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    # ... other hyperparameters
)
```

**Design Decisions**:
- **Small batch size (4)**: Balances memory usage with training stability
- **1 epoch**: Quick training for demonstration (can be increased)
- **Evaluation per epoch**: Monitors training progress
- **mlm=False**: Enables causal language modeling (next-word prediction)

### **4. Training Execution (Lines 57-65)**
```python
# Create trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
```

**What Happens**:
- Fine-tunes GPT-2 on WikiText-2 data
- Saves checkpoints during training
- Logs training metrics (loss, learning rate, etc.)
- Automatically handles gradient updates and optimization

---

##  **Evaluation Components**

### **5. Perplexity Calculation (Lines 67-70)**
```python
eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"])
print(f"Final Perplexity: {perplexity:.2f}")
```

**Metric Explanation**:
- **Perplexity**: Measures how "surprised" the model is by the test data
- **Lower is better**: Perfect model would have perplexity of 1
- **Typical range**: 20-200 for language models
- **Calculation**: e^(cross_entropy_loss)

### **6. Top-k Accuracy Implementation (Lines 72-130)**
```python
def calculate_top_k_accuracy(model, tokenizer, eval_dataset, k_values=[1, 5, 10], max_samples=1000):
    # Detailed accuracy calculation for k=1, 5, 10
    # Handles padding tokens correctly
    # Processes batches efficiently
    return final_accuracies, total_predictions
```

**Advanced Features**:
- **Multiple k values**: Tests top-1, top-5, and top-10 predictions
- **Batch processing**: Efficient evaluation on large datasets
- **Padding awareness**: Skips invalid tokens in accuracy calculation
- **Progress tracking**: Shows evaluation progress for long runs

**Expected Results**:
- **Top-1 accuracy**: Usually 15-25% (hardest metric)
- **Top-5 accuracy**: Usually 35-50% (more forgiving)
- **Top-10 accuracy**: Usually 45-60% (most forgiving)

---

##  **Interactive Features**

### **7. Next Word Prediction Function (Lines 150-170)**
```python
def predict_next_words(model, tokenizer, text, num_predictions=5, temperature=0.8):
    # Tokenizes input text
    # Generates predictions with temperature control
    # Returns top predictions with probabilities
```

**Key Parameters**:
- **temperature**: Controls randomness (0.8 = slightly creative, 1.0 = normal, 0.1 = very focused)
- **num_predictions**: How many top predictions to return
- **Probability scores**: Shows confidence in each prediction

### **8. Interactive Testing Loop (Lines 172-200)**
```python
def interactive_prediction(model, tokenizer):
    # Command-line interface for testing
    # Handles user input and error cases
    # Provides example prompts for testing
```

**User Experience**:
- Type any text and get next-word predictions
- See probability scores for each prediction
- Easy exit with 'quit' or 'exit'
- Helpful examples provided

### **9. Automated Examples (Lines 202-220)**
```python
example_texts = [
    "The weather today is",
    "Machine learning is", 
    "The quick brown fox",
    # ... more examples
]
```

**Purpose**: Demonstrates model capabilities without user interaction

---

##  **Expected Output Flow**

### **Training Phase**
```
Loading dataset...
Tokenizing data...
Training: [====>   ] 50%   Loss: 3.245
Training: [========] 100%  Loss: 2.891
```

### **Evaluation Phase**
```
Final Perplexity: 45.23

==================================================
CALCULATING TOP-K ACCURACY
==================================================
Processing batch 0/62
Processing batch 50/62
Top-k Accuracy Results (evaluated on 4,856 predictions):
Top-1 Accuracy: 0.2156 (21.56%)
Top-5 Accuracy: 0.4234 (42.34%)
Top-10 Accuracy: 0.5412 (54.12%)
```

### **Example Predictions**
```
==================================================
EXAMPLE PREDICTIONS
==================================================

Input: 'The weather today is'
Top 3 predictions:
  1. ' very' (prob: 0.1245)
  2. ' quite' (prob: 0.0987)
  3. ' really' (prob: 0.0823)
```

### **Interactive Mode**
```
==================================================
INTERACTIVE NEXT WORD PREDICTION
==================================================
Enter text to predict next words (or 'quit' to exit)

Enter text: The quick brown fox
Next word predictions for: 'The quick brown fox'
----------------------------------------
1. ' jumps' (probability: 0.2345)
2. ' jumped' (probability: 0.1567)
3. ' runs' (probability: 0.0987)
```

---

##  **Assignment Requirements Checklist**

###  **Completed Objectives**
- [x] **Build transformer language model** → GPT-2 fine-tuning
- [x] **Fine-tune pre-trained model** → GPT-2 on WikiText-2
- [x] **Evaluate with perplexity** → Comprehensive perplexity calculation
- [x] **Evaluate with top-k accuracy** → Top-1, 5, 10 accuracy metrics
- [x] **Tokenizer alignment** → Proper GPT-2 tokenizer setup
- [x] **Model adaptation** → Token embedding resizing
- [x] **Text preprocessing** → Consistent tokenization pipeline

###  **Technical Implementation**
- [x] **Data Loading**: WikiText-2 with `datasets` library
- [x] **Tokenization**: GPT-2 tokenizer with padding
- [x] **Model Selection**: GPT2LMHeadModel for causal LM
- [x] **Fine-tuning**: Trainer API with proper configuration
- [x] **Evaluation**: Both perplexity and top-k accuracy
- [x] **Interactive Testing**: Command-line prediction interface

---

##  **Running the Code**

### **Prerequisites**
```bash
pip install torch transformers datasets numpy
```

### **Execution**
```bash
python transformer.py
```

### **Expected Runtime**
- **Training**: 10-30 minutes (depends on hardware)
- **Evaluation**: 5-10 minutes
- **Interactive mode**: Until user exits

### **Hardware Requirements**
- **GPU**: Recommended (CUDA-capable)
- **RAM**: 8GB+ recommended
- **Storage**: 2GB for model and data

---

##  **Key Learning Points**

### **Transformer Architecture**
- Understanding causal attention in GPT-2
- How pre-trained models can be fine-tuned
- Importance of tokenizer-model alignment

### **Evaluation Metrics**
- **Perplexity**: Measures model uncertainty
- **Top-k accuracy**: Practical prediction quality
- **Trade-offs**: Accuracy vs. creativity in predictions

### **Practical Implementation**
- **Batch processing**: Efficient evaluation techniques
- **Memory management**: Handling large models
- **User interaction**: Making models accessible

---

##  **Extension Opportunities**

### **Model Improvements**
- Increase training epochs for better performance
- Experiment with different learning rates
- Try larger models (GPT-2 medium/large)

### **Dataset Variations**
- Use domain-specific datasets
- Combine multiple datasets
- Implement custom data preprocessing

### **Advanced Features**
- Beam search for better predictions
- Fine-tuning on specific writing styles
- Multi-language support

---

##  **Troubleshooting Tips**

### **Common Issues**
1. **CUDA out of memory**: Reduce batch size to 2 or 1
2. **Slow training**: Ensure GPU is being used
3. **Poor accuracy**: Increase training epochs
4. **Tokenizer errors**: Check pad_token setup

### **Performance Optimization**
- Use mixed precision training (`fp16=True`)
- Implement gradient accumulation
- Use DataLoader with multiple workers

---

This implementation provides a complete, production-ready next-word prediction system that demonstrates all key concepts from the 4-week NLP journey while meeting every assignment requirement. 