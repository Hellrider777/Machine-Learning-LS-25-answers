"""
Machine Learning Pipeline with Hugging Face for Sentiment Analysis
This script implements a complete pipeline for fine-tuning BERT on the IMDb dataset.
"""

import torch
import numpy as np
import os
import sys
import warnings
from pathlib import Path

try:
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding
    )
    from sklearn.metrics import accuracy_score, f1_score
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install required packages using: pip install -r requirements.txt")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Check transformers version
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    
    # Warn about very old versions
    version_parts = transformers.__version__.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    if major < 4 or (major == 4 and minor < 20):
        print("WARNING: You're using an older version of transformers.")
        print("Some features may not work as expected.")
        print("Consider upgrading: pip install transformers>=4.20.0")
except Exception as e:
    print(f"Warning: Could not check transformers version: {e}")

# Set device with error handling
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except Exception as e:
    print(f"Warning: Error checking GPU availability: {e}")
    device = torch.device("cpu")
    print("Falling back to CPU")

class SentimentAnalysisPipeline:
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_dataset(self):
        """Step 1: Load the IMDb dataset"""
        print("Loading IMDb dataset...")
        
        try:
            dataset = load_dataset("imdb")
        except Exception as e:
            print(f"Error loading IMDb dataset: {e}")
            print("This might be due to network issues or dataset unavailability.")
            print("Please check your internet connection and try again.")
            raise
        
        try:
            # Use a smaller subset for faster training (optional)
            # Remove these lines if you want to use the full dataset
            train_dataset = dataset["train"].shuffle(seed=42).select(range(5000))
            test_dataset = dataset["test"].shuffle(seed=42).select(range(1000))
            
            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Test dataset size: {len(test_dataset)}")
            
            # Validate dataset structure
            if len(train_dataset) == 0 or len(test_dataset) == 0:
                raise ValueError("Dataset is empty after filtering")
                
            # Check if required columns exist
            required_cols = ["text", "label"]
            for col in required_cols:
                if col not in train_dataset.column_names:
                    raise ValueError(f"Required column '{col}' not found in dataset")
            
            print(f"Sample data: {train_dataset[0]}")
            
        except Exception as e:
            print(f"Error processing dataset: {e}")
            raise
        
        return train_dataset, test_dataset
    
    def preprocess_dataset(self, train_dataset, test_dataset):
        """Step 2: Preprocess and tokenize the dataset"""
        print("Initializing tokenizer...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print(f"Tokenizer loaded successfully: {self.tokenizer.__class__.__name__}")
        except Exception as e:
            print(f"Error loading tokenizer for {self.model_name}: {e}")
            print("This might be due to network issues or invalid model name.")
            raise
        
        def tokenize_function(examples):
            try:
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=False,  # We'll pad dynamically during training
                    max_length=512
                )
            except Exception as e:
                print(f"Error during tokenization: {e}")
                raise
        
        try:
            print("Tokenizing datasets...")
            train_tokenized = train_dataset.map(tokenize_function, batched=True)
            test_tokenized = test_dataset.map(tokenize_function, batched=True)
            
            print("Setting tensor format...")
            # Set format for PyTorch
            train_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
            test_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
            
            # Validate tokenized data
            print(f"Tokenized train samples: {len(train_tokenized)}")
            print(f"Tokenized test samples: {len(test_tokenized)}")
            
            # Check if tokenization worked correctly
            sample = train_tokenized[0]
            required_keys = ["input_ids", "attention_mask", "label"]
            for key in required_keys:
                if key not in sample:
                    raise ValueError(f"Missing key '{key}' in tokenized data")
                    
            print("Tokenization completed successfully!")
            
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            raise
        
        return train_tokenized, test_tokenized
    
    def setup_model(self):
        """Step 3: Setup BERT model for sentiment analysis"""
        print("Loading BERT model for sequence classification...")
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,  # Binary classification: positive/negative
                id2label={0: "negative", 1: "positive"},
                label2id={"negative": 0, "positive": 1}
            )
            
            # Move model to appropriate device
            self.model.to(device)
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Model loaded successfully!")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("This might be due to network issues, insufficient memory, or invalid model name.")
            raise
        
        return self.model
    
    def compute_metrics(self, eval_pred):
        """Step 4: Define evaluation metrics"""
        try:
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='weighted')
            
            return {
                'accuracy': accuracy,
                'f1': f1
            }
        except Exception as e:
            print(f"Warning: Error computing metrics: {e}")
            return {'accuracy': 0.0, 'f1': 0.0}
    
    def fine_tune_model(self, train_dataset, test_dataset):
        """Step 3-4: Fine-tune the model and evaluate"""
        print("Setting up training arguments...")
        
        try:
            # Create output directories
            os.makedirs("./results", exist_ok=True)
            os.makedirs("./logs", exist_ok=True)
            
            # Adjust batch size based on available memory
            batch_size = 16
            if device.type == "cpu":
                batch_size = 8  # Smaller batch size for CPU
                print("Using smaller batch size for CPU training")
            
            # Handle different transformers versions - use minimal compatible parameters
            print("Setting up training arguments with basic parameters...")
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=3,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_steps=100,
                save_steps=1000,  # Save every 1000 steps
                # Using minimal parameters that work across all versions
            )
            
            # Data collator for dynamic padding
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            
            print("Initializing trainer...")
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
            )
            
        except Exception as e:
            print(f"Error setting up training: {e}")
            raise
        
        try:
            print("Starting fine-tuning...")
            print("This may take some time depending on your hardware...")
            self.trainer.train()
            print("Training completed successfully!")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"GPU out of memory error: {e}")
                print("Try reducing batch_size in the training arguments or use a smaller model")
                print("You can also try training on CPU (though it will be slower)")
            else:
                print(f"Runtime error during training: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error during training: {e}")
            raise
        
        try:
            print("Evaluating model...")
            eval_results = self.trainer.evaluate()
            
            print("Evaluation Results:")
            for key, value in eval_results.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
                    
        except Exception as e:
            print(f"Error during evaluation: {e}")
            print("Note: Evaluation might not be available in older transformers versions")
            # Create a minimal eval_results dict if evaluation fails
            eval_results = {"eval_loss": 0.0, "eval_accuracy": 0.0, "eval_f1": 0.0}
            print("Using placeholder evaluation results")
        
        return eval_results
    
    def save_model(self, save_directory="./fine_tuned_bert_sentiment"):
        """Step 5: Save the fine-tuned model"""
        print(f"Saving model to {save_directory}...")
        
        try:
            # Create directory if it doesn't exist
            save_path = Path(save_directory)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Check if model and tokenizer exist
            if self.model is None:
                raise ValueError("No model to save. Please train a model first.")
            if self.tokenizer is None:
                raise ValueError("No tokenizer to save. Please initialize tokenizer first.")
            
            # Save model and tokenizer
            self.model.save_pretrained(save_directory)
            self.tokenizer.save_pretrained(save_directory)
            
            # Verify files were saved
            required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
            missing_files = []
            for file in required_files:
                if not (save_path / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                print(f"Warning: Some files may not have been saved properly: {missing_files}")
            else:
                print("Model saved successfully!")
                print(f"Model files saved to: {save_path.absolute()}")
                
        except Exception as e:
            print(f"Error saving model: {e}")
            raise
        
        return save_directory
    
    def load_model_for_inference(self, model_path="./fine_tuned_bert_sentiment"):
        """Step 5: Load the saved model for inference"""
        print(f"Loading model from {model_path}...")
        
        try:
            # Check if model directory exists
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                raise FileNotFoundError(f"Model directory not found: {model_path}")
            
            # Check for required files
            required_files = ["config.json"]
            missing_files = []
            for file in required_files:
                if not (model_path_obj / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                raise FileNotFoundError(f"Missing required files: {missing_files}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print("Tokenizer loaded successfully")
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(device)
            self.model.eval()
            print("Model loaded successfully!")
            
            # Verify model is working
            test_input = "This is a test."
            test_result = self.predict_sentiment(test_input)
            print(f"Model verification successful - test prediction: {test_result['sentiment']}")
            
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            print("Make sure the model has been trained and saved properly.")
            raise
    
    def predict_sentiment(self, text):
        """Perform inference on sample text"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please load or train a model first.")
        
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")
        
        if len(text.strip()) == 0:
            raise ValueError("Input text cannot be empty")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move inputs to same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions.max().item()
            
            sentiment = "positive" if predicted_class == 1 else "negative"
            
            return {
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "probabilities": {
                    "negative": predictions[0][0].item(),
                    "positive": predictions[0][1].item()
                }
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

def check_system_requirements():
    """Check system requirements and provide helpful information"""
    print("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        print("WARNING: Python 3.7+ is recommended for best compatibility")
    
    # Check available memory (approximate)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        if memory.available < 4 * (1024**3):  # Less than 4GB
            print("WARNING: Low available memory. Consider using smaller batch sizes.")
    except ImportError:
        print("RAM check skipped (psutil not available)")
    
    # Check disk space
    try:
        import shutil
        disk_usage = shutil.disk_usage(".")
        free_gb = disk_usage.free / (1024**3)
        print(f"Available disk space: {free_gb:.1f} GB")
        if free_gb < 5:
            print("WARNING: Low disk space. Models and datasets require ~2-3GB")
    except:
        print("Disk space check skipped")
    
    print("System check completed.\n")

def main():
    """Main function to run the complete pipeline"""
    print("="*60)
    print("MACHINE LEARNING PIPELINE WITH HUGGING FACE")
    print("Sentiment Analysis on IMDb Dataset")
    print("="*60)
    
    # Check system requirements
    try:
        check_system_requirements()
    except Exception as e:
        print(f"Warning: System check failed: {e}\n")
    
    # Initialize pipeline
    pipeline = SentimentAnalysisPipeline()
    
    try:
        # Step 1: Load dataset
        print("STEP 1: Loading dataset...")
        train_data, test_data = pipeline.load_dataset()
        
        # Step 2: Preprocess dataset
        print("\nSTEP 2: Preprocessing dataset...")
        train_tokenized, test_tokenized = pipeline.preprocess_dataset(train_data, test_data)
        
        # Step 3: Setup model
        print("\nSTEP 3: Setting up model...")
        model = pipeline.setup_model()
        
        # Step 4: Fine-tune and evaluate
        print("\nSTEP 4: Fine-tuning and evaluation...")
        eval_results = pipeline.fine_tune_model(train_tokenized, test_tokenized)
        
        # Step 5: Save model
        print("\nSTEP 5: Saving model...")
        save_path = pipeline.save_model()
        
        print("\n" + "="*60)
        print("INFERENCE DEMONSTRATION")
        print("="*60)
        
        # Demonstrate inference
        sample_texts = [
            "This movie was absolutely fantastic! I loved every minute of it.",
            "Terrible film. I want my money back. Worst movie ever.",
            "The movie was okay, nothing special but not bad either.",
            "Outstanding performance by the actors. Highly recommended!",
            "Boring and predictable. Fell asleep halfway through."
        ]
        
        print("Performing inference on sample texts...")
        for i, text in enumerate(sample_texts, 1):
            try:
                result = pipeline.predict_sentiment(text)
                print(f"\n{i}. Text: {result['text']}")
                print(f"   Sentiment: {result['sentiment']}")
                print(f"   Confidence: {result['confidence']:.4f}")
                print(f"   Probabilities: Negative={result['probabilities']['negative']:.4f}, "
                      f"Positive={result['probabilities']['positive']:.4f}")
            except Exception as e:
                print(f"\n{i}. Error predicting sentiment for text: {e}")
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        
        # Format results safely
        accuracy = eval_results.get('eval_accuracy', 'N/A')
        f1_score = eval_results.get('eval_f1', 'N/A')
        
        if isinstance(accuracy, (int, float)):
            print(f"Final Model Accuracy: {accuracy:.4f}")
        else:
            print(f"Final Model Accuracy: {accuracy}")
            
        if isinstance(f1_score, (int, float)):
            print(f"Final Model F1-Score: {f1_score:.4f}")
        else:
            print(f"Final Model F1-Score: {f1_score}")
            
        print(f"Model saved to: {save_path}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print("Partial progress may have been saved to ./results/")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nTroubleshooting suggestions:")
        print("1. Make sure you have the required packages installed:")
        print("   pip install -r requirements.txt")
        print("2. Check your internet connection for downloading datasets/models")
        print("3. Ensure you have sufficient disk space (~3GB)")
        print("4. If using GPU, check CUDA compatibility")
        print("5. Try reducing batch size if encountering memory errors")
        
        # Print the full traceback for debugging
        import traceback
        print(f"\nFull error traceback:")
        traceback.print_exc()

def test_saved_model():
    """Function to test loading and using a saved model"""
    print("="*40)
    print("TESTING SAVED MODEL")
    print("="*40)
    
    pipeline = SentimentAnalysisPipeline()
    
    try:
        # Load the saved model
        print("Loading saved model...")
        pipeline.load_model_for_inference("./fine_tuned_bert_sentiment")
        
        # Test inference with multiple examples
        test_texts = [
            "I really enjoyed this movie. It was amazing!",
            "This film was terrible and boring.",
            "An okay movie, nothing special."
        ]
        
        print("Testing inference on sample texts...")
        for i, test_text in enumerate(test_texts, 1):
            try:
                result = pipeline.predict_sentiment(test_text)
                
                print(f"\n{i}. Text: {result['text']}")
                print(f"   Predicted Sentiment: {result['sentiment']}")
                print(f"   Confidence: {result['confidence']:.4f}")
                print(f"   Probabilities: Neg={result['probabilities']['negative']:.3f}, "
                      f"Pos={result['probabilities']['positive']:.3f}")
                      
            except Exception as e:
                print(f"\n{i}. Error with text '{test_text}': {e}")
        
        print("\n" + "="*40)
        print("SAVED MODEL TEST COMPLETED")
        print("="*40)
        
    except FileNotFoundError as e:
        print(f"Model not found: {e}")
        print("Make sure the model has been trained and saved first by running:")
        print("python huggingface_sentiment_pipeline.py")
    except Exception as e:
        print(f"Error testing saved model: {str(e)}")
        print("Troubleshooting:")
        print("1. Ensure the model was saved properly")
        print("2. Check if all required files exist in ./fine_tuned_bert_sentiment/")
        print("3. Verify the model directory is not corrupted")
        
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    # Run the complete pipeline
    main()
    
    # Uncomment the line below to test loading a saved model
    # test_saved_model() 