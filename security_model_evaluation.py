import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = "models/final_security_classifier"
TEST_FILE = "processed/test_security_dataset.csv"
OUTPUT_PREDICTIONS_FILE = "logs/model_predictions.csv"
OUTPUT_CONFUSION_MATRIX_FILE = "logs/confusion_matrix.png"

# Label mapping
LABEL_MAP = {
    "safe": 0,
    "suspicious": 1,
    "malicious": 2
}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads the test dataset and encodes labels."""
    logger.info(f"Loading test dataset from {file_path}")
    try:
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns.")
        
        # Drop NAs
        df = df.dropna(subset=['text', 'label']).copy()
        
        # Map text labels to integers validation
        df['true_label'] = df['label'].str.lower().map(LABEL_MAP)
        df = df.dropna(subset=['true_label'])
        df['true_label'] = df['true_label'].astype(int)
        
        logger.info(f"Loaded {len(df)} test samples successfully.")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def load_model(model_path: str):
    """Loads the trained BERT classifier using HuggingFace pipeline."""
    logger.info(f"Loading trained classifier from {model_path}...")
    try:
        classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, top_k=None) # top_k=None returns all scores
        logger.info("Model loaded successfully.")
        return classifier
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}. Error: {e}")
        raise


def run_predictions(classifier, df: pd.DataFrame) -> pd.DataFrame:
    """Runs inference on the dataset and stores predictions and confidences."""
    logger.info("Running predictions on test dataset...")
    
    predicted_labels = []
    confidence_scores = []
    predicted_text_labels = []
    
    texts = df['text'].tolist()
    
    # Process in batches or one-by-one, pipeline handles batching if properly configured but list comprehension is fine for smaller sets
    # Truncate texts to 512 tokens max roughly by character count for safety if tokenizer doesn't handle it cleanly via pipeline default
    for i, text in enumerate(texts):
        if i % 100 == 0 and i > 0:
            logger.info(f"Processed {i}/{len(texts)} samples...")
            
        try:
            # Run inference
            results = classifier(text[:2000])[0] # Get all scores, first item in batch out
            
            # Find the highest scoring label
            best_score = -1.0
            best_label_id = None
            
            for res in results:
                if res['score'] > best_score:
                    best_score = res['score']
                    best_label_id = res['label']
            
            # Map Label ID back
            pred_text = "unknown"
            if best_label_id == "LABEL_0" or best_label_id == "safe":
                pred_text = "safe"
                pred_int = 0
            elif best_label_id == "LABEL_1" or best_label_id == "suspicious":
                pred_text = "suspicious"
                pred_int = 1
            elif best_label_id == "LABEL_2" or best_label_id == "malicious":
                pred_text = "malicious"
                pred_int = 2
            else:
                 # Fallback
                 pred_text = str(best_label_id).lower()
                 pred_int = LABEL_MAP.get(pred_text, -1)
            
            predicted_labels.append(pred_int)
            predicted_text_labels.append(pred_text)
            confidence_scores.append(round(float(best_score), 4))
            
        except Exception as e:
            logger.error(f"Inference failed on text sample {i}: {e}")
            predicted_labels.append(-1)
            predicted_text_labels.append("error")
            confidence_scores.append(0.0)

    df['predicted_label_int'] = predicted_labels
    df['predicted_label'] = predicted_text_labels
    df['confidence_score'] = confidence_scores
    
    # Filter out any errors for metric evaluation
    df = df[df['predicted_label_int'] != -1].copy()
    
    logger.info("Predictions completed.")
    return df


def compute_metrics(df: pd.DataFrame):
    """Computes and prints sklearn evaluation metrics."""
    logger.info("Computing evaluation metrics...")
    
    y_true = df['true_label']
    y_pred = df['predicted_label_int']
    
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "="*50)
    print("             MODEL EVALUATION REPORT")
    print("="*50)
    print(f"Total test samples : {len(df)}")
    print(f"Accuracy           : {acc:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1-score           : {f1:.4f}")
    print("="*50)
    
    print("\nDetailed Classification Report:")
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=["safe", "suspicious", "malicious"],
        zero_division=0
    )
    print(report)
    
    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, output_path: str):
    """Generates and saves a confusion matrix plot using seaborn."""
    logger.info("Generating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    target_names = ["safe", "suspicious", "malicious"]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    
    plt.title('Confusion Matrix - Security Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Confusion matrix saved to {output_path}")


def save_results(df: pd.DataFrame, output_path: str):
    """Saves the prediction dataset to CSV."""
    logger.info("Saving prediction results...")
    
    # Select columns for final output
    output_df = df[['text', 'label', 'predicted_label', 'confidence_score']].copy()
    output_df = output_df.rename(columns={'label': 'true_label'})
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")


def main():
    try:
        # 1-2. Load test dataset and encode labels
        df = load_dataset(TEST_FILE)
        
        # 3. Load trained model
        classifier = load_model(MODEL_DIR)
        
        # 4-6. Run predictions
        results_df = run_predictions(classifier, df)
        
        # 7. Compute metrics and 12. Print Report
        y_true, y_pred = compute_metrics(results_df)
        
        # 8-10. Generate and save confusion matrix
        plot_confusion_matrix(y_true, y_pred, OUTPUT_CONFUSION_MATRIX_FILE)
        
        # 11. Save predictions dataset
        save_results(results_df, OUTPUT_PREDICTIONS_FILE)
        
        logger.info("Evaluation pipeline executed successfully.")
        
    except Exception as e:
        logger.critical(f"Evaluation script crashed: {e}")

if __name__ == "__main__":
    main()
