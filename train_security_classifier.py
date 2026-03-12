import os
import logging
import pandas as pd
from typing import Tuple
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "models/security_classifier"
INPUT_FILE = "processed/mcp_security_dataset.csv"

# Label mapping
LABEL_MAP = {
    "safe": 0,
    "suspicious": 1,
    "malicious": 2
}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads the dataset from the specified CSV file."""
    logger.info(f"Loading dataset from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset complete with {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def prepare_dataset(df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
    """Prepares the generic dataframe for HuggingFace Dataset conversion."""
    logger.info("Preparing dataset structure...")

    # Validate required columns
    if 'combined_text' not in df.columns or 'risk_label' not in df.columns:
        raise ValueError("Dataset must contain 'combined_text' and 'risk_label' columns.")

    # Drop rows with null values in required columns
    df = df.dropna(subset=['combined_text', 'risk_label']).copy()

    # Map text labels to integers, ignoring any unexpected labels
    df['label'] = df['risk_label'].str.lower().map(LABEL_MAP)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # Rename column for clarity in HF pipeline
    df = df.rename(columns={'combined_text': 'text'})
    
    # Select only required columns
    df = df[['text', 'label']]
    logger.info(f"Dataset size after cleaning: {len(df)} rows.")

    # Train/Validation Split (80/20)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    logger.info(f"Split into {len(train_df)} training and {len(val_df)} validation examples.")

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)

    return train_ds, val_ds


def tokenize_dataset(train_ds: Dataset, val_ds: Dataset, tokenizer) -> DatasetDict:
    """Tokenizes the text column using the defined tokenizer."""
    logger.info(f"Tokenizing dataset using `{MODEL_NAME}` tokenizer...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

    # Apply tokenization mapping
    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_val = val_ds.map(tokenize_function, batched=True)

    # Group into an HF DatasetDict
    dataset = DatasetDict({
        "train": tokenized_train,
        "validation": tokenized_val
    })
    
    return dataset


def compute_metrics(p: EvalPrediction):
    """Computes Evaluation metrics during Trainer validation steps."""
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_model(dataset: DatasetDict, tokenizer):
    """Initializes and trains the sequence classification model using HuggingFace Trainer."""
    logger.info(f"Initializing `{MODEL_NAME}` model for training...")
    
    # Load BERT Sequence Classifier
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=3,
        id2label=REVERSE_LABEL_MAP,
        label2id=LABEL_MAP
    )

    # Define Configuration
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=10,
        evaluation_strategy="epoch",  # evaluates at the end of each epoch
        save_strategy="epoch",        # saves a checkpoint per epoch
        load_best_model_at_end=True,  # loads best performing model at the end
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    logger.info("Executing model training...")
    trainer.train()

    return trainer


def evaluate_model(trainer: Trainer, val_dataset: Dataset):
    """Evaluates the best trained model on the validation dataset and prints sklearn metrics."""
    logger.info("Evaluating model on validation data...")
    predictions = trainer.predict(val_dataset)
    
    preds_flat = predictions.predictions.argmax(-1)
    labels_flat = predictions.label_ids

    print("\n" + "="*50)
    print("                 DETAILED EVALUATION")
    print("="*50)
    report = classification_report(
        labels_flat, 
        preds_flat, 
        target_names=["safe (0)", "suspicious (1)", "malicious (2)"],
        zero_division=0
    )
    print(report)
    print("="*50 + "\n")


def main():
    try:
        # Load Raw Pandas DF
        df = load_dataset(INPUT_FILE)

        # Prepare into HF Dataset representation
        train_ds, val_ds = prepare_dataset(df)

        # Initialize Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Tokenize the datasets
        tokenized_datasets = tokenize_dataset(train_ds, val_ds, tokenizer)

        # Train 
        trainer = train_model(tokenized_datasets, tokenizer)

        # Evaluate Validation Set
        evaluate_model(trainer, tokenized_datasets["validation"])

        # Final serialization of the model
        logger.info(f"Saving finalized model and tokenizer to {OUTPUT_DIR}...")
        trainer.save_model(OUTPUT_DIR) # Automatically saves both best model and tokenizer
        logger.info("Model saving complete. Training pipeline executed successfully.")

    except Exception as e:
        logger.critical(f"Training pipeline crashed: {e}")


if __name__ == "__main__":
    main()
