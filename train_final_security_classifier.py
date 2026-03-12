import os
import logging
import pandas as pd
from typing import Tuple
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
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
OUTPUT_DIR = "models/final_security_classifier"
TRAIN_FILE = "processed/train_security_dataset.csv"
TEST_FILE = "processed/test_security_dataset.csv"

# Label mapping
LABEL_MAP = {
    "safe": 0,
    "suspicious": 1,
    "malicious": 2
}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the training and testing datasets from CSV files."""
    logger.info("Loading training and testing datasets...")
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        
        logger.info(f"Loaded {len(train_df)} training samples and {len(test_df)} testing samples.")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes string labels into mapped integers."""
    # Ensure 'label' column exists
    if 'label' not in df.columns or 'text' not in df.columns:
        raise ValueError("DataFrames must contain 'text' and 'label' columns.")
        
    # Drop NAs
    df = df.dropna(subset=['text', 'label']).copy()
    
    # Map text labels to ints
    df['label'] = df['label'].str.lower().map(LABEL_MAP)
    
    # Drop any unmapped labels
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    return df[['text', 'label']]


def prepare_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
    """Encodes labels and prepares the datasets for HuggingFace Dataset conversion."""
    logger.info("Encoding labels and preparing dataset structure...")

    train_df = encode_labels(train_df)
    test_df = encode_labels(test_df)

    logger.info(f"Final Train size: {len(train_df)} | Final Test size: {len(test_df)}.")

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)

    return train_ds, test_ds


def tokenize_dataset(train_ds: Dataset, test_ds: Dataset, tokenizer) -> DatasetDict:
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
    tokenized_test = test_ds.map(tokenize_function, batched=True)

    # Group into an HF DatasetDict
    dataset = DatasetDict({
        "train": tokenized_train,
        "test": tokenized_test
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
        num_train_epochs=6,
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
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    logger.info("Executing model training workflow...")
    trainer.train()

    return trainer


def evaluate_model(trainer: Trainer, test_dataset: Dataset):
    """Evaluates the best trained model on the test dataset and prints sklearn metrics."""
    logger.info("Evaluating final model on test data...")
    predictions = trainer.predict(test_dataset)
    
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
        # 1-3. Load Datasets
        train_df, test_df = load_datasets()

        # 4-5. Prepare into HF Dataset representation (includes label encoding)
        train_ds, test_ds = prepare_dataset(train_df, test_df)

        # 6. Initialize Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # 7. Tokenize the datasets
        tokenized_datasets = tokenize_dataset(train_ds, test_ds, tokenizer)

        # 8-11. Train 
        trainer = train_model(tokenized_datasets, tokenizer)

        # 12-13. Evaluate validation set
        evaluate_model(trainer, tokenized_datasets["test"])

        # 14. Final serialization of the model
        logger.info(f"Saving finalized model and tokenizer to {OUTPUT_DIR}...")
        trainer.save_model(OUTPUT_DIR) # Automatically saves both best model and tokenizer
        logger.info("Model saving complete. Training pipeline executed successfully.")

    except Exception as e:
        logger.critical(f"Training pipeline crashed: {e}")


if __name__ == "__main__":
    main()
