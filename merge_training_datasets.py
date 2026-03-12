import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths
DATASET_1_PATH = "processed/mcp_security_dataset.csv"
DATASET_2_PATH = "mcp_guard_training_dataset.csv"

FINAL_DATASET_PATH = "processed/final_security_training_dataset.csv"
TRAIN_DATASET_PATH = "processed/train_security_dataset.csv"
TEST_DATASET_PATH = "processed/test_security_dataset.csv"

def load_datasets():
    """Loads both original datasets."""
    logger.info("Loading datasets...")
    try:
        df1 = pd.read_csv(DATASET_1_PATH)
        logger.info(f"Loaded Dataset 1 ({DATASET_1_PATH}) with {len(df1)} rows.")
    except Exception as e:
        logger.error(f"Failed to load Dataset 1: {e}")
        raise

    try:
        df2 = pd.read_csv(DATASET_2_PATH)
        logger.info(f"Loaded Dataset 2 ({DATASET_2_PATH}) with {len(df2)} rows.")
    except Exception as e:
        logger.error(f"Failed to load Dataset 2: {e}")
        raise

    return df1, df2

def preprocess_dataset1(df1: pd.DataFrame) -> pd.DataFrame:
    """Extracts text and label from Dataset 1."""
    logger.info("Preprocessing Dataset 1...")
    df = df1[['combined_text', 'risk_label']].copy()
    df.rename(columns={'combined_text': 'text', 'risk_label': 'label'}, inplace=True)
    return df

def preprocess_dataset2(df2: pd.DataFrame) -> pd.DataFrame:
    """Combines text fields and maps labels from Dataset 2."""
    logger.info("Preprocessing Dataset 2...")
    df = pd.DataFrame()
    
    # Fill NA to prevent concatenation issues
    prompt = df2['prompt'].fillna('')
    tool_name = df2['tool_name'].fillna('')
    arguments = df2['arguments'].fillna('')

    # Combine text
    df['text'] = prompt.astype(str) + " " + tool_name.astype(str) + " " + arguments.astype(str)
    
    # Map labels: 0 -> safe, 1 -> malicious
    label_mapping = {0: 'safe', 1: 'malicious'}
    df['label'] = df2['label'].map(label_mapping)
    
    return df

def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Concatenates, cleans, and shuffles both datasets."""
    logger.info("Merging datasets...")
    
    # Combine the dataframes
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    logger.info(f"Rows before cleaning: {len(merged_df)}")
    
    # Remove empty rows
    merged_df.dropna(subset=['text', 'label'], inplace=True)
    merged_df = merged_df[merged_df['text'].str.strip() != '']
    
    # Remove duplicates
    merged_df.drop_duplicates(inplace=True)
    
    # Shuffle randomly
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Rows after cleaning: {len(merged_df)}")
    
    return merged_df

def print_summary_statistics(df: pd.DataFrame):
    """Prints total sample count and label distribution."""
    total = len(df)
    safe_count = (df['label'] == 'safe').sum()
    susp_count = (df['label'] == 'suspicious').sum()
    mal_count = (df['label'] == 'malicious').sum()
    
    print("\n" + "="*50)
    print("             DATASET SUMMARY STATISTICS")
    print("="*50)
    print(f"Total samples                  : {total}")
    print(f"Number of safe samples         : {safe_count}")
    print(f"Number of suspicious samples   : {susp_count}")
    print(f"Number of malicious samples    : {mal_count}")
    print("="*50 + "\n")

def split_dataset(df: pd.DataFrame):
    """Splits the merged dataset into 80% train and 20% test."""
    logger.info("Splitting dataset into 80% train and 20% test...")
    # Using stratify to keep label proportions similar across splits
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    logger.info(f"Train set size: {len(train_df)}")
    logger.info(f"Test set size: {len(test_df)}")
    
    return train_df, test_df

def save_datasets(final_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Saves all datasets to disk."""
    logger.info(f"Saving final dataset to {FINAL_DATASET_PATH}...")
    final_df.to_csv(FINAL_DATASET_PATH, index=False)
    
    logger.info(f"Saving train split to {TRAIN_DATASET_PATH}...")
    train_df.to_csv(TRAIN_DATASET_PATH, index=False)
    
    logger.info(f"Saving test split to {TEST_DATASET_PATH}...")
    test_df.to_csv(TEST_DATASET_PATH, index=False)
    logger.info("All datasets successfully saved.")

def main():
    try:
        # 1-3. Load datasets
        df1, df2 = load_datasets()
        
        # 2-6. Preprocess individually
        clean_df1 = preprocess_dataset1(df1)
        clean_df2 = preprocess_dataset2(df2)
        
        # 7-9. Merge, clean, and shuffle
        merged_df = merge_datasets(clean_df1, clean_df2)
        
        # 10. Print statistics
        print_summary_statistics(merged_df)
        
        # 12. Split dataset
        train_df, test_df = split_dataset(merged_df)
        
        # 11-12. Save results
        save_datasets(merged_df, train_df, test_df)

    except Exception as e:
        logger.critical(f"Dataset merging pipeline failed: {e}")

if __name__ == "__main__":
    main()
