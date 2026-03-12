import os
import time
import logging
import logging
import pandas as pd
from groq import Groq
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads the dataset from the specified CSV file."""
    logger.info(f"Loading dataset from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def label_tool_with_llm(client: Groq, combined_text: str) -> str:
    """Uses LLM to classify the security risk level of an MCP tool."""
    prompt = f"""You are a cybersecurity analyst evaluating MCP tools.

Classify the security risk level of the following MCP tool.

Tool description:
{combined_text}

Risk Categories:
safe → read-only APIs or harmless tools
suspicious → tools that access or modify files, databases, or network resources
malicious → tools that execute system commands, run code, or control processes

Return ONLY one word:
safe
suspicious
malicious"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        label = response.choices[0].message.content.strip().lower()
        
        # Strict validation
        if label not in ["safe", "suspicious", "malicious"]:
            logger.warning(f"Unexpected label received: '{label}'. Defaulting to suspicious.")
            return "suspicious"
            
        return label
    except Exception as e:
        logger.error(f"Error calling LLM APIs: {e}")
        return "error"

def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Iterates over the dataset and queries the LLM for risk labels."""
    
    # Check if API key is set
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not set.")
        raise ValueError("GROQ_API_KEY environment variable not set.")
        
    client = Groq(api_key=api_key)
    
    labels = []
    total_tools = len(df)
    
    logger.info("Starting to classify tools...")
    for idx, row in df.iterrows():
        text = str(row.get('combined_text', ''))
        
        if not text.strip() or text.lower() == 'nan':
            logger.warning(f"Row {idx} has an empty 'combined_text'. Skipping classification.")
            labels.append("error")
            continue
            
        # Call LLM for labeling
        label = label_tool_with_llm(client, text)
        labels.append(label)
        
        # Progress tracking
        if (idx + 1) % 50 == 0:
            logger.info(f"Progress: Processed {idx + 1}/{total_tools} tools.")
            
        # Rate limiting block - Groq free tier limit is 30 RPM, so wait 2.5s
        time.sleep(2.5)
        
    df['risk_label'] = labels
    return df

def save_results(df: pd.DataFrame, output_path: str):
    """Saves the final dataframe with risk labels to a CSV."""
    logger.info(f"Saving results to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Results saved successfully.")

def main():
    input_file = "processed/mcp_tools_dataset.csv"
    output_file = "processed/mcp_security_dataset.csv"
    
    try:
        # Load dataset
        df = load_dataset(input_file)
        
        if 'combined_text' not in df.columns:
            raise ValueError("The dataset does not contain a 'combined_text' column.")
            
        # Process and label tools
        df = process_dataset(df)
        
        # Save results
        save_results(df, output_file)
        
        # Summary
        total = len(df)
        safe_count = (df['risk_label'] == 'safe').sum()
        suspicious_count = (df['risk_label'] == 'suspicious').sum()
        malicious_count = (df['risk_label'] == 'malicious').sum()
        error_count = (df['risk_label'] == 'error').sum()
        
        print("\n" + "="*40)
        print("          PROCESSING SUMMARY          ")
        print("="*40)
        print(f"Total tools processed      : {total}")
        print(f"Number of safe tools       : {safe_count}")
        print(f"Number of suspicious tools : {suspicious_count}")
        print(f"Number of malicious tools  : {malicious_count}")
        print("="*40)
        
        if error_count > 0:
            print(f"WARNING: There were {error_count} tools that could not be processed due to errors or missing data.")
            
    except Exception as e:
        logger.critical(f"Process failed: {e}")

if __name__ == "__main__":
    main()
