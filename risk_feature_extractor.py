import pandas as pd
import re
import os

# Define security feature categories and their corresponding regex keyword patterns.
# Using word boundaries (\b) to avoid partial matches (e.g., finding "db" inside "feedback").
RISK_PATTERNS = {
    'shell_access': r'\b(shell|bash|cmd|command line|terminal|powershell|exec|sh)\b',
    'file_access': r'\b(file|directory|folder|path|read|write|delete|remove|fs|filesystem)\b',
    'network_access': r'\b(http|https|request|api|webhook|network|fetch|curl|wget|tcp|udp|socket)\b',
    'database_access': r'\b(sql|database|query|db|select|insert|update|mysql|postgres|postgresql|mongodb|redis)\b',
    'process_control': r'\b(process|kill|spawn|task|pid|daemon|thread)\b',
    'code_execution': r'\b(eval|run|execute|dynamic|script)\b'
}

def load_dataset(filepath):
    """Load the dataset safely and handle missing text values."""
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    # Handle missing text values safely by filling with empty string
    df['combined_text'] = df['combined_text'].fillna('')
    return df

def extract_features(text, patterns):
    """
    Extract security capability features from text based on regex patterns.
    Returns a dictionary of binary features (1 if detected, 0 otherwise).
    """
    # Perform lowercase normalization
    text = str(text).lower()
    
    features = {}
    for feature_name, pattern in patterns.items():
        if re.search(pattern, text):
            features[feature_name] = 1
        else:
            features[feature_name] = 0
            
    return features

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, 'processed')
    
    input_csv = os.path.join(processed_dir, 'mcp_tools_dataset.csv')
    output_csv = os.path.join(processed_dir, 'mcp_risk_features.csv')
    
    # 1. Load the dataset
    df = load_dataset(input_csv)
    
    # Ensure there are no duplicates for strictness (using unique combined texts or server/tool pairs)
    df = df.drop_duplicates(subset=['server_name', 'tool_name']).reset_index(drop=True)
    
    print("Extracting security capability features...")
    
    # 2 & 3 & 4. Extract features for each row
    features_list = []
    for text in df['combined_text']:
        features_list.append(extract_features(text, RISK_PATTERNS))
        
    # 5. Add feature columns to the DataFrame
    features_df = pd.DataFrame(features_list)
    df = pd.concat([df, features_df], axis=1)
    
    # 6. Format output dataset structure
    output_columns = ['server_name', 'tool_name', 'combined_text'] + list(RISK_PATTERNS.keys())
    
    df_out = df[output_columns]
    
    # 8. Save the resulting dataset
    print(f"Saving risk features dataset to {output_csv}...")
    df_out.to_csv(output_csv, index=False)
    
    # 7. Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Total tools analyzed: {len(df_out)}")
    
    print("\nCount of tools with each capability:")
    for feature in RISK_PATTERNS.keys():
        count = df_out[feature].sum()
        print(f" - {feature}: {count}")
        
    print("\nSample preview of extracted features:")
    print(df_out.head())

if __name__ == "__main__":
    main()
