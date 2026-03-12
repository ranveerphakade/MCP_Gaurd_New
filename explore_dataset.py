import pandas as pd
import json
import os
import numpy as np

def load_data(filepath):
    """Load JSON dataset into a pandas DataFrame."""
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def analyze_dataset(df, dataset_name, f_out):
    """Perform dataset analysis and write to summary file."""
    f_out.write(f"--- Analysis for {dataset_name} ---\n")
    f_out.write(f"Total number of records: {len(df)}\n")
    
    f_out.write(f"\nMetadata attributes (columns):\n")
    for col in df.columns:
        f_out.write(f" - {col}\n")
        
    f_out.write(f"\nMissing values count:\n")
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        f_out.write(f" - {col}: {count}\n")
        
    f_out.write(f"\nFirst 5 rows preview:\n")
    # Convert head(5) to string to easily write to file
    f_out.write(df.head(5).to_string() + "\n\n")

    # Identify fields useful for security analysis
    security_fields = ['name', 'description', 'tools', 'server_command', 'server_config', 'url', 'github']
    available_fields = [f for f in security_fields if f in df.columns]
    
    f_out.write(f"Fields useful for security analysis found: {available_fields}\n")
    for field in available_fields:
        non_null_count = df[field].notnull().sum()
        f_out.write(f" - {field}: {non_null_count} non-null entries\n")
        
        # Optionally, preview a few samples of tools or commands if present
        if field in ['tools', 'server_command', 'server_config'] and non_null_count > 0:
            samples = df[df[field].notnull()][field].head(2).tolist()
            f_out.write(f"   [Sample {field}]: {samples}\n")
    
    f_out.write("\n==========================================\n\n")

def main():
    print("Starting dataset exploration...")
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    servers_path = os.path.join(base_dir, 'Website', 'mcpso_servers_cleaned.json')
    clients_path = os.path.join(base_dir, 'Website', 'mcpso_clients_cleaned.json')
    
    processed_dir = os.path.join(base_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    summary_path = os.path.join(processed_dir, 'dataset_summary.txt')
    servers_out_csv = os.path.join(processed_dir, 'servers_dataframe.csv')
    clients_out_csv = os.path.join(processed_dir, 'clients_dataframe.csv')
    
    # 1, 2, 3. Load datasets into pandas DataFrames
    df_servers = load_data(servers_path)
    df_clients = load_data(clients_path)
    
    print("Data loaded successfully. Performing analysis...")
    
    # 4, 5, 6, 8. Perform analysis and generate summary statistics
    with open(summary_path, 'w', encoding='utf-8') as f_out:
        f_out.write("====== MCPCorpus Dataset Exploration Report ======\n\n")
        analyze_dataset(df_servers, "MCP Servers", f_out)
        analyze_dataset(df_clients, "MCP Clients", f_out)
        print(f"Summary statistics report saved to {summary_path}")

    # 7. Save structured datasets to CSV (using index=False to omit pandas index)
    # Pandas to_csv handles dictionaries (e.g., github column) by stringifying them.
    df_servers.to_csv(servers_out_csv, index=False)
    print(f"Servers dataset saved to {servers_out_csv}")
    
    df_clients.to_csv(clients_out_csv, index=False)
    print(f"Clients dataset saved to {clients_out_csv}")
    
    print("Dataset exploration completed successfully.")

if __name__ == "__main__":
    main()
