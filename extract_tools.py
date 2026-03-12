import pandas as pd
import json
import os

def load_json(filepath):
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.lower().strip()

def process_servers(servers):
    records = []
    
    print("Processing servers and extracting tools...")
    for server in servers:
        server_name = server.get('name', '')
        server_description = server.get('description', '')
        # Clean up none types
        server_description = server_description if server_description is not None else ""
        
        # Tools are sometimes strings representing JSON arrays
        tools_str = server.get('tools')
        
        tools_list = []
        if isinstance(tools_str, str) and tools_str.startswith('['):
            try:
                tools_list = json.loads(tools_str)
            except json.JSONDecodeError:
                pass
        elif isinstance(tools_str, list):
            tools_list = tools_str
            
        # Iterate and create a record for each tool
        for tool in tools_list:
            if not isinstance(tool, dict):
                continue
                
            tool_name = tool.get('name', '')
            tool_description = tool.get('description', '')
            tool_commands = tool.get('commands', '') # Sometimes commands might not exist
            
            # Clean up none types
            tool_name = tool_name if tool_name is not None else ""
            tool_description = tool_description if tool_description is not None else ""
            tool_commands = tool_commands if tool_commands is not None else ""
            
            # Create combined text
            combined_text = f"{server_description} {tool_description} {tool_commands}".strip()
            
            record = {
                'server_name': server_name,
                'server_description': server_description,
                'tool_name': tool_name,
                'tool_description': tool_description,
                'tool_commands': tool_commands,
                'combined_text': combined_text
            }
            records.append(record)
            
    return records

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    servers_path = os.path.join(base_dir, 'Website', 'mcpso_servers_cleaned.json')
    processed_dir = os.path.join(base_dir, 'processed')
    
    os.makedirs(processed_dir, exist_ok=True)
    out_csv = os.path.join(processed_dir, 'mcp_tools_dataset.csv')
    
    # 1. Load data
    servers_data = load_json(servers_path)
    
    # 2 & 3 & 4 & 5. Extract and format records
    records = process_servers(servers_data)
    
    # 6. Convert to DataFrame
    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} tool records before cleaning.")
    
    # 7. Clean dataset
    # Convert text to lowercase
    text_columns = ['server_name', 'server_description', 'tool_name', 'tool_description', 'tool_commands', 'combined_text']
    for col in text_columns:
        df[col] = df[col].apply(clean_text)
        
    # Remove empty descriptions
    df = df[df['tool_description'].str.len() > 0]
    
    # Remove duplicates
    # A tool from the same server with the same tool name and description is a duplicate
    df = df.drop_duplicates(subset=['server_name', 'tool_name', 'tool_description'])
    
    print(f"Dataset contains {len(df)} tool records after cleaning.")
    
    # 8. Save to CSV
    df.to_csv(out_csv, index=False)
    print(f"Saved processed dataset to {out_csv}")
    
    # 9. Print useful statistics
    total_servers_processed = df['server_name'].nunique()
    total_tools_extracted = len(df)
    unique_tool_names = df['tool_name'].nunique()
    
    print("\n--- Summary Statistics ---")
    print(f"Total MCP Servers Processed: {total_servers_processed}")
    print(f"Total Tools Extracted: {total_tools_extracted}")
    print(f"Number of Unique Tool Names: {unique_tool_names}")
    
    print("\nSample Preview:")
    print(df.head())

if __name__ == "__main__":
    main()
