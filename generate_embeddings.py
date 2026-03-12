import pandas as pd
import numpy as np
import os
import time
from sentence_transformers import SentenceTransformer

def load_dataset(filepath):
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Handle missing text safely by filling NaNs with empty strings
    df['combined_text'] = df['combined_text'].fillna('')
    return df

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    print(f"Loading SentenceTransformer model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(texts)} texts...")
    start_time = time.time()
    
    # Generate embeddings
    # Using encode with an appropriate batch size, progress_bar=True
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Embedding generation completed in {runtime:.2f} seconds.")
    
    return embeddings, model.get_sentence_embedding_dimension(), runtime

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, 'processed')
    
    input_csv = os.path.join(processed_dir, 'mcp_tools_dataset.csv')
    output_npy = os.path.join(processed_dir, 'mcp_embeddings.npy')
    output_csv = os.path.join(processed_dir, 'mcp_embedding_dataset.csv')
    
    # 1 & 2. Load dataset and handle missing text
    df = load_dataset(input_csv)
    texts_to_embed = df['combined_text'].tolist()
    
    # 3 & 4. Generate embeddings
    embeddings, emb_dim, runtime = generate_embeddings(texts_to_embed)
    
    # 5 & 6. Attach embeddings to dataset and save
    print(f"Saving embeddings to {output_npy}...")
    np.save(output_npy, embeddings)
    
    # Add embedding vectors as lists/strings in the DataFrame and save required columns
    print(f"Constructing CSV with embeddings...")
    # Store the vector directly 
    df['embedding_vector'] = list(embeddings)
    
    # Keep only specified columns
    out_df = df[['server_name', 'tool_name', 'combined_text', 'embedding_vector']]
    
    print(f"Saving dataset with embeddings to {output_csv}...")
    out_df.to_csv(output_csv, index=False)
    
    # 7. Print useful statistics
    print("\n--- Summary Statistics ---")
    print(f"Total Tools Processed: {len(df)}")
    print(f"Embedding Dimension: {emb_dim}")
    print(f"Runtime Summary: {runtime:.2f} seconds")
    
    if len(embeddings) > 0:
        print("\nExample Embedding Vector (first 5 dimensions of first record):")
        print(embeddings[0][:5])
    
    print("\nDataset with embeddings generation completed successfully.")

if __name__ == "__main__":
    main()
