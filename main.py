from equipment_recommender_rag.embeddings.embed_file import store_embedded_file
import pandas as pd

if __name__ == "__main__":
    
    store_embedded_file(
        file_path="data/raw/equipment_database_sample.csv",
        columns_embeddings=[
            "Short description (2-3 sentences)",
            "Typical applications (3+ bullets)",
        ],
        output_path="data/processed/equipment_database_sample_output.csv",
    )
