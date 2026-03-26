from equipment_recommender_rag.embeddings.embed_file import store_embedded_file
from equipment_recommender_rag.retrieval.dense_retrieval import dense_retrieval_query

CREATE_EMBEDDING = False
RETRIEVE_TOP_K = True

if __name__ == "__main__":
    if CREATE_EMBEDDING:
        store_embedded_file(
            file_path="data/raw/equipment_database_sample.csv",
            columns_embeddings=[
                "Short description (2-3 sentences)",
                "Typical applications (3+ bullets)",
            ],
            output_path="data/processed/equipment_database_sample_output.csv",
        )

    if RETRIEVE_TOP_K:
        dense_retrieval_query(
            query="What instrument scans a sample with a focused electron beam to generate signals that reveal surface topography and composition?",
            embedding_database_path="data/processed/equipment_database_sample_output.csv",
            column_name="Short description (2-3 sentences)",
        )
