import ast
import pandas as pd

from equipment_recommender_rag.embeddings.create_embeddings import (
    create_embedding_openai,
)

def dense_retrieval_query(
    query: str, embedding_database_path: str, column_name: str, k: int = 5
):

    # Create embedding of the query (this MUST be the same embedder as the one used for the database)
    query_embed = create_embedding_openai(query)

    # Read database
    df = pd.read_csv(embedding_database_path)

    # Convert embedding to float instead of str
    df[f"{column_name}_embeddings"] = df[f"{column_name}_embeddings"].apply(
        ast.literal_eval
    )

    print(df.head())
