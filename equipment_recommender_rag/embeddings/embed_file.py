import pandas

from equipment_recommender_rag.embeddings.create_embeddings import (
    create_embedding_openai,
)
from equipment_recommender_rag.utils.highlighted_output import print_highlight


# Takes in dataframe and columns that need to be embedded
# Creates extra columns with COLUMN_NAME_embeddings and return full df
def embed_df(df: pandas.DataFrame, columns_embeddings: list[str]) -> pandas.DataFrame:

    for column_name in columns_embeddings:
        print_highlight(f"Creating Embeddings for {column_name}!")
        df[f"{column_name}_embeddings"] = df[column_name].apply(create_embedding_openai)

    return df


# Takes in file (.csv) and outputs the same file including embeddings in extra columns
def store_embedded_file(
    file_path: str, columns_embeddings: list[str], output_path: str
):
    df = pandas.read_csv(file_path)

    print_highlight(f"Reading of '{file_path}' successful")

    df_embedded = embed_df(df=df, columns_embeddings=columns_embeddings)

    print_highlight("Embedding succesful")

    df_embedded.to_csv(output_path, index=False)

    print_highlight(f"Results stored in '{output_path}'")
